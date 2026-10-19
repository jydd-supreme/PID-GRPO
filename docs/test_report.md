# PID-GRPO 阶段工作汇报：通用能力约束下的动态 KL 控制

**汇报日期：2026 年 10 月 19 日**

**运行环境：Ubuntu 22.04，Python 3.12.3，PyTorch 2.5.1+cu124，CUDA 12.4，NVIDIA GeForce RTX 4080 SUPER 32 GB**

## 一、研究目标与本阶段完成的工作

本项目要解决的问题是：使用 GRPO 对模型进行任务强化训练时，模型虽然可能在目标任务上获得更高奖励，但也可能逐渐偏离原模型，造成原有能力下降。固定 KL 系数只能在训练前人工设定约束强度，训练中无论模型状态如何都保持不变。系数过小可能约束不住能力退化，系数过大又会限制目标任务学习。因此，本阶段尝试把独立能力评测接入训练过程，用 PID 控制器根据能力变化动态调整 KL 系数 `beta`，并配合安全阈值、检查点和回滚机制，形成一个能够实际运行的训练闭环。

目前已经完成 P0 阶段的工程实现。每隔固定数量的 optimizer step，程序会在独立的 control 数据上评测一次模型，先用原始分数执行安全检查，再将平滑后的分数送入 PID 控制器。模型处于安全区域时正常训练；能力下降时提高 `beta`，使当前策略受到更强的参考模型约束；原始分数达到 Hard 阈值或评测结果无效时，立即停止当前训练轨迹，并恢复最近一次确认安全的 checkpoint。checkpoint 不只保存 LoRA 权重，也保存 optimizer、scheduler、TrainerState、随机数状态以及 PID 和安全控制状态，因此恢复后可以继续沿着一致的训练状态运行，而不是只把模型参数退回去。

训练环境也已经固定。当前基础镜像中的 PyTorch 和 CUDA 保持不变，训练栈锁定为 Transformers 4.49.0、TRL 0.15.2、PEFT 0.14.0、Datasets 3.3.2 和 Accelerate 1.4.0，并记录了各上游版本对应的源码 commit。所有 Python 依赖及文件哈希写入了 `requirements-training.lock`，模型 revision 和 GSM8K 数据集 revision 同样固定。模型、数据集、依赖缓存和实验输出均放在 `/root/autodl-tmp/GRPO` 数据盘，避免长时间训练占满系统盘。详细版本记录见 [dependency_lock.md](dependency_lock.md)，完整安装锁文件见 [requirements-training.lock](../requirements-training.lock)。

## 二、核心方法、闭环实现与测试情况

训练控制流程如下图所示。一次参数更新完成后，TRL callback 根据设定周期触发 control 评测；SafetyGuard 首先检查未经平滑的原始结果，防止 EMA 掩盖突然的能力下降；通过安全检查后，PID 才会计算新的 `beta` 并发布 checkpoint。若出现 Hard 或 Invalid 状态，则终止当前轨迹、恢复 latest-safe，并对恢复结果重新评测。

![P0 训练控制流程](../outputs/demo/control_flow.png)

本方法控制的是 GRPO 目标中的 KL 惩罚系数。GRPO 根据同一道题的多个回答及其相对奖励更新策略，KL 项则限制当前策略相对参考模型的偏移，可以概括为 `L = L_GRPO + beta × KL`。`beta` 越大，模型越难偏离参考模型，原有能力通常受到更强保护，但目标任务的更新空间也会变小；`beta` 越小，策略优化更自由，同时也更容易出现能力退化。本研究没有把 `beta` 固定为一个经验常数，而是把 control 集上的能力变化作为反馈信号，在训练过程中闭环调节它。

安全阈值由训练前基线 `g0` 和人为设定的最大相对退化比例 `d_max` 共同产生。当前实现还用 `safe_ratio` 和 `warn_ratio` 在基线与 Hard 底线之间划分缓冲区：

```text
Safe = g0 × (1 - safe_ratio × d_max)
Warn = g0 × (1 - warn_ratio × d_max)
Hard = g0 × (1 - d_max)

当前默认：safe_ratio = 0.5，warn_ratio = 0.8
状态划分：score ≥ Safe 为 SAFE；Warn < score < Safe 为 BUFFER；
          Hard < score ≤ Warn 为 WARNING；score ≤ Hard 为 HARD。
```

控制系统同时保留 raw 和 EMA 两种能力信号，但二者承担不同职责。raw 是本次 control 评测的原始分数，直接交给 SafetyGuard；只要 raw 达到 Hard、结果不是有限数值、有效标记缺失或者评测样本数不足，系统就不会继续依赖平滑结果，而是停止训练并按配置请求回滚。EMA 的计算为 `EMA_t = alpha × raw_t + (1-alpha) × EMA_(t-1)`，用于减少小样本评测波动对 PID 的影响。当前真实 smoke 配置继承 `alpha=0.4`。这种“raw 负责保底、EMA 负责调节”的分工，是为了避免突发能力下降被移动平均掩盖，同时又避免 `beta` 因为单次分数抖动频繁跳变。

PID 使用 Warn 到 Hard 的距离作为归一化尺度 `s = Warn-Hard`，因此同一组控制参数可以适用于 0～1 或 0～100 等不同量纲的能力分数。当前 P、I、D 的计算逻辑可以写为：

```text
e_t = max(0, (Warn - EMA_t) / s)
P_t = Kp × e_t

I_state,t = clip(I_state,t-1 + e_t, I_min, I_max)
I_t = Ki × I_state,t

trend_t = (EMA_(t-1) - EMA_t) / s
D_t = Kd × filtered(trend_t)
```

P 项反映当前能力低于 Warn 的程度，越接近 Hard，P 项越大；I 项记录风险持续时间，能力长期停留在 Warning 区间时会逐步增强约束。为防止 I 无限累积，积分状态被限制在 `[I_min, I_max]` 内；进入 Safe 区后不再累积，而是按 `safe_integral_leak` 衰减，当前每次保留一半。D 项反映下降速度：EMA 正在下降时 D 为正，可以在 P、I 启动前提前提高 `beta`；EMA 恢复时 D 为负，使约束逐步释放。D 还经过一层指数平滑，避免相邻两次评测的差值直接造成剧烈控制动作。

P 和 I 并不是直接全部加到控制输出中。系统先根据 EMA 在 Safe、Warn、Hard 之间的位置计算连续 risk gate：Safe 以上风险权重为 0，到 Warn 时增加至 0.8，到 Hard 时增加至 1。最终控制量和 `beta` 更新为：

```text
u_t = risk_gate_t × (P_t + I_t) + D_t
beta_(t+1) = clip(beta_t × exp(clip(u_t)), beta_min, beta_max)
```

使用指数形式意味着控制器调节的是 `beta` 的相对倍率，而不是每次加减固定数值。当前参数设置为 `Kp=2.0、Ki=0.2、Kd=1.0`，`beta` 被限制在 0.001～1.0 之间，单次最多扩大或缩小 2 倍，第一个控制周期只建立状态而不修改 `beta`。如果模型已经回到 Safe，且控制输出不再要求加强约束，`beta` 会按 25% 的释放速度回到基础值 0.04。这些参数是用于验证闭环稳定性和边界条件的初始工程值，还没有经过正式对照实验调优。

这套设计目前属于“单向安全制动”：P 和 I 只在 EMA 低于 Warn 后启动，D 根据下降趋势提前介入，Hard 则由独立的 raw guard 强制保护。它能够回答“能力下降时怎样逐步加强 KL 约束”，但不会主动把能力从高位推向某个设定点。若后续研究目标改为使能力稳定收敛在 Warn 上方，需要在保留 Hard guard 的同时增加带正负误差、死区和抗饱和机制的目标跟踪模式。该差异也解释了为什么当前微型 GPT-2 演示中实际改变 `beta` 的是 D 项：该演示的最低 EMA 为 0.9205425，仍略高于 Warn=0.92，所以 P 和 I 一直为 0，只有 D 对下降和恢复趋势作出了反应。

目前共完成 49 项自动化测试，结果为 `49 passed in 5.79s`，没有失败、报错或跳过项。测试覆盖了三种 KL 模式、PID 比例/积分/微分项、积分上限和安全区衰减、`beta` 的上下界及单步变化率、raw Hard 优先拦截、无效评测停止、latest-safe 与 best-feasible 选择、原子 checkpoint 发布、数据集交叉污染检查、数学答案解析、数据盘路径限制，以及真实 TRL loss 和 checkpoint 恢复。机器可读结果保存在 [pytest-results.xml](../outputs/reports/pytest-results.xml)。

这里最重要的不是单元测试数量，而是对真实训练链路的验证。第一项集成测试直接调用锁定版本 TRL 的 `GRPOTrainer.compute_loss`，确认将 `beta` 从 0.1 调至 0.2 后，实际 KL loss 随之变为两倍，说明控制器修改的不是一个没有参与计算的配置值。第二项集成测试实际运行了 GRPO optimizer step，验证 callback、能力评测、PID 更新、`trainer.beta` 和 checkpoint 可以连通。第三项测试保存并恢复模型 tensor、optimizer、scheduler、TrainerState 以及 Python、NumPy、CPU 和 CUDA 的随机数状态，确认回滚语义覆盖了完整训练状态。

为便于观察控制器内部行为，项目中还保留了一组微型 GPT-2 演示。这个演示使用随机初始化的 4,336 参数模型、合成 reward 和预先给定的能力分数，只用于检查控制链路，不属于模型效果实验。图中的 raw 是当次能力评测值，EMA 是平滑后的控制信号；risk gate 表示当前能力位置对应的风险权重；P 反映已经低于 Warn 后的即时偏差，I 累积持续风险，D 则根据能力下降或恢复的趋势提前增减约束。该例中 EMA 最低为 0.9205425，仍略高于 Warn=0.92，所以 P 和 I 始终为 0，`beta` 的变化主要由 D 项产生。这也说明当前控制器本质上是单向安全制动器，而不是主动把能力调节到某一目标值附近。

![微型 GRPO 与 PID 控制曲线](../outputs/demo/toy_training_process.png)

## 三、DeepSeek 1.5B 真实 LoRA 训练结果

完成工程链路后，本阶段使用 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 进行了真实 smoke test，模型固定在 commit `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。训练采用 LoRA，设置为 `r=16`、`alpha=32`、dropout=0.05，挂载到 attention 的 q/k/v/o projection 和 MLP 的 gate/up/down projection。训练过程中基础模型参数保持冻结，optimizer 更新的是 LoRA 参数；每完成一次 optimizer step，LoRA 参数更新一次。本轮共运行 10 个 optimizer step，batch size 为 4，每道训练题生成 4 个回答，每个回答最多生成 512 tokens，学习率为 `1e-6`，初始 KL 系数为 0.04。完整参数见 [smoke_qwen_pid.yaml](../configs/smoke_qwen_pid.yaml)。

数据来自固定 revision 的 GSM8K。训练集使用官方 train 中的 128 题；control、domain validation 和 final test 从官方 test 中按固定种子划分，数量分别为 64、16 和 16。四个子集的样本 ID 完全互斥：训练集用于计算 GRPO reward；control 用于监测训练期间的能力变化并决定安全状态；domain validation 用于在安全 checkpoint 中选择任务表现最好的模型；final test 只在训练结束且模型选择完成后使用一次。数据来源、切分数量和文件 SHA256 均记录在 [smoke_manifest.json](../data/smoke_manifest.json)。

在正式更新 LoRA 参数前，DeepSeek 模型在 64 道 control 题上的 exact-match 为 `39/64=0.609375`。本轮 baseline 在挂载 LoRA 后执行，但 PEFT 的初始 LoRA 分支对模型输出为零，因此它等价于未经过 LoRA 优化的基础模型结果。将生成上限从早期实验的 256 tokens 提高到 512 tokens 后，基线从 `11/64` 提高到 `39/64`，表明这个推理模型需要足够的思考长度；先前的低分很大程度上来自回答在给出最终答案前被截断，而不能简单解释为模型不会做题。

本次 smoke 配置暂时使用了 25% 的最大相对退化预算，由此得到：

```text
Baseline = 0.609375
Safe     = 0.533203125
Warn     = 0.487500000
Hard     = 0.457031250
```

训练期间每两步进行一次 control 和 domain validation。control 分数依次为：step 0 的 `39/64`、step 2 的 `36/64`、step 4 的 `36/64`、step 6 的 `37/64`、step 8 的 `38/64`、step 10 的 `38/64`。所有评测点均高于 Safe 阈值，因此没有触发 Warning、Hard 或回滚，EMA 也始终位于安全区域，`beta` 全程保持 0.04。这轮实验验证了真实模型、LoRA、GRPO、评测、安全 checkpoint 和选模能够共同运行，但由于能力分数没有进入控制区间，它没有提供 PID 动态干预效果的真实证据。

![DeepSeek 真实训练与 control 轨迹](../outputs/smoke-qwen-pid/metrics/control_plot.png)

训练 reward 依次为 `0.75、1.0、0.5、0.5、1.0、0.75、0.75、0.75、1.0、1.0`。其中 6 步的四个回答存在奖励差异，能够产生有效的组内 advantage；其余 4 步的回答奖励相同，该组的 GRPO advantage 为零，只剩下很小的 KL 梯度。训练期间 KL 从 0 增长到约 `2.60×10⁻⁴`，step 0、step 8 和 step 10 的 LoRA adapter 哈希均不同，证明 LoRA 参数确实发生了更新。512-token 设置下平均回答长度约为 314～476 tokens，不再像 256-token 实验那样持续撞到生成上限。

domain validation 在 step 2、4、6、8、10 上的结果分别为 `10/16、11/16、11/16、13/16、11/16`。因此 step 8 被记录为 best-feasible，step 10 被记录为 latest-safe。正常训练结束后程序恢复了 step 8，并导出为当前 `final_model`；导出 adapter 的 SHA256 与 step 8 一致、与 step 10 不同，证明选模和恢复逻辑实际生效。恢复后的模型在只使用一次的 final test 上得到 `9/16=0.5625`。由于 final test 只有 16 题，这个数字的作用是验证完整实验协议，不能当作正式 GSM8K 成绩。

本轮完整训练约耗时 26 分钟，实际显存占用约 8.9 GB，32 GB 显卡仍有约 23 GB 余量。当前主要耗时不是 LoRA 训练本身，而是频繁进行 512-token 自回归评测。也就是说，后续扩大 control 集或者缩短评测间隔时，首先需要考虑评测吞吐，而不是显存容量。

## 四、对当前结果的判断与局限

从工程角度看，P0 目标已经完成：依赖版本可复现，真实 LoRA 参数能够更新，动态 `beta` 已确认进入 TRL 的 loss，安全评测按 optimizer step 触发，危险状态不会被 EMA 隐藏，安全 checkpoint 可以保存和恢复完整状态，模型和实验数据也已转移到数据盘。但从实验结论看，目前最多只能称为“真实模型上的闭环 smoke test”，还不能证明 PID 比固定 KL 更好，也不能证明系统已经保护了模型的通用能力。

首先，当前 25% 的最大退化预算明显偏宽。以 baseline `0.609375` 计算，Hard 只有 `0.45703125`，模型要从 39 题下降到 29 题左右才会触发硬停止，所以本轮全部处于 Safe 并不意外。如果正式实验采用“通用能力最多相对下降 5%”的约束，按现有阈值比例计算，Safe、Warn、Hard 将约为 `0.59414、0.58500、0.57891`。但 64 题的分辨率是 `1/64=0.015625`：答对 38 题时为 0.59375，刚刚落到 Buffer；答对 37 题时为 0.578125，已经低于 Hard，中间几乎没有可供 PID 调节的 Warning 区间。因此，不能只把配置中的 25% 改成 5%，还必须先扩大 control 样本量或者改用更连续、带置信区间的能力指标。

其次，当前 control、domain validation 和 final test 都来自 GSM8K。它们虽然样本互斥，但衡量的仍然是同一种数学能力，不能代表语言理解、知识问答或一般推理能力，也不足以支撑“没有发生灾难性遗忘”的结论。64 题 control 和 16 题 validation/final 的统计波动都很大，step 8 的 `13/16` 与 step 10 的 `11/16` 相差两题，既可能来自参数变化，也可能受到生成随机性和小样本方差影响。目前只有一条训练轨迹和一个随机种子，也无法将训练收益与偶然波动分开。

再次，现有 PID 使用单侧误差 `max(0, (Warn-EMA)/(Warn-Hard))`。能力高于 Warn 时，P 和 I 不会主动把模型推向 Warn；进入安全区后，积分项还会衰减，`beta` 只会逐步回到基础值。这种设计适合“能力下降时刹车”，却不能保证准确率平稳收敛在 Warn 或 Hard 附近。事实上，Hard 应当作为紧急停止线，不适合作为训练目标；更合理的目标是让能力稳定在 Warn 上方并留出统计余量，同时尽可能提高目标任务表现。另外，当前正常结束会将 best-feasible 导出为 `final_model`，latest-safe 虽然仍保存在 checkpoint 目录中，但尚未以清晰的 `last_model` 和 `best_model` 两套终态产物分别导出，不利于后续做统一训练轮数的公平比较。

## 五、后续工作方向

### 1. 重新确定能力底线和控制目标

下一步应先与实验口径对齐“下降 5%”究竟指相对下降 5%，还是下降 5 个百分点，然后冻结该定义。Hard 保留为不可越过的安全底线，控制目标设置在 Warn 上方，而不是让模型逼近 Hard。control 集需要扩充到至少数百题，并报告置信区间，使 Safe、Warn 和 Hard 之间能够容纳多个可观测分数档位。在此基础上，再增加可选的目标跟踪控制模式：使用带正负号的误差，让能力高于目标时适度释放 KL 约束，低于目标时加强约束，并配合死区、积分抗饱和和单步 `beta` 变化限制，减少大幅反弹和来回振荡。现有单向安全模式仍然保留，作为对照和故障保护。

### 2. 完善终态保存和公平比较协议

训练结束时应明确保存两套模型：`last_model` 对应固定训练步数结束时的参数，`best_model` 对应只依据 domain validation 选出的最佳安全参数。不同方法的主要比较应使用相同模型、相同数据、相同随机种子、相同生成预算和相同 optimizer step 数，以 `last_model` 作为固定预算下的主结果；`best_model` 作为实际部署和早停能力的补充结果。final test 不参与选模，只在实验方案和超参数全部冻结后评测。正式对照至少包括 no-KL、fixed-KL 和 PID-KL，并运行多个随机种子，统一报告目标任务收益、control 能力变化、越界次数、回滚次数、训练时间和置信区间。

### 3. 扩大评测范围并提高实验效率

数学训练仍使用 GSM8K，但通用能力 control 应增加与训练域不同的任务，至少覆盖语言理解、常识知识和一般推理；数学 domain validation 与通用能力 control 分开报告，避免用同域成绩代替通用能力。训练侧可以保留已经验证可用的 4 generations 和 512-token 上限，同时尝试批量推理、评测缓存或 vLLM 后端，降低频繁 control 评测的时间成本。奖励方面需要记录每步四个回答、解析后的答案、reward、reward standard deviation、KL、梯度范数和生成长度，以便判断零 advantage 来自题目过易、回答一致、答案解析失败还是采样不足。

下一阶段不宜直接扩大正式训练规模。更稳妥的顺序是先完成 control 扩容、5% 阈值重算、目标跟踪控制和双终态保存，再用短训练验证控制器确实能够进入调节区且不会振荡；确认协议后，再按统一训练步数运行 no-KL、fixed-KL 和 PID-KL 的正式对照。这样得到的结果才能回答核心问题：PID 是否能在守住预定能力底线的同时，比固定 KL 获得更好的任务收益。

## 六、结果与复现材料

当前真实实验的训练前基线保存在 [baseline.json](../outputs/smoke-qwen-pid/metrics/baseline.json)，逐次 control 记录和绘图数据保存在 [training.csv](../outputs/smoke-qwen-pid/metrics/training.csv)，安全 checkpoint 及 best/latest 指针保存在 [manifest.json](../outputs/smoke-qwen-pid/checkpoints/manifest.json)。当前导出的模型位于 `outputs/smoke-qwen-pid/final_model`，其内容对应 step 8 的 best-feasible LoRA adapter。

复现时应使用新的空输出目录，避免覆盖本次成功记录。主要命令如下：

```bash
# 检查基础镜像和锁定依赖
python3 scripts/verify_environment.py

# 运行全部自动化测试
python3 -m pytest -q \
  --junitxml=outputs/reports/pytest-results.xml \
  --durations=10

# 按固定 revision 和随机种子重建数据
python3 scripts/prepare_smoke_data.py \
  --config configs/smoke_qwen_pid.yaml

# 训练前基线评测
python3 scripts/evaluate_baseline.py \
  --config configs/smoke_qwen_pid.yaml

# 10-step DeepSeek + LoRA + PID-GRPO smoke test
python3 scripts/train.py \
  --config configs/smoke_qwen_pid.yaml \
  --enable-real-training
```
