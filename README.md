# PID-GRPO

本项目在 GRPO 后训练中，根据独立通用能力评测，以风险感知 PID 动态调节
KL 正则系数 `beta`。当前版本已经接通锁定版本的 TRL 单 GPU 真实训练路径、
离散安全评测、动态 beta、完整训练 checkpoint 和回滚复测；它仍不声称 PID
已经在真实 benchmark 上被证明有效。

## 固定运行环境

- Ubuntu 22.04、Python 3.12.x
- PyTorch `2.5.1+cu124`、CUDA 12.4（由 AutoDL 镜像提供）
- Transformers `4.49.0`
- TRL `0.15.2`
- PEFT `0.14.0`
- Datasets `3.3.2`
- Accelerate `1.4.0`

完整传递依赖、wheel SHA256、官方 tag commit 和安装方式见
[`requirements-training.lock`](requirements-training.lock) 与
[`docs/dependency_lock.md`](docs/dependency_lock.md)。

模型、Hub、Datasets、Torch、Triton、CUDA JIT、pip、临时文件、训练输出和
checkpoint 均被显式路由到 `/root/autodl-tmp/GRPO`。训练前会确认数据盘至少
还有 20 GiB 可用，并拒绝指向数据盘之外的 data/cache/output 路径。

```bash
python3 scripts/verify_environment.py --base-only
PIP_CACHE_DIR=/root/autodl-tmp/GRPO/.cache/pip \
  python3 -m pip install --no-deps --require-hashes -r requirements-training.lock
python3 scripts/verify_environment.py
```

## 数据准备

按需读取 MMLU、ARC-Challenge、HellaSwag、MATH-500、GSM8K 和 IFEval 的
新评测入口见 [在线评测说明](docs/online_evaluation.md)。它使用固定版本与样本 manifest，
无需预下载完整评测数据，且可通过 `configs/online_pid_eval.yaml` 显式接入 PID 反馈。
已按批准范围新增 DeepMath 的 Math-Verify 奖励。完整的
`DeepMath → Qwen2.5-1.5B → 每 100 次更新评测 → PID 调节 beta` 实验入口和
三组对照配置见 [服务器实验流程](docs/experiment_workflow.md)。

真实配置需要四个互斥 JSONL 文件：

```text
data/train/train.jsonl
data/control/control.jsonl
data/domain_validation/domain_validation.jsonl
data/final_test/final_test.jsonl
```

每行至少包含：

```json
{"id":"unique-id","prompt":"Question text","answer":"42"}
```

四个文件的 `id` 必须全局唯一。`D_control` 只负责安全反馈，domain validation
只负责安全 checkpoint 之间选模，final test 只在训练与选模结束后评测。

## 运行

默认入口仍受保护，不会误启动真实训练：

```bash
python3 scripts/train.py
```

填好数据并审定 benchmark、样本数与 PID 参数后，必须同时启用配置和命令行开关：

```bash
python3 scripts/train.py \
  --config configs/real_pid_kl.yaml \
  --enable-real-training
```

默认模型 `Qwen/Qwen2.5-0.5B-Instruct` 已固定到不可变 model commit。真实配置
要求至少 1000 个 control 样本，但这不是统计充分性的保证；正式实验仍须根据能力
阈值宽度、方差和置信区间重新计算样本量。

## 控制与恢复闭环

1. 训练前在 `D_control` 上测量基线 `g0`，生成 Safe/Warn/Hard 阈值。
2. Transformers `on_step_end` 只在真实 `optimizer.step()` 后触发；gradient
   accumulation micro-batch 不会推进控制计数。
3. 每 `K` 个 optimizer update 评测 domain validation 和 `D_control`。
4. raw score 先经过 SafetyGuard；异常、超时、无效结果和 Hard 都不会进入 PID。
5. `pid_kl` 更新 TRL loss 实际读取的 `trainer.beta`；另外两种模式保持不变量。
6. 安全点原子保存模型/LoRA、tokenizer、optimizer、scheduler、TrainerState、
   Python/NumPy/CPU/CUDA RNG、PID、Guard 和控制循环状态。
7. Hard/Invalid 时停止当前尝试，回退到当前分支最近的 Warn 以上 checkpoint 并重新评测；通过后以更强 KL、更低学习率、更密集评测继续剩余步数，重试次数有上限。
8. 最后一步即使不在 `K` 的整数倍也强制评测；完成后分别导出 `last_model` 和按 domain validation 选出的 `best_model`，final test 只评测后者。选择信息写入 `model_selection.json`。

当前真实入口明确限制为单 GPU。DDP/FSDP/DeepSpeed 下的 beta 广播、rank barrier
和分布式恢复尚未验证，因此会 fail fast。

## 安全语义

```text
g_safe = (1 - safe_ratio * d_max) * g0
g_warn = (1 - warn_ratio * d_max) * g0
g_hard = (1 - d_max) * g0
```

真实配置 `d_max=5%`，默认 Safe/Warn/Hard 分别为基线的 97.5%/96%/95%。
Warn 本身就是 Hard 上方的余量，不再叠加另一个目标余量。PID 使用
Warning–Hard 区间归一化后的 EMA 正负误差：高于 Warn 时 P 可释放 KL，
低于 Warn 时 P 加强 KL；I 只在低于 Warn 时累积，D 可提前响应下降趋势。
beta 按乘性更新并受全局边界和单次变化率限制。离散评测只能约束被测时刻，
不能证明两个评测点之间从未短暂越界，也不能保证一定收敛到 Warn。

## 验证

```bash
python3 -m compileall -q src scripts tests
python3 -m pytest -q
python3 scripts/inspect_kl_path.py
```

测试包含：PID/Guard/rollback 单元测试、真实 TRL `compute_loss` 的 beta 数值测试、
tensor checkpoint 往返，以及一个无需下载模型的微型 GRPO optimizer-step 闭环。

## 尚未完成

- 准备和冻结真实四路数据，并做题面哈希、近重复与来源级泄漏检查。
- 选定能力 benchmark，预注册统计功效、失败重试、`K`、退化预算和 PID 参数。
- 完成真实 Qwen 小规模试跑、显存/吞吐测量和中断恢复演练。
- 加入 DDP/FSDP/DeepSpeed 集成测试后再开放多卡训练。
- 当前数学奖励只做保守的字符串/数值 exact match，正式数学实验应接符号 verifier。

详见 [`docs/architecture.md`](docs/architecture.md)、
[`docs/experiment_protocol.md`](docs/experiment_protocol.md) 和
[`docs/dependency_lock.md`](docs/dependency_lock.md)。
