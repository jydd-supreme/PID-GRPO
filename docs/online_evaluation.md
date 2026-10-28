# 流式数据与在线评测

此配置对应所提供的数据集表。数据由 Hugging Face 在线流式读取，模型推理在训练服务器执行；
没有将模型交给第三方评测服务，也不需要预下载完整数据集或生成评测 JSONL。
流式读取仍然产生网络传输，HF 会缓存少量元数据，远程 Parquet 也可能读取整个行组。
训练进程第一次成功读取各来源后，会把已选中的固定题目保存在数据盘缓存中；
后续续训先核对缓存内容与冻结 manifest，再复用已选题目，不重新下载完整数据集。
manifest 本身仍只保存样本 ID、内容哈希和协议哈希。
实现依据为 [HF streaming 文档](https://huggingface.co/docs/datasets/stream)。

## 数据角色

配置文件为 `configs/online_benchmarks.yaml`，所有数据源固定为不可变 commit。

| 角色 | 数据与数量 | 使用时机 |
| --- | --- | --- |
| train | DeepMath 20,000 | 新实验配置的数学训练池，使用已批准的 Math-Verify 奖励 |
| domain_validation | DeepMath 512，排除上述训练池 | 训练期间选 checkpoint |
| control | MMLU 500 + ARC-Challenge 250 + HellaSwag 250 | PID 周期反馈 |
| forgetting | 另取 MMLU 500 + ARC-Challenge 250 + HellaSwag 250 | 配置冻结后，比较训练前后遗忘 |
| final_math | MATH-500 全部 500 + GSM8K test 全部 1319 | 最终数学评测，分别报告 |
| instruction | IFEval 全部 541 | 最终指令遵循评测 |

MMLU 在 57 个学科间近似等额分层抽样，control 与 forgetting 各自覆盖所有学科。
ARC 从 test 抽样，HellaSwag 从有标签的 validation 抽样；同一源的两个角色互斥。
DeepMath 采用大小 1024 的缓冲随机采样，通常流式读取至少 21,536 行，而非全库无偏抽样；
只请求 question/final_answer/difficulty/topic，不载入三份长推理参考解答。
其他小型评测源会扫描评测 split，在内存中分层或蓄水池抽样；训练进程
只持久化选中的题目，不保存完整 split。

配置和种子决定固定样本；单独请求 control 时也会排除其对应的 forgetting 预留题。
固定数量的角色在抽样前按题目标准化文本哈希去重，不足目标数量时中止；
完整最终集和跨数据源的重复仍会中止。它不是语义近重复检测，
DeepMath 上游的去污染声明也不等于此项目已经完成来源级泄漏审计。
来源：[DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K)、
[MMLU](https://huggingface.co/datasets/cais/mmlu)、
[ARC](https://huggingface.co/datasets/allenai/ai2_arc)、
[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)、
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、
[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、
[IFEval](https://huggingface.co/datasets/google/IFEval)。

## 评分协议

- MMLU：零样本原始选择题提示，比较答案字母的条件 log-prob，报告 accuracy。
- ARC/HellaSwag：比较答案文本/续写的条件 log-prob，报告 acc 与按候选字符长度归一化的 acc_norm，后者进入控制。
- 数学：贪心生成，使用 Math-Verify 符号解析与等价验证，报告解析失败和截断比例。
- IFEval：调用 lm-eval 0.4.8 的规则检查器，报告 prompt/instruction 的 strict/loose 四项准确率。

生成任务使用模型 chat template；选择题不使用 chat template，以固定的零样本格式比较候选。
这是本项目固定的轻量评测协议，不声称复现 MMLU 5-shot 或完整 lm-eval 排行榜配置。
控制总分按样本数加权，实际权重为 MMLU .5、ARC .25、HellaSwag .25，保留分项。
目前 guard 只监测总分，不能保证每个学科或每个 benchmark 都不退化；分项 hard guard 需另行审批。
1000 题不构成 5% 相对退化约束的统计充分性保证。
参考评分实现：[HellaSwag](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/hellaswag/utils.py)、
[IFEval](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/ifeval/utils.py)、
[Math-Verify](https://github.com/huggingface/Math-Verify)。

## 本地提交、服务器运行

本地提交脚本、配置、源码与文档。已修正 `.gitignore` 中 `data/` 误忽略 `src/data/` 的规则；
务必将现在可见的 `src/data/__init__.py` 与 `dataset_builder.py` 一并纳入提交。
原始数据和模型不提交 Git，不需要在本地下载这七份数据集。

服务器使用锁定训练环境；新增评分器依赖列在 `requirements-evaluation.in`，它尚不是完整哈希锁。
先在克隆的环境中解析并核对与 `requirements-training.lock` 的兼容性，不能直接升级训练栈。
不使用 IFEval 的控制评测无需安装 lm-eval；只安装 Math-Verify 即可支持数学 validation。
IFEval 的 nltk 规则还需要 punkt/punkt_tab 资源，可将 NLTK_DATA 指向数据盘后准备这些小型资源。
当前机器没有完整评分器和 CUDA，因此没有声称这些扩展已经通过真实模型集成验收。

1. 不联网、不加载模型，检查角色与配置：

```bash
python scripts/online_benchmarks.py plan
```

2. 在服务器首次流式选择并冻结 control 和 validation：

```bash
python scripts/online_benchmarks.py freeze \
  --roles control domain_validation \
  --manifest benchmark_manifests/online_v1.json
```

该文件无题目正文，可以纳入版本控制。已有 manifest 不会被覆盖。
修改数据配置、抽样种子或数量时，必须建立新的 manifest 和基线，不能继续旧实验。
训练进程首次读取时按来源逐个保存已选题目；新进程优先使用持久化缓存，
校验题目内容和冻结 manifest 后再装入内存。后续 PID 评测不再请求数据源。
网络或内容校验失败不会退回随机样本或旧分数。

3. 基线评测（模型仍使用仓库的固定 Qwen revision）：

```bash
python scripts/online_benchmarks.py evaluate --roles control \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output outputs/control_baseline.json
```

4. 接入训练期间的能力反馈：

`configs/online_pid_eval.yaml` 选择 `evaluation.backend=online_suite`。
它仍从 `data.train_file` 读取已有训练 JSONL；不再要求本地 control/validation/final JSONL。
配置默认 `training.enabled=false`。准备好原有训练集、评测依赖与 manifest，
在独立运行配置中启用训练后，使用现有 `scripts/train.py --enable-real-training` 启动。
PID 公式、GRPO loss、训练 reward 与回滚逻辑没有修改；旧配置仍走原始本地 JSONL 路径。
在线配置训练结束只导出模型，不自动使用 final/forgetting/IFEval 选模或调参。

**DeepMath 新训练架构已接入。** 用户已批准新配置使用 Math-Verify 奖励，
入口为 `scripts/run_experiment.py`，配置为 `configs/experiments/deepmath_pid.yaml`。
它在线选取 train/control/validation 并在内存中构建 TRL Dataset，模型为 Qwen2.5-1.5B-Instruct，
每 100 个 update 调节 beta；具体步骤见 [实验流程](experiment_workflow.md)。
此处 `online_pid_eval.yaml` 仍保留为“原有本地训练集 + 在线评测”的兼容配置，未替换旧奖励。
DeepMath 难题仍可能导致小模型大量全错，应先确认有效奖励组比例。

5. 超参数冻结后，单独冻结最终评测身份（只抽题，不评模型）：

```bash
python scripts/online_benchmarks.py freeze \
  --roles forgetting final_math instruction \
  --manifest benchmark_manifests/final_v1.json
```

先用 base 模型生成独立 baseline，再用保存的 LoRA 对照；两次命令使用相同角色、manifest 和生成预算。
最终角色需要显式 `--allow-final`，不能在每轮调参时反复运行。

```bash
python scripts/online_benchmarks.py evaluate \
  --roles forgetting final_math instruction --allow-final \
  --manifest benchmark_manifests/final_v1.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output outputs/final_baseline.json

python scripts/online_benchmarks.py evaluate \
  --roles forgetting final_math instruction --allow-final \
  --manifest benchmark_manifests/final_v1.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --adapter /root/autodl-tmp/GRPO/outputs/online-pid-eval/best_model \
  --baseline-results outputs/final_baseline.json \
  --output outputs/final_best.json
```

结果按 benchmark 分项，并给出与 baseline 的差值和相对退化；last_model 可以同样评测。
不同运行的 output 不能覆盖。首次评测模型仍会下载模型权重，除非 --model 指向本地模型目录。
