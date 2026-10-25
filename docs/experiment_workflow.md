# DeepMath / Qwen2.5-1.5B 实验运行流程

流程：DeepMath 数学 GRPO → 每 100 次 optimizer update 用 MMLU/ARC-C/HellaSwag
评测 → PID 更新下一阶段 beta。训练奖励按批准范围使用 Math-Verify，GRPO loss 与 PID 公式不变。

## 本地提交、服务器运行

本地提交新代码、配置、测试、文档，以及 `.gitignore` 修复后可见的 `src/data/`。
服务器 `git pull` 后执行以下命令；不需要本地下载完整数据或模型。
主入口为 `scripts/run_experiment.py`，仅显式 train 且传入开关时启动训练。

模型固定为 Qwen2.5-1.5B-Instruct，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
数据源身份已通过官方 HF API 核对；尚未读取这七个任务的数据内容生成正式 manifest。

先按 `docs/dependency_lock.md` 安装并验证训练环境。
`requirements-evaluation.in` 是新增评分扩展，尚不是完整哈希锁；
先在复制的环境中解析并确认未改变锁定的训练栈。训练期需要 Math-Verify，
IFEval/lm-eval 可在独立最终评测环境安装。

```bash
python scripts/verify_environment.py
python -m pytest -q
```

## 1. 查看计划

```bash
python scripts/run_experiment.py plan --config configs/experiments/deepmath_pid.yaml
```

不联网、不加载模型。默认 2000 updates、K=100、G=4、batch=4、累积=8，
每 update 为 8 个题组、32 个回答。生成上限 2048 tokens；这些是待实测的起点。
显存、吞吐、零优势组比例和截断率须在服务器确认，不能据配置声称训练有效。

服务器数据盘不是 `/root/autodl-tmp/GRPO` 时，在所有运行命令中一致添加
`--root-dir /实际数据盘/GRPO`，同步重定位 cache/data/output。

## 2. 首次冻结数据身份

```bash
python scripts/run_experiment.py freeze --config configs/experiments/deepmath_pid.yaml
```

生成 `benchmark_manifests/deepmath_v1.json`：train 20,000、validation 512、control 1000。
文件仅含 ID/哈希，可以提交；题目保存在内存中，不生成数据 JSONL。
三个 KL 模式共用同一份 manifest，无需重复 freeze。换训练 seed 不改变样本成员。

## 3. 预检与工程短跑

```bash
python scripts/run_experiment.py preflight --config configs/experiments/deepmath_pid.yaml
python scripts/run_experiment.py train --config configs/experiments/deepmath_smoke.yaml --enable-real-training
```

preflight 核对环境、GPU、manifest、验证器和数据盘，不下载模型。
smoke 使用正式 K=100，在第 100 次更新测评并调节 beta，然后结束。
它验证边界接通，不能证明新 beta 改善了后续训练；需要该验证时另建 max_steps>100 的 pilot。
检查 reward_std、KL、截断率、control 分项及 checkpoint 恢复。
若 1.5B 在 DeepMath 上大量全错，应另行校准难度和样本计划，不能靠 PID 弥补无训练信号。

## 4. 三组对照

```bash
python scripts/run_experiment.py train --config configs/experiments/deepmath_no_kl.yaml --seed 42 --enable-real-training
python scripts/run_experiment.py train --config configs/experiments/deepmath_fixed.yaml --seed 42 --enable-real-training
python scripts/run_experiment.py train --config configs/experiments/deepmath_pid.yaml --seed 42 --enable-real-training
```

三组共用训练、生成和评测预算，按 mode/seed 隔离输出。主配置触发 Hard/Invalid 后停止，
不执行改变 LR/K 的恢复续训。早停必须报告失败与实际成本，不能作为完整预算结果。
可按相同配置继续 seed 43、44；已有 mode/seed 不可覆盖。
以固定预算 last 为主、validation-selected best 为辅；固定 beta=.04 是初始对照，
正式结论还需与经过同预算调参的固定 KL 比较。

## 5. 独立最终评测

按 `docs/online_evaluation.md` freeze `forgetting final_math instruction`，
再以相同 manifest 与生成预算比较 base、各组 last/best。
将该文档示例中的模型和 revision 换成上述 1.5B 版本，adapter 例如：

```text
/root/autodl-tmp/GRPO/outputs/deepmath-qwen15b/pid/seed-42/last_model
/root/autodl-tmp/GRPO/outputs/deepmath-qwen15b/pid/seed-42/best_model
```

使用 `--baseline-results` 输出各 benchmark 的差值与相对退化。
MATH-500/GSM8K 分别报告数学收益，独立 heldout 报告遗忘，IFEval 报告指令遵循。
最终集不参与 PID 或选模；1000 题 composite control 不等于总体能力保证。
