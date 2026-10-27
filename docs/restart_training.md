# 四小时服务器会话：训练记录与断点续训

从本版本启动的训练支持跨进程重启。所有文件应放在平台明确保证重启后保留的挂载盘上，
例如当前使用的 `/mnt/afs/L202500740/GRPO`。代码不能恢复已经随容器销毁的本地磁盘。

## 默认行为

DeepMath 主配置新增 `checkpointing`：

```yaml
checkpointing:
  interval_optimizer_steps: 25
  interval_seconds: 600
  keep_last: 3
  session_max_seconds: 10800
```

- 每完成 25 次 optimizer update，或距离上次保存达到 600 秒，在完整 update 边界保存。
  这不是墙钟严格每十分钟：正在执行的单个更新需要先完成。
- 每次长评测前和评测完成后额外保存，不改变 K=100。
- 默认从命令启动起计时，约 3 小时请求暂停，预留保存和平台关机余量。
  若节点已经运行很久，应按实际剩余时长调小 `--session-max-seconds`。
- Ctrl+C/SIGTERM 请求在安全边界保存并退出；SIGKILL/突然断电无法现场保存，
  下次从最近的完整检查点恢复，需要重算之后尚未提交的更新。
- 自动保留最近三个完整 restart checkpoint。安全 rollback checkpoint 独立管理，
  不把未评测的普通 restart snapshot 认证为安全。

session 时间上限只能在 update/评测题目边界响应；单个模型加载、生成、文件写入或阻塞 I/O
不会被强制抢占，因此不能保证恰好在某秒退出。默认一小时余量也不能替代服务器短跑测时。

## 首次训练

在服务器激活与上次相同的锁定环境，完成数据 freeze/preflight 后执行：

```bash
python scripts/run_experiment.py train \
  --config configs/experiments/deepmath_pid.yaml \
  --root-dir /mnt/afs/L202500740/GRPO \
  --seed 42 \
  --session-max-seconds 10800 \
  --enable-real-training
```

训练结果为 `paused` 是正常会话暂停，与安全失败 `stopped_hard` 分开。
运行目录为：

```text
/mnt/afs/L202500740/GRPO/outputs/deepmath-qwen15b/pid/seed-42/
```

## 重启后的续训

重新激活环境、进入同一代码目录；使用相同 config/root/seed，加上 `--resume`：

```bash
python scripts/run_experiment.py train \
  --config configs/experiments/deepmath_pid.yaml \
  --root-dir /mnt/afs/L202500740/GRPO \
  --seed 42 \
  --resume \
  --session-max-seconds 10800 \
  --enable-real-training
```

每个新会话都执行这条续训命令，直到 `status=complete`。不用重新 freeze，不用增加 max_steps。
若保存到 step 725，原目标为 1000，则继续 step 726 至 1000，不重新执行 1000 步。
session 时间预算可以逐次通过 CLI 改变，它不属于冻结的训练超参数。

续训会核对完整配置与数据身份，恢复模型/LoRA、optimizer、scheduler、TrainerState、
Python/NumPy/Torch/CUDA RNG、PID/Guard、beta、控制计数、活动 checkpoint 分支和恢复倍率。
原始 step-0 baseline 始终沿用，不能用中断模型重新定义安全阈值。
采样器按 seed/epoch 重建锁定版 TRL 的排列序列，再由 Trainer 跳过已完成数据位置。
这用于恢复同一实验，不支持中途更换模型、数据、batch、梯度累积、PID 参数或总步数。

未传 `--resume` 时不会覆盖已有实验；同一目录由进程锁保护，不能同时启动两个训练任务。
已完成的运行不会再多执行一步；Hard/Invalid 安全停止或待处理 rollback 不会被 `--resume` 绕过。
如果训练在基线阶段就被打断，尚未产生 optimizer checkpoint，则重新构建相同基础模型，
通过评测缓存继续 step-0 基线，而不是从一个不明来源的模型开始训练。

## 评测阶段中断

保存点注明“该 optimizer update 已完成，但对应评测尚未完成”。
重启时先完成这一边界的评测，再进行下一步训练，避免多更新一步。
评测结果逐题落盘，并绑定冻结协议、样本 ID 和模型权重指纹；
相同权重可复用已完成题目，权重改变即失效。
如果中断发生在安全 checkpoint 已发布、但 post-evaluation snapshot 尚未保存的极短窗口，
恢复会对齐已发布的控制状态，避免对同一次评测重复调节 PID。
超时暂停不会伪装成 Invalid 能力分数，也不会触发无谓回滚。
逐题评测缓存针对新的 `online_suite` 路径；旧本地 exact-match 路径仍可能重做整次评测。

## 查看记录

```bash
cat /mnt/afs/L202500740/GRPO/outputs/deepmath-qwen15b/pid/seed-42/progress.json
tail -n 10 /mnt/afs/L202500740/GRPO/outputs/deepmath-qwen15b/pid/seed-42/sessions.jsonl
```

| 文件 | 内容 |
| --- | --- |
| `progress.json` | 最近持久化的 update、beta、控制计数、checkpoint、是否等待评测 |
| `sessions.jsonl` | 会话 ID、UTC 时间、启动/恢复/保存/暂停/错误/完成事件 |
| `restart_checkpoints/` | 最近三个完整训练快照；权重与控制状态有校验和 |
| `evaluation_progress/` | 按模型与样本身份绑定的逐题评测结果，不含完整数据集 |
| `metrics/grpo.jsonl` | session ID、update、reward/KL/长度等 TRL 日志 |
| `metrics/benchmarks.jsonl` | session ID、update、数据角色及 benchmark 分项 |
| `metrics/training.csv` | PID、beta、安全状态曲线，跨会话追加 |
| `run_result.json` | 最近返回的 paused/complete/stopped_hard 状态 |

突然关机后，日志可能保留比最近 checkpoint 更靠后的步数；这些更新会重算，
同一步号可能跨 session 出现多次。以 restart snapshot 和恢复事件界定实际有效轨迹，
不要把追加日志的最后一行当作必然可恢复的状态。

恢复只选择完成原子发布并通过校验的快照；最新快照损坏时尝试前一个完整快照。
正在写入的临时目录不能用于恢复。原子文件发布不能保证底层存储系统永不丢数据，
平台的持久化承诺仍是前提。

## 验证范围

新增单元验证覆盖中途写入失败、损坏快照回退、pending 评测恢复、避免重复 PID、
会话预算暂停、逐题缓存恢复以及跨 epoch 的采样顺序。
真实 TRL 0.15.2 / Transformers 4.49.0 的无需下载模型 CPU 集成实验覆盖全参与 LoRA，
对比连续训练和中断恢复后的最终模型参数与 PID 状态。
测试机 Torch 为 2.7 CPU 路径，测试中为旧版 Trainer 的 RNG 加载使用兼容环境变量；
这不是目标 Torch 2.5.1+CUDA 环境的验证，不应把测试用兼容变量加入服务器运行命令。
服务器仍需执行一次小模型或短跑的真实 GPU 中断恢复演练。
