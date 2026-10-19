# TRL GRPO KL 路径静态分析

> 2026-10-19 更新：项目现已固定并安装 TRL 0.15.2，`inspect_kl_path.py` 与
> `tests/test_trl_integration.py` 已分别完成源码路径和数值 loss 验证。下文保留
> 第一轮选择版本之前的调研背景。

分析日期：2026-10-16。当前环境未安装 TRL，因此没有运行或 monkey-patch 本地包；结论来自 Hugging Face 官方文档与官方 GitHub `main` 源码，下一轮必须在锁定版本后复核。

## 当前官方实现观察

官方 `GRPOConfig` 暴露 `beta`，当前文档默认值为 `0.0`。官方 `GRPOTrainer` 源码在初始化时执行等价于 `self.beta = args.beta` 的复制，loss 路径使用 `self.beta * per_token_kl`，而不是每步重新读取 `trainer.args.beta`。因此适配器更新 `trainer.beta`，并只把 `args.beta` 作为可读配置镜像同步。

当初始化 beta 为 0 时，当前官方源码不需要/不创建 reference model。由此得到重要运行约束：`no_kl` 必须永久保持 0；`pid_kl` 必须以正 beta 初始化，不能先以 no-KL 构造 Trainer 再动态切成正 beta。PEFT 场景可能通过禁用 adapter 得到 reference 行为，但仍需在锁定版本上实测。

reference model 是 KL 参照策略；old policy/生成时策略用于 importance ratio；clipping 约束 surrogate ratio；KL estimator 计算 reference 与当前策略的差异。动态 beta 只改变 KL 项权重，不会更新 reference model，也不会替代 clipping 或 old-policy 机制。

## 适配策略

`DynamicBetaAdapter` 在构造时检查 `trainer.beta` 是否存在并核对模式不变量。若未来版本删除/改名该属性，它会直接失败，要求重新审查，不会静默修改 `args.beta`。`scripts/inspect_kl_path.py` 可对已安装版本检查三个关键源码片段，但它只是诊断工具，不构成语义证明。

计划中的真实集成优先采用外层 callback/coordinator，在真实 optimizer step 边界触发评测；如果 Hugging Face callback 无法安全访问该边界或同步 beta，再采用最小 subclass。不得重写 GRPO loss。

## 待真实环境验证

- 锁定 release/commit 后，检查 loss 的确读取 `self.beta`，并用 toy batch 对 beta 攭动做数值测试。
- 梯度累积、DeepSpeed/FSDP 和多进程下 optimizer-step callback 的准确计数与 beta 广播。
- PEFT 下 reference log-prob 的生成、缓存，以及 beta 从小正数变化时的生命周期。
- checkpoint 恢复后 beta、reference state、old-policy state、scheduler/RNG 的一致性。
- `measured_kl` 的官方日志字段和跨设备聚合语义。

官方资料：

- https://huggingface.co/docs/trl/grpo_trainer
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
