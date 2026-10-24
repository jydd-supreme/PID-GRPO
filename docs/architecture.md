# 架构说明

## 分层

控制面与训练面刻意解耦。训练器只负责优化和暴露 loss 实际读取的 beta；外层控制循环只在成功的 optimizer update 后推进计数。Evaluator 产生统一结果，raw 结果先进入 SafetyGuard，只有允许继续时才进入 PID。CheckpointManager 不依赖 TRL，并以四个独立状态槽保存 PID、安全、Trainer 和 Optimizer 状态。

```text
optimizer.step()
  -> PIDGRPOControlLoop.after_optimizer_step()
     -> 到达 K 或 max_steps? -- 否 -> 返回
     -> CapabilityEvaluator(D_control)
     -> SafetyGuard(raw result)
        -> HARD/INVALID -> stop + Warn 以上的当前分支 checkpoint 回退请求
        -> SAFE/BUFFER/WARNING
           -> PIDController.update()（仅 pid_kl）
           -> DynamicBetaAdapter.set_beta()
           -> save safety-verified checkpoint

回退后复测，合格且未耗尽重试次数时，从对应 Trainer checkpoint 续训；
提高短期 KL 下限、收紧学习率和评测间隔。达到固定步数后分别导出
`last_model` 与当前有效分支中 domain validation 最优的 `best_model`。
```

## 关键接口

- `CapabilityEvaluator.evaluate(model, dataset) -> dict`：必须含 `overall_score`、`subscores`、`num_samples`、`valid`。
- `DynamicBetaAdapter`：维护 no/fixed/PID 模式不变量；修改当前 TRL loss 读取的 `trainer.beta`。
- `PIDGRPOControlLoop.after_optimizer_step`：调用契约要求每个真实 `optimizer.step()` 恰好一次。
- `CheckpointManager.save/restore/request_rollback`：原子发布 checkpoint 与状态协议。
- `HFTrainingStateIO`：保存/恢复模型或 LoRA、optimizer、scheduler、TrainerState 与 RNG。
- `PIDControlCallback`：绑定 Transformers 的真实 optimizer-step 边界，安全停止时设置 Trainer control。

## 真实接口与边界

`RealCapabilityEvaluator` 使用本地数据做确定性生成和 exact-match；真实 TRL callback
及 tensor checkpoint 后端已经接通。当前只开放单 GPU，且 evaluator/reward 仍是第一版
数学 exact-match。多卡同步与 benchmark 专用 verifier 尚未实现。
