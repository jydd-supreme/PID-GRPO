# 安全设计

## 阈值和状态

`thresholds.py` 是阈值的唯一来源。配置必须满足 `0 < d_max < 1` 和 `0 < safe_ratio < warn_ratio < 1`，基线必须有限且大于 0。

- SAFE：`g >= g_safe`
- BUFFER：`g_warn < g < g_safe`
- WARNING：`g_hard < g <= g_warn`
- HARD：`g <= g_hard`
- INVALID：失败、NaN、非数值或样本不足

## Guard 优先级

SafetyGuard 始终检查未经 EMA 的 raw score。HARD 返回 `allow_training=false, hard_violation=true`；INVALID 也禁止继续，但与能力已确认越界区分，`hard_violation=false`。两者均可按照配置请求回滚。这样既不会让 EMA 掩盖一次 Hard 越界，也不会把评测失败默认为安全。

## Checkpoint 与回滚

每条记录包含 optimizer/control step、能力、独立 domain validation score、安全认证、创建时间和状态文件位置。`latest_safe` 只看已认证记录；`best_feasible` 先过滤安全记录，再比较 domain validation score。能力评测恢复不等于原始能力已恢复，恢复后仍须重新评测。

恢复前强制检查 `safety_verified`。当前单 GPU 后端先在临时目录写入协议状态、
模型或 LoRA、tokenizer、optimizer、scheduler、TrainerState 和 RNG，再原子发布目录；
共享/远程文件系统及分布式训练仍需重新验证原子性和 barrier。

## 保证边界

这是离散测量保护机制，而不是逐 token、逐 micro-batch 或连续时间的硬安全证明。两个评测点之间的瞬时退化可能无法观测。
