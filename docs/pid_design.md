# PID 与 Risk Gate 设计

能力经 EMA 得到 `g_hat_t`。PID 的能力量统一用 Warning 到 Hard
区间 `s = g_warn - g_hard` 归一化，避免 benchmark 使用 0–1 或 0–100
量表时改变控制器增益：

```text
e_t = (g_warn - g_hat_t) / s # 正负误差；deadband 也使用该无量纲单位
P_t = Kp * e_t
I_t = Ki * clamp(sum(max(e, 0)), I_min, I_max) # 仅 Warn 以下累积
D_t = Kd * EMA_D((g_hat_(t-1) - g_hat_t) / s)
u_t = w * (max(P_t, 0) + I_t) + (1-w) * min(P_t, 0) + D_t
beta_(t+1) = clip(beta_t * exp(u_t))
```

因此 `e<0` 表示 EMA 高于 Warn，`e=0` 表示到达 Warn，`e=1` 表示到达
Hard。`Kp`、`Ki`、`Kd`、`deadband`、积分状态及积分边界均不再依赖原始
能力分数的量表。PID checkpoint 状态版本 3 使用此控制语义；旧版状态会被拒绝，
不能把旧的带单位积分状态直接恢复到新控制器。

Risk Gate 在 `g_safe` 为 0，在 `g_warn` 默认为 0.8，在 `g_hard` 为 1；两段均为线性插值，因而连续、有界，且能力下降时单调不减。D 项不乘 Gate，使安全区内突然下降仍可能提前制动。

积分有上下界。误差非正时积分按 `safe_integral_leak` 泄放，beta 已到上限时停止继续积分。Buffer 区的负 P 会逐步释放 KL；若能力下降太快，正 D 可提前抵消释放甚至制动。`safe_release_rate` 限制每次评测最多降低的 beta 比例；`released_kl` 记录实际下降。warm-up 仅暂停 I/D 控制，P 从第一次评测就生效。

单周期倍率限制先约束到 `[beta/max_multiplier, beta*max_multiplier]`，并对下降另设 `safe_release_rate` 限速，随后施加全局 `[beta_min, beta_max]`。回退恢复时暂设 beta 下限，连续两次 raw 能力不低于 Warn 后解除。No-KL 不实例化 PID，也不进入乘性更新。

控制目标就是 Warn，没有在 Warn 上方另加余量；Hard 是独立停训与回退线。上述默认增益是无量纲的保守初值，并非理论保证的最优值，也尚未用真实训练验证“提高 beta 必然恢复能力”。

## 其余超参数的尺度审查

- `safe_ratio`、`warn_ratio`、`max_degradation_ratio`、Risk Gate 权重、EMA
  系数、泄放率、beta 倍率和 beta 边界本身都是比例或无量纲量，不受能力量表影响。
- `integral_min/max` 与 `deadband` 现在使用归一化误差单位；其现有数值仍是待实验
  整定的初值，而不是理论最优值。
- `Kp/Ki/Kd` 虽已消除分数量表依赖，仍依赖真实训练对象、评测噪声和控制采样周期。
  特别是改变 `eval_interval_optimizer_steps` 后应重新整定 `Ki/Kd`。
- `min_eval_samples=1` 只适合 Mock；真实准确率评测必须根据置信区间和 Warning–Hard
  区间宽度设置样本量，否则测量噪声可能大于整个安全区间。
