# PID 与 Risk Gate 设计

能力经 EMA 得到 `g_hat_t`。PID 的能力量统一用 Warning 到 Hard
区间 `s = g_warn - g_hard` 归一化，避免 benchmark 使用 0–1 或 0–100
量表时改变控制器增益：

```text
e_t = max(0, (g_warn - g_hat_t) / s) # deadband 也使用该无量纲单位
P_t = Kp * e_t
I_t = Ki * clamp(sum(e), I_min, I_max)
D_t = Kd * EMA_D((g_hat_(t-1) - g_hat_t) / s)
u_t = w(g_hat_t) * (P_t + I_t) + D_t
beta_(t+1) = clip(beta_t * exp(u_t))
```

因此 `e=0` 表示能力尚未低于 Warning，`e=1` 表示 EMA 能力到达
Hard。`Kp`、`Ki`、`Kd`、`deadband`、积分状态及积分边界均不再依赖原始
能力分数的量表。PID checkpoint 状态版本 2 起采用此语义；旧版状态会被拒绝，
不能把旧的带单位积分状态直接恢复到新控制器。

Risk Gate 在 `g_safe` 为 0，在 `g_warn` 默认为 0.8，在 `g_hard` 为 1；两段均为线性插值，因而连续、有界，且能力下降时单调不减。D 项不乘 Gate，使安全区内突然下降仍可能提前制动。

积分有上下界。raw safety state 为 SAFE 时积分按 `safe_integral_leak` 泄放；若控制输出不为正，beta 按 `safe_release_rate` 向 `base_beta` 平滑回归。该释放动作通过 `released_toward_base` 记录。warm-up 周期更新 EMA/PID 内部状态但冻结 beta。

单周期倍率限制先约束到 `[beta/max_multiplier, beta*max_multiplier]`，随后施加全局 `[beta_min, beta_max]`。No-KL 不实例化 PID，也不进入乘性更新。

注意：这些符号方向和稳定化机制只是可审查的控制设计，尚未用真实训练验证“提高 beta 必然恢复能力”。

## 其余超参数的尺度审查

- `safe_ratio`、`warn_ratio`、`max_degradation_ratio`、Risk Gate 权重、EMA
  系数、泄放率、beta 倍率和 beta 边界本身都是比例或无量纲量，不受能力量表影响。
- `integral_min/max` 与 `deadband` 现在使用归一化误差单位；其现有数值仍是待实验
  整定的初值，而不是理论最优值。
- `Kp/Ki/Kd` 虽已消除分数量表依赖，仍依赖真实训练对象、评测噪声和控制采样周期。
  特别是改变 `eval_interval_optimizer_steps` 后应重新整定 `Ki/Kd`。
- `min_eval_samples=1` 只适合 Mock；真实准确率评测必须根据置信区间和 Warning–Hard
  区间宽度设置样本量，否则测量噪声可能大于整个安全区间。
