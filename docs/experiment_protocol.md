# 后续实验协议（草案）

## 公平比较

三组实验为 `no_kl`、`fixed_kl`、`pid_kl`。除 beta 策略外，应保持初始 checkpoint、LoRA 配置、数据与顺序、随机种子、reward、GRPO 超参数、评测集、optimizer-update 预算和硬件条件一致。至少运行多个预先声明的种子，报告单次轨迹和聚合不确定性。

## 数据职责

- `D_train`：特定任务训练。
- `D_control`：只用于周期能力反馈和 Guard/PID。
- domain validation：只用于安全可行 checkpoint 之间选取 best feasible。
- `D_final_test`：在所有阈值、超参数和 checkpoint 选择冻结后仅作最终评估。

样本 ID 去重是最低限度，下一轮还需做规范化题面哈希、近重复和来源级泄漏检查。

## 预注册项

在真实运行前固定：`d_max`、阈值比例、K（optimizer updates）、最小有效样本数、失败/重试上限、PID/EMA/倍率参数、停止与回滚政策、domain 指标、capability 指标以及统计分析方法。不得基于 final test 调参。

## 结果声明

记录原始与 EMA 能力、阈值、Warning–Hard 归一化尺度、归一化误差、风险权重、P/I/D、旧/新 beta、实际 KL、domain 指标、学习率、回滚和 checkpoint 选择。Mock/synthetic 数据必须带 `data_source`，图上必须水印。能力约束只能表述为离散观测点上的经验结果，不能扩展成未测时刻的严格保证。
