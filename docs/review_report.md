# 第一轮静态审查报告

> 历史记录：本文件描述 2026-10-16 的第一轮状态。当前 P0 实现与验证结果以
> README、`docs/dependency_lock.md` 和当前测试为准。

日期：2026-10-16；环境：Python 3.10.12。

## 完成情况

- 已实现配置继承及安全语义校验，三组 KL baseline 默认禁止训练。
- 已实现 Threshold、连续 Risk Gate、PID 稳定化和状态序列化。
- 已实现 raw-score SafetyGuard 与轻量 checkpoint/rollback 完整状态机。
- 已实现 Mock/Real evaluator 边界、数据角色隔离和本地 JSONL builder。
- 已实现第一版 MathReward、CSV logger、TensorBoard 扩展接口和绘图入口。
- 已实现当前 TRL 设计对应的动态 beta adapter，以及按 optimizer update 计数的外层闭环。
- 已提供 8 个要求的测试文件、README 和 5 份设计/协议文档。

## 验证结果

- `python3 -m compileall -q src scripts tests`：通过。
- 配置加载与核心闭环独立冒烟检查：通过。
- `pytest`：34 项通过，用时约 0.08 秒。
- `scripts/evaluate.py`：能够直接运行，并把输出标为 `source=mock`。
- `scripts/train.py`：默认明确退出；即使未来设置命令行开关，第一轮仍无真实训练后端。
- 当前环境没有 TRL；`scripts/inspect_kl_path.py` 正确报告缺失且不会安装。

测试所需 pytest 仅临时安装在 `/tmp/pid_grpo_test_deps`，没有向项目环境安装 PyTorch、TRL、模型或数据。

## 已知边界

- TRL 分析基于分析日的官方文档与 `main` 源码，而非本地锁定版本；动态 beta 仍标记为待目标版本验证。
- `RealCapabilityEvaluator` 明确未实现；Mock 不可解释为实验结果。
- checkpoint JSON 后端验证状态机和恢复协议，不保存真实模型权重或 optimizer tensor。
- 没有真实训练、benchmark、GPU 实验或控制有效性结论。
- 离散评测无法证明测量间隔内不存在短暂越界。

## 下一轮阻断项

在锁定 TRL/Transformers/PyTorch 版本、数据协议、benchmark、K、PID 参数和真实 checkpoint 后端并完成 toy integration test 前，不应启动 Mini-GRPO。
