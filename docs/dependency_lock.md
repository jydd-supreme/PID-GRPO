# 训练栈锁定说明

目标基础镜像：Ubuntu 22.04、Python 3.12、PyTorch 2.5.1+cu124、CUDA 12.4。
PyTorch 与 torchvision 由镜像提供，不允许从 PyPI 覆盖；安装前后均应运行
`scripts/verify_environment.py`。

## 顶层版本与源码提交

| package | version | upstream tag commit |
|---|---:|---|
| Transformers | 4.49.0 | `a22a4378d97d06b7a1d9abad6e0086d30fdea199` |
| TRL | 0.15.2 | `fc2b041b58f6fbe766dceaec819bc5a8f9d209da` |
| PEFT | 0.14.0 | `de88c703065fdd3a05521da1054c0463d16ea33c` |
| Datasets | 3.3.2 | `b37230c09f29aea04799ed4a833f91ccf779a186` |
| Accelerate | 1.4.0 | `b431d1f02733dfcb1668c7f3cffe1e2f25fb94e7` |

这些版本全部声明支持 Python 3.12。TRL 0.15.2 的发行元数据要求
Transformers >=4.46、Datasets >=2.21、Accelerate >=0.34；本项目已在目标镜像上
验证 import，并验证 GRPO loss 从 `trainer.beta` 读取 KL 系数。

默认 Qwen 模型也固定在官方仓库 commit
`7ae557604adf67be50417f59c2c2f167def9a775`，配置中不得改回漂移的 `main`。

## 安装

基础镜像中必须已经存在正确的 CUDA PyTorch，然后执行：

```bash
python3 scripts/verify_environment.py --base-only
PIP_CACHE_DIR=/root/autodl-tmp/GRPO/.cache/pip \
  python3 -m pip install --no-deps --require-hashes -r requirements-training.lock
python3 scripts/verify_environment.py
```

`--no-deps` 是有意的：lockfile 已完整列出非 PyTorch 依赖，而
`constraints-runtime.txt` 描述不可被替换的镜像层。不要直接执行无约束的
`pip install trl`。
