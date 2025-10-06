# Simple_RLHF_DPO

English / 中文（中英双语说明）

## Overview

EN: A minimal implementation of a simple RLHF-like workflow with Direct Preference Optimization (DPO).
This repository contains small example scripts and helper code to train a generator model (`train_gen.py`) and
to run DPO-style preference training (`train_dpo.py`) based on generated choice/reject pairs.

ZH: 本仓库提供了一个简化的 RLHF 风格工作流示例，包含 Direct Preference Optimization (DPO) 的训练代码。
包含训练生成器模型的脚本 (`train_gen.py`) 以及使用生成器输出进行偏好训练的脚本 (`train_dpo.py`)，
以及一些辅助模块（`common.py`、`dpo.py`）。

## Files

- `common.py` - tokenizer、ModelGEN、generate 等基础工具和模型定义（notebook 中抽取并整合）。
- `train_gen.py` - 生成器（gen）模型训练脚本（从 notebook `1.train_gen.ipynb` 转换）。
- `train_dpo.py` - DPO 训练脚本（从 notebook `2.train_dpo.ipynb` 转换）。
- `dpo.py` - 封装的 DPO 实现（含 `DPO.compute_loss`）。
- `gen.model`, `dpo.model` - 训练脚本会分别输出/读取这些文件（二进制 PyTorch 保存）。
- Notebooks: `1.train_gen.ipynb`, `2.train_dpo.ipynb`, `3.test.ipynb`, `common.ipynb`（原始 notebook，供参考）。

## Requirements / 环境依赖

EN: This project requires Python and PyTorch. Additionally, some utilities rely on `transformers` and other packages.
Because transformers and scikit-learn may include precompiled binaries, you may encounter binary-compatibility
errors (e.g. "numpy.dtype size changed"). If that happens, create a clean virtual environment and install compatible
package versions.

ZH: 本项目需要 Python 与 PyTorch；另外部分工具使用 `transformers`。由于这些库包含编译后二进制，
可能会遇到类似 “numpy.dtype size changed” 的二进制兼容性错误。建议在干净的虚拟环境中安装依赖。

Suggested quick setup (example):

```bash
# create and activate venv
python -m venv .venv
source .venv/bin/activate

# install essentials (adjust versions for your platform/torch)
pip install --upgrade pip
pip install torch torchvision torchaudio   # follow https://pytorch.org/ for platform-specific command
pip install transformers numpy scipy scikit-learn
```

Notes:
- If you see an error like "ValueError: numpy.dtype size changed, may indicate binary incompatibility",
  it's typically a mismatch between installed numpy and some compiled extension (scikit-learn, transformers
  dependencies, etc.). Recreate the virtualenv and reinstall pinned package versions or use conda which
  often handles binary dependencies more robustly.

## Quick start / 快速开始

EN: Train generator (toy example):

```bash
# run generator training (short demo)
python train_gen.py
```

This will train a small generator model and save it as `gen.model`.

EN: Train DPO (requires `gen.model`):

```bash
# run DPO training, expects a trained gen.model in the working directory
python train_dpo.py
```

ZH: 训练生成器：

```bash
python train_gen.py
```

这会训练一个小模型并把权重保存为 `gen.model`。

ZH: 运行 DPO（需要已存在 `gen.model`）：

```bash
python train_dpo.py
```

运行后会把最终模型保存为 `dpo.model`。

## Implementation notes / 实现说明

- Data generation: `Tokenizer.get_data` 在 `common.py` 中生成简单的问答对（字符串 -> token id 列表），形式为 S <answer> = <question> E。
- `train_gen.py` 的训练目标是 next-token prediction（CrossEntropyLoss，忽略 pad id）；
- `train_dpo.py`（以及 `dpo.py`）采用 DPO 风格的 loss：比较生成器与 reference 在 choice/reject 上的 log-prob 差值并构造 loss。

## Troubleshooting / 常见问题排查

- Import-time binary errors (e.g. numpy dtype size changed):
  - Recreate virtualenv, pin numpy and related binary packages, or use conda.
  - Example: `conda create -n simple_dpo python=3.10` then `conda install pytorch -c pytorch` and `pip install transformers`.
- If training crashes due to missing '=' token in generated sequence: the tokenization/data code assumes a specific format; ensure `Tokenizer` is not modified.
- If GPU memory is insufficient: reduce batch size or move model to CPU for debugging.

## Next steps / 建议的改进

- Add a `requirements.txt` with pinned versions for reproducibility.
- Add unit tests for `get_batch_data` / `get_prob_log` to check masking and shapes.
- Allow command-line args for batch size, learning rate, number of steps.

## License / 联系

EN: No license specified. Use as reference.
ZH: 未指定许可证，仅供参考与学习使用。
# DPO方法训练大语言模型,简易实现代码

环境信息:

python=3.10

torch==2.1.0(cuda)

transformers==4.34.0

datasets==2.14.5

trl==0.7.2

视频课程:https://www.bilibili.com/video/BV1Fa4y1X7xh
