# Redo-GPT2

参考内容 Zero to Heros 系列 (复现GPT-2)
这个仓库是一个从基础神经网络到 GPT-2 复现的学习项目。代码主要跟着课程和实验一步一步写，重点不是做一个完整产品，而是把语言模型的关键部分自己实现一遍。

## 仓库内容

- `P1`：micrograd 相关练习，用来理解自动求导和反向传播。
- `P2-P6`：makemore 系列 notebook，用简单语言模型逐步学习 embedding、MLP、BatchNorm、训练流程等内容。
- `P7-P10`：字符级 Transformer 训练代码，使用 `input.txt` 做小规模文本生成实验。
- `redoGPT`：更接近 GPT-2 的实现，包括模型结构、训练脚本、数据下载和分词脚本。

## 主要代码

- `P7-P10/train.py`：一个小型 Transformer 语言模型，从字符级数据开始训练。
- `redoGPT/train_gpt2.py`：GPT-2 风格模型训练脚本，支持 DDP 分布式训练。
- `redoGPT/my_train_gpt.py`：使用 FineWeb-Edu 分片数据训练 GPT 模型。
- `redoGPT/get.py`：下载并处理 FineWeb-Edu 10B token 数据。
- `redoGPT/get_1bt_mirror.py`：使用 Hugging Face 镜像下载并处理 1B token 数据。
- `redoGPT/test.py`：检查 PyTorch 和 transformers 环境是否可用。

## 需要的环境

项目主要使用 Python 和 PyTorch。常用依赖包括：

- `torch`
- `transformers`
- `tiktoken`
- `datasets`
- `numpy`
- `tqdm`

## 简单运行

先安装依赖：

```bash
pip install torch transformers tiktoken datasets numpy tqdm
```

运行环境检查：

```bash
python redoGPT/test.py
```

运行字符级 Transformer：

```bash
cd P7-P10
python train.py
```

如果要训练 `redoGPT` 里的 GPT 模型，需要先准备对应的数据文件。数据脚本会下载和分词，文件较大，运行前要确认磁盘和网络都够用。

## 说明

这个项目更适合作为学习记录。先从简单模型开始，再逐步写到 Transformer 和 GPT-2 结构。
