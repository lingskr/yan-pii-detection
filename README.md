# PII 烟箱知识提取


## 推理

推理过程详见存储库里，概述了使用训练模型预测和处理测试数据的过程。

## 目录

- [推理](#inferencing)
- [目录](#table-of-contents)
- [设置](#setup)
- [硬件](#hardware)
- [软件](#software)
- [依赖项](#dependencies)
- [数据集](#datasets)
- [训练](#training)

## 设置

### 硬件

- **实例**：Ubuntu 20.04.5 LTS（128 GB 启动盘）
- **CPU**：Intel(R) Xeon(R) Silver 4216 @ 2.10GHz（7 个 vCPU）
- **GPU**：8 x NVIDIA A100 40GB

### 软件

- **Python**：3.10.13
- **CUDA**：12.1

### 依赖项

克隆存储库并安装所需的 Python 包：

```shell
git clone https://github.com/lingskr/yan-pii-detection.git
cd pii-detection-main
pip install -r requirements.txt
```

### 数据集

数据集需要根据PaddleOCR识别的结构转成json格式

注意：该脚本在父目录中创建一个“data”文件夹并在那里下载外部数据集。

## 训练

解决方案涉及五个 Deberta-v3-large 模型，结合了不同的架构以实现多样性和性能。以下是一些变体及其训练命令：

- 多样本 Dropout 自定义模型：提高训练稳定性和性能。

```shell
python train_multi_dropouts.py
```

- BiLSTM 层自定义模型：添加 BiLSTM 层以增强特征提取，包括特定初始化以防止 NaN 丢失问题。

```shell
python train_bilstm.py
```

- 知识蒸馏：利用表现良好的模型作为教师来提升学生模型的性能，利用不同的数据集。它需要一个教师模型。我们使用了最好的多样本 dropout 模型。
注意：它需要一个教师模型来进行蒸馏。我们使用了最好的多样本 dropout 模型。

```shell
python train_distil.py
```

- 实验 073：使用名称交换的增强数据。

```shell
python train_exp073.py
```

- 实验 076：在训练数据中引入随机添加的重要名称。

```shell
python train_exp076.py
```