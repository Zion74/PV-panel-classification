# 智慧能源系统工程 - 光伏板积灰判断项目

这是一个基于深度学习的光伏板积灰判断项目，旨在通过图像识别技术，自动判断光伏板的积灰程度。本项目使用了 PyTorch 深度学习框架，并提供了训练、评估和预测的完整流程。

## 项目概述

随着光伏发电的普及，光伏板表面的积灰问题日益突出，严重影响发电效率。本项目旨在开发一个智能系统，通过分析光伏板图像，自动识别并判断其积灰程度，为光伏电站的运维提供数据支持。

## 主要功能

*   **图像数据预处理**：对光伏板图像进行统一尺寸调整和归一化处理。
*   **深度学习模型训练**：支持使用多种预训练模型（如 MobileNetV2, ResNet18, SimpleCNN）进行迁移学习，对积灰图像进行分类训练。
*   **模型评估**：在验证集上评估模型的性能，包括准确率、损失等指标。
*   **图像预测**：提供图形用户界面 (GUI) 进行单张或批量图像的积灰程度预测。

## 环境配置

本项目推荐使用 Anaconda 或 Miniconda 创建虚拟环境，并安装所需的依赖。

1.  **创建虚拟环境**：
    ```bash
    conda create -n pv_dust python=3.x # 推荐使用 Python 3.8 或更高版本
    conda activate pv_dust
    ```

2.  **安装依赖**：
    项目所需的所有 Python 包都列在 `requirements.txt` 文件中。请确保您的 PyTorch 版本与 `requirements.txt` 中指定的版本兼容，特别是 CUDA 版本（如果使用 GPU）。
    ```bash
    pip install -r requirements.txt
    ```
    **注意**：`torch==2.6.0+cu126` 表示 PyTorch 2.6.0 版本，支持 CUDA 12.6。请根据您的 GPU 和 CUDA 版本调整此行，或访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 获取适合您环境的安装命令。

## 数据集准备

本项目的数据集应组织在 `data/train` 和 `data/val` 目录下，每个目录下包含按类别划分的子文件夹。例如：

```
data/
├── train/
│   ├── 0_ashless/       # 无灰尘图像
│   ├── 1_little_ashes/  # 轻微积灰图像
│   └── 2_all_ashes/     # 严重积灰图像
└── val/
    ├── 0_ashless/
    ├── 1_little_ashes/
    └── 2_all_ashes/
```

**重要提示**：`data/` 文件夹通常不直接上传到 GitHub，因为它可能包含大量图像数据。请确保您在本地拥有这些数据，或者在 `README.md` 中提供获取数据的说明。

### 计算数据集均值和标准差

在训练模型之前，建议计算训练数据集的均值和标准差，用于图像归一化。您可以使用以下脚本来完成：

```python
# calculate_mean_std.py (示例代码，请根据实际路径和需求调整)
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 数据集根目录
data_dir = 'd:\\Documents\\研究生\\课程论文\\智慧能源系统工程\\CV\\data\\train'

# 定义一个简单的转换，只将图片转换为 Tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建 ImageFolder 数据集
dataset = ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

mean = 0.
std = 0.
total_images_count = 0

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    total_images_count += batch_samples

mean /= total_images_count

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    std += ((images - mean.unsqueeze(1))**2).sum(2).sum(0)

std = torch.sqrt(std / (total_images_count * images.size(2)))

print(f"数据集均值 (Mean): {mean}")
print(f"数据集标准差 (Std): {std}")
```

将计算得到的 `mean` 和 `std` 值更新到 `dataset.py` 中的 `transforms.Normalize` 函数中。

## 项目结构

```
. # 项目根目录
├── .idea/                 # IDE 配置文件 (通常不上传)
├── best_model_MobileNetV2.pth # 训练好的 MobileNetV2 模型权重
├── best_model_ResNet18.pth    # 训练好的 ResNet18 模型权重
├── best_model_SimpleCNN.pth   # 训练好的 SimpleCNN 模型权重
├── data/                  # 数据集目录 (通常不上传)
│   ├── __pycache__/       # Python 缓存文件
│   ├── train/             # 训练集
│   └── val/               # 验证集
├── dataset.py             # 数据集加载和预处理脚本
├── main.py                # 项目主入口，可能包含训练和评估的逻辑
├── model.py               # 模型定义脚本 (包含 SimpleCNN, MobileNetV2, ResNet18 等模型)
├── predict_gui.py         # 预测功能的图形用户界面脚本
├── requirements.txt       # 项目依赖包列表
└── train_eval.py          # 模型训练和评估脚本
```

## 使用方法

### 训练模型

运行 `train_eval.py` 脚本来训练模型。您可能需要根据 `train_eval.py` 中的参数进行调整，例如选择不同的模型、设置学习率、批大小等。

```bash
python train_eval.py
```

### 使用 GUI 进行预测

运行 `predict_gui.py` 脚本以启动图形用户界面，您可以加载训练好的模型并对新的光伏板图像进行积灰程度预测。

```bash
python predict_gui.py
```