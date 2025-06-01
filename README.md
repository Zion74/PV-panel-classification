# 智慧能源系统工程 - 光伏板积灰判断项目

这是一个基于深度学习的光伏板积灰判断项目，旨在通过图像识别技术，自动判断光伏板的积灰程度。本项目使用了 PyTorch 深度学习框架，并提供了训练、评估和预测的完整流程。

## 项目概述

随着光伏发电的普及，光伏板表面的积灰问题日益突出，严重影响发电效率。本项目旨在开发一个智能系统，通过分析光伏板图像，自动识别并判断其积灰程度，为光伏电站的运维提供数据支持。

## 主要功能

*   **图像数据预处理**：对光伏板图像进行统一尺寸调整和归一化处理。
*   **深度学习模型训练**：支持使用多种预训练模型（如 MobileNetV2, ResNet18, SimpleCNN）进行迁移学习，对积灰图像进行分类训练。
*   **模型评估**：在验证集上评估模型的性能，包括准确率、损失等指标。
*   **图像预测**：提供图形用户界面 (GUI) 进行单张或批量图像的积灰程度预测。

## 🚀 快速开始

### 配置

1. 克隆或下载本项目

   ```bash
   git clone https://github.com/yourusername/PV-panel-classification.git
   cd PV-panel-classification
   ```

2. 创建虚拟环境
    ```bash
    conda create -n pv_dust python=3.x # 推荐使用 Python 3.8 或更高版本
    conda activate pv_dust

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

### 数据集准备

数据集需按以下结构组织：

```
data/
├── train/
│   ├── 0_ashless/         # 无积灰图像
│   ├── 1_little_ashes/    # 轻微积灰图像
│   └── 2_all_ashes/       # 严重积灰图像
└── val/
    ├── 0_ashless/
    ├── 1_little_ashes/
    └── 2_all_ashes/
```

### 模型训练

运行以下命令训练模型：

```bash
python train.py --model resnet18 --epochs 30 --batch_size 32
```

支持的模型选项：

- `simplecnn` - 简单卷积神经网络
- `resnet18` - 轻量级残差网络
- `resnet50` - 深度残差网络
- `mobilenetv2` - 移动端优化网络
- `mobilenetv3` - 改进移动端网络
- `googlenet` - Inception 网络
- `alexnet` - AlexNet 网络

### 模型评估与比较

比较不同模型的性能：

```bash
python compare_models.py
```

### 启动图形界面

```bash
python GUI.py
```

## 🖥️ 图形界面使用指南


1. **启动应用**：运行 `python GUI.py`
2. **选择模型**：从左侧下拉菜单选择预训练模型，点击"加载选中模型"
3. **图像识别**：
   - 点击"上传图片"按钮上传自定义图片
   - 或点击"随机验证集图片"按钮使用验证集图片
4. **查看结果**：
   - 识别结果将显示预测的积灰等级
   - 置信度条直观显示预测的可信度
   - 如果使用验证集图片，还会显示真实标签


## 📁 项目结构

```
PV-panel-classification/
├── GUI.py             # 图形界面主程序
├── train.py           # 模型训练脚本
├── model.py           # 模型定义
├── dataset.py         # 数据集加载和预处理
├── compare_models.py  # 模型比较脚本
├── requirements.txt   # 项目依赖
├── checkpoint/        # 保存训练好的模型
├── data/              # 数据集目录
└── results/           # 结果和日志
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)

---

<div align="center">
  <p>💖 感谢使用光伏板积灰程度智能识别系统 💖</p>
</div>
