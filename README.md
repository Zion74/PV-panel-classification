# 光伏板积灰程度智能识别系统

这是一个基于深度学习的光伏板积灰程度识别系统，可以自动识别光伏板的积灰程度，分为三个等级：无积灰、轻微积灰和严重积灰。

## 系统要求

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.1+
- PyQt5 5.15.0+
- 其他依赖见 requirements.txt

## 安装

1. 克隆或下载本项目
2. 安装依赖：

```
pip install -r requirements.txt
```

## 数据集结构

数据集应按以下结构组织：

```
data/
├── train/
│   ├── 0_ashless/
│   ├── 1_little_ashes/
│   └── 2_all_ashes/
└── val/
    ├── 0_ashless/
    ├── 1_little_ashes/
    └── 2_all_ashes/
```

## 使用方法

### 训练模型

运行以下命令训练模型：

```
python train.py --model resnet18 --epochs 30 --batch_size 32
```

可用的模型选项：

- simplecnn
- resnet18
- resnet50
- mobilenet_v2
- mobilenet_v3
- alexnet
- googlenet

### 比较不同模型

运行以下命令比较不同模型的性能：

```
python compare_models.py
```

### 图形界面预测

运行以下命令启动图形界面进行预测：

```
python predict_gui.py
```

## 新版 GUI 界面

新版 GUI 基于 PyQt5 开发，提供了更现代化、更美观的用户界面，功能包括：

1. **模型选择**：可以从 checkpoint 文件夹中选择并加载任意训练好的模型
2. **图像预测**：支持上传自定义图片进行预测
3. **随机验证**：可以从验证集中随机选择图片进行预测
4. **结果展示**：直观显示预测结果、置信度和真实标签（如果有）
5. **状态反馈**：通过状态栏和颜色编码提供清晰的状态反馈

### GUI 使用步骤

1. 启动应用：`python predict_gui.py`
2. 从左侧下拉菜单选择一个模型，点击"加载选中模型"
3. 选择以下操作之一：
   - 点击"上传图片"按钮上传自定义图片进行预测
   - 点击"随机验证集图片"按钮从验证集随机选择一张图片进行预测
4. 查看右侧面板中的预测结果、置信度和真实标签信息

## 构建可执行文件

本项目提供了一个脚本用于将应用程序打包成可执行文件，使其能在没有安装 Python 环境的计算机上运行。

### 前提条件

- 安装 PyInstaller：`pip install pyinstaller`
- 确保所有依赖项已安装：`pip install -r requirements.txt`

### 构建步骤

1. 运行构建脚本：

```
python build_executable.py
```

2. 默认情况下，这将创建一个单文件可执行程序。如果要创建文件夹模式的应用程序，请使用：

```
python build_executable.py --onefile False
```

3. 其他可选参数：

   - `--name NAME`：指定输出的可执行文件名称（默认：PV_Panel_Classifier）
   - `--icon PATH`：指定应用程序图标路径
   - `--noclean`：保留之前的构建文件

4. 构建完成后，可执行文件将位于`dist`文件夹中。

### 注意事项

- 单文件模式（`--onefile`）会将所有依赖打包到一个.exe 文件中，启动较慢但分发方便
- 文件夹模式启动更快，但需要分发整个文件夹
- 确保`checkpoint`文件夹中至少有一个训练好的模型文件（.pth）
- 确保`data/val`文件夹包含验证数据，以便使用"随机验证集图片"功能

## 模型性能

详细的模型性能比较可以参考：

- final_model_comparison.xlsx
- model_comparison.xlsx
- model_training_times.xlsx

## 项目结构

- `train.py`: 模型训练脚本
- `model.py`: 模型定义
- `dataset.py`: 数据集加载和预处理
- `compare_models.py`: 模型比较脚本
- `predict_gui.py`: 图形界面预测程序
- `main.py`: 主程序入口
- `build_executable.py`: 构建可执行文件脚本
- `checkpoint/`: 保存训练好的模型
- `data/`: 数据集目录
- `results/`: 保存结果和日志

## 许可证

MIT

## 联系方式

如有问题，请提交 Issue 或 Pull Request。
