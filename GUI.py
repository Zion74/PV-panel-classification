import os
import sys
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets

# 设置模型下载路径
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')

# 导入PyQt5库
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QFileDialog, QFrame, 
    QSplitter, QGroupBox, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor

# 导入模型定义
from model import (
    SimpleCNN, get_resnet18, get_resnet50, get_mobilenet_v2, 
    get_mobilenet_v3, get_alexnet, get_googlenet
)
from dataset import calculate_mean_std

# 类别标签
CLASS_NAMES = ['0_ashless', '1_little_ashes', '2_all_ashes']
CLASS_DESCRIPTIONS = {
    '0_ashless': '无积灰',
    '1_little_ashes': '轻微积灰',
    '2_all_ashes': '严重积灰'
}

# 模型名称映射
MODEL_NAMES = {
    'resnet18': 'ResNet18 (轻量级残差网络)',
    'resnet50': 'ResNet50 (深度残差网络)',
    'simplecnn': 'SimpleCNN (简单卷积网络)',
    'mobilenetv2': 'MobileNetV2 (移动端优化网络)',
    'mobilenetv3': 'MobileNetV3 (改进移动端网络)',
    'googlenet': 'GoogLeNet (Inception网络)',
    'alexnet': 'AlexNet'
}

# 模型排序顺序
MODEL_ORDER = [
    'resnet18',
    'resnet50',
    'simplecnn',
    'mobilenetv2',
    'mobilenetv3',
    'googlenet',
    # 'alexnet'
]

# 模型加载函数映射
MODEL_LOADERS = {
    'resnet18': get_resnet18,
    'resnet50': get_resnet50,
    'simplecnn': SimpleCNN,
    'mobilenetv2': get_mobilenet_v2,
    'mobilenetv3': get_mobilenet_v3,
    'googlenet': get_googlenet,
    # 'alexnet': get_alexnet
}

# 模型文件名映射
MODEL_FILE_PATTERNS = {
    'resnet18': ['resnet18'],
    'resnet50': ['resnet50'],
    'simplecnn': ['simplecnn'],
    'mobilenetv2': ['mobilenetv2', 'mobilenet_v2'],
    'mobilenetv3': ['mobilenetv3', 'mobilenet_v3'],
    'googlenet': ['googlenet'],
    'alexnet': ['alexnet']
}

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算训练集的均值和标准差
data_dir = "./data"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4973815083503723, 0.5052969455718994, 0.5267759561538696], [0.2140694409608841, 0.18967010080814362, 0.16519778966903687])
])

# 设置全局字体
CHINESE_FONT = "Microsoft YaHei"  # 微软雅黑
ENGLISH_FONT = "Arial"

# 预测线程类
class PredictionThread(QThread):
    finished = pyqtSignal(int, float, str)
    
    def __init__(self, model, image_path, is_pil_image=False):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.is_pil_image = is_pil_image
        
    def run(self):
        try:
            if self.is_pil_image:
                img = self.image_path
            else:
                img = Image.open(self.image_path).convert('RGB')
                
            # 使用transform处理图像
            input_tensor = transform(img)
            # 添加batch维度并移动到指定设备
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = float(probs[0][pred_idx].item())
                
            # 获取真实标签（如果有）
            true_label = "未知"
            if not self.is_pil_image and isinstance(self.image_path, str):
                for class_name in CLASS_NAMES:
                    if class_name in self.image_path:
                        true_label = class_name
                        break
                
            self.finished.emit(pred_idx, confidence, true_label)
        except Exception as e:
            print(f"预测过程中出错: {e}")
            self.finished.emit(-1, 0.0, str(e))

# 主窗口类
class PVPanelClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = ""
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.current_image_path = ""
        self.model_files = {}  # 存储模型名称到文件路径的映射
        
        self.init_ui()
        self.load_models_list()
        
    def init_ui(self):
        # 设置窗口基本属性
        self.setWindowTitle("光伏板积灰程度智能识别系统")
        self.setMinimumSize(1200, 900)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("光伏板积灰程度智能识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont(CHINESE_FONT, 22, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2C3E50; margin: 15px; letter-spacing: 1px;")
        main_layout.addWidget(title_label)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #3498DB; height: 2px;")
        main_layout.addWidget(line)
        
        # 创建中央区域的水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：控制面板
        control_panel = QGroupBox("控制面板")
        control_panel.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 20px;
                font-weight: bold;
                border: 1px solid #BDC3C7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #F8F9F9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #2980B9;
            }
        """)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 20, 15, 15)
        control_layout.setSpacing(20)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_group.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 18px;
                border: 1px solid #D5DBDB;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
                color: #34495E;
            }
        """)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(12)
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                font-family: 'Microsoft YaHei', 'Arial';
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                padding: 6px;
                min-height: 30px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                font-family: 'Microsoft YaHei', 'Arial';
                selection-background-color: #3498DB;
            }
        """)
        model_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("加载选中模型")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                font-family: 'Microsoft YaHei';
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1A5276;
            }
        """)
        model_layout.addWidget(self.load_model_btn)
        
        control_layout.addWidget(model_group)
        
        # 图像选择
        image_group = QGroupBox("图像选择")
        image_group.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 18px;
                border: 1px solid #D5DBDB;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
                color: #34495E;
            }
        """)
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(12)
        
        # 数据集选择
        dataset_label = QLabel("选择数据集:")
        dataset_label.setFont(QFont(CHINESE_FONT, 10))
        dataset_label.setStyleSheet("color: #34495E; margin-bottom: 5px;")
        image_layout.addWidget(dataset_label)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["训练集", "验证集", "测试集"])
        self.dataset_combo.setCurrentText("测试集")  # 默认选择验证集
        self.dataset_combo.setStyleSheet("""
            QComboBox {
                font-family: 'Microsoft YaHei', 'Arial';
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                padding: 6px;
                min-height: 25px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                font-family: 'Microsoft YaHei', 'Arial';
                selection-background-color: #3498DB;
            }
        """)
        image_layout.addWidget(self.dataset_combo)
        
        self.upload_img_btn = QPushButton("上传图片")
        self.upload_img_btn.setStyleSheet("""
            QPushButton {
                font-family: 'Microsoft YaHei';
                background-color: #2ECC71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
        """)
        image_layout.addWidget(self.upload_img_btn)
        
        self.random_img_btn = QPushButton("随机选择图片")
        self.random_img_btn.setStyleSheet("""
            QPushButton {
                font-family: 'Microsoft YaHei';
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #922B21;
            }
        """)
        image_layout.addWidget(self.random_img_btn)
        
        control_layout.addWidget(image_group)
        
        # 模型信息
        info_group = QGroupBox("模型信息")
        info_group.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 18px;
                border: 1px solid #D5DBDB;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
                color: #34495E;
            }
        """)
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setAlignment(Qt.AlignCenter)
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("font-family: 'Microsoft YaHei', 'Arial'; font-size: 12px; padding: 8px;")
        info_layout.addWidget(self.model_info_label)
        
        control_layout.addWidget(info_group)
        
        # 添加弹性空间
        control_layout.addStretch(1)
        
        # 右侧：结果显示
        result_panel = QGroupBox("图片")
        result_panel.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 20px;
                font-weight: bold;
                border: 1px solid #BDC3C7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #F8F9F9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #2980B9;
            }
        """)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(15, 20, 15, 15)
        result_layout.setSpacing(15)
        
        # 图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("""
            border: 2px dashed #BDC3C7;
            border-radius: 8px;
            background-color: #EAECEE;
            padding: 5px;
        """)
        self.image_label.setText("请选择图片")
        self.image_label.setFont(QFont(CHINESE_FONT, 12))
        result_layout.addWidget(self.image_label)
        
        # 预测结果
        result_group = QGroupBox("预测详情")
        result_group.setStyleSheet("""
            QGroupBox {
                font-family: 'Microsoft YaHei';
                font-size: 18px;
                border: 1px solid #D5DBDB;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
                color: #34495E;
            }
        """)
        result_details_layout = QVBoxLayout(result_group)
        result_details_layout.setSpacing(15)
        
        # 预测结果标签 - 字体与真实标签一致
        self.result_label = QLabel("等待预测...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont(CHINESE_FONT, 14))  # 与真实标签字体一致
        self.result_label.setStyleSheet("color: #2C3E50; margin: 10px; letter-spacing: 1px;")
        result_details_layout.addWidget(self.result_label)
        
        # 真实标签 - 移到预测标签下面
        self.true_label = QLabel("真实标签: 未知")
        self.true_label.setAlignment(Qt.AlignCenter)
        self.true_label.setFont(QFont(CHINESE_FONT, 14))  # 稍小一些的字体
        self.true_label.setStyleSheet("color: #34495E; padding: 5px; margin-top: 5px;")
        result_details_layout.addWidget(self.true_label)
        
        # 置信度进度条 - 使用水平布局使文本显示在右侧
        confidence_layout = QHBoxLayout()
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)  # 不显示内置文本
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 4px;
            }
        """)
        confidence_layout.addWidget(self.confidence_bar, 8)  # 进度条占70%宽度
        
        # 添加置信度文本标签
        self.confidence_label = QLabel("置信度: 0%")
        self.confidence_label.setFont(QFont(CHINESE_FONT, 11))
        self.confidence_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.confidence_label.setStyleSheet("color: #34495E; padding-left: 10px;")
        confidence_layout.addWidget(self.confidence_label, 2)  # 文本占30%宽度
        
        result_details_layout.addLayout(confidence_layout)
        
        result_layout.addWidget(result_group)
        
        # 添加到分割器
        splitter.addWidget(control_panel)
        splitter.addWidget(result_panel)
        splitter.setSizes([300, 600])  # 设置初始大小比例
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().setFont(QFont(CHINESE_FONT, 10))
        self.statusBar().setStyleSheet("background-color: #F8F9F9; color: #34495E; padding: 5px;")
        self.statusBar().showMessage(f"系统就绪 | 运行设备: {DEVICE}")
        
        # 连接信号和槽
        self.load_model_btn.clicked.connect(self.load_selected_model)
        self.upload_img_btn.clicked.connect(self.upload_image)
        self.random_img_btn.clicked.connect(self.random_val_image)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QLabel {
                font-family: 'Microsoft YaHei', 'Arial';
                color: #2C3E50;
            }
            QMessageBox {
                font-family: 'Microsoft YaHei', 'Arial';
            }
            QMessageBox QPushButton {
                font-family: 'Microsoft YaHei';
                padding: 5px 15px;
                border-radius: 4px;
                background-color: #3498DB;
                color: white;
            }
        """)
        
    def load_models_list(self):
        """加载checkpoint文件夹中的所有模型，并按指定顺序排序"""
        try:
            checkpoint_dir = "./checkpoint"
            model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            
            # 清空现有模型
            self.model_combo.clear()
            self.model_files = {}
            
            # 按模型类型分类文件
            model_by_type = {}
            for model_file in model_files:
                model_file_lower = model_file.lower()
                matched = False
                
                # 使用模型文件名模式匹配
                for model_key, patterns in MODEL_FILE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in model_file_lower:
                            if model_key not in model_by_type:
                                model_by_type[model_key] = []
                            model_by_type[model_key].append(model_file)
                            matched = True
                            break
                    if matched:
                        break
                
                # 如果没有匹配到任何模式，尝试使用旧的匹配方式
                if not matched:
                    print(f"未匹配到模型类型: {model_file}")
            
            # 打印找到的模型
            for model_key, files in model_by_type.items():
                print(f"找到模型 {model_key}: {files}")
            
            # 按指定顺序添加模型
            for model_key in MODEL_ORDER:
                if model_key in model_by_type:
                    for model_file in model_by_type[model_key]:
                        display_name = MODEL_NAMES.get(model_key, model_key)
                        self.model_combo.addItem(display_name)
                        self.model_files[display_name] = model_file
            
            # 如果有ResNet18模型，默认选择它
            for i in range(self.model_combo.count()):
                if "ResNet18" in self.model_combo.itemText(i):
                    self.model_combo.setCurrentIndex(i)
                    break
            
            if model_files:
                self.statusBar().showMessage(f"找到 {len(model_files)} 个模型文件")
            else:
                self.statusBar().showMessage("未找到模型文件，请检查checkpoint文件夹")
                
        except Exception as e:
            self.statusBar().showMessage(f"加载模型列表失败: {str(e)}")
            print(f"加载模型列表失败: {str(e)}")
    
    def load_selected_model(self):
        """加载选中的模型"""
        if self.model_combo.count() == 0:
            QMessageBox.warning(self, "警告", "没有可用的模型文件")
            return
            
        display_name = self.model_combo.currentText()
        model_file = self.model_files.get(display_name)
        
        if not model_file:
            QMessageBox.warning(self, "警告", "无法找到对应的模型文件")
            return
            
        model_path = os.path.join("./checkpoint", model_file)
        
        try:
            # 根据显示名称确定模型类型
            model_key = None
            for key in MODEL_NAMES:
                if MODEL_NAMES[key] == display_name:
                    model_key = key
                    break
            
            if not model_key:
                # 如果无法通过显示名称确定，则尝试通过文件名判断
                model_file_lower = model_file.lower()
                for key, patterns in MODEL_FILE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in model_file_lower:
                            model_key = key
                            break
                    if model_key:
                        break
            
            if not model_key:
                QMessageBox.warning(self, "警告", "无法识别模型类型")
                return
                
            # 加载对应的模型
            if model_key == 'simplecnn':
                self.model = MODEL_LOADERS[model_key](num_classes=3)
            else:
                self.model = MODEL_LOADERS[model_key](num_classes=3)
                
            self.model_name = MODEL_NAMES.get(model_key, model_key)
                
            # 加载模型权重
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            
            # 更新界面
            self.model_info_label.setText(f"当前模型: {self.model_name}\n文件: {model_file}")
            self.statusBar().showMessage(f"成功加载模型: {self.model_name}")
            
            # 加载数据集
            self.load_datasets()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.statusBar().showMessage(f"加载模型失败: {str(e)}")

    def load_datasets(self):
        """加载所有数据集"""
        try:
            data_dir = "./data"
            
            # 加载验证集
            if os.path.exists(f"{data_dir}/val"):
                self.val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
                val_count = len(self.val_dataset)
            else:
                self.val_dataset = None
                val_count = 0
            
            # 加载训练集
            if os.path.exists(f"{data_dir}/train"):
                self.train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
                train_count = len(self.train_dataset)
            else:
                self.train_dataset = None
                train_count = 0
            
            # 加载测试集
            if os.path.exists(f"{data_dir}/test"):
                self.test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
                test_count = len(self.test_dataset)
            else:
                self.test_dataset = None
                test_count = 0
            
            status_msg = f"已加载数据集 - 训练集: {train_count}, 验证集: {val_count}, 测试集: {test_count} 张图片"
            self.statusBar().showMessage(status_msg)
            
        except Exception as e:
            self.statusBar().showMessage(f"加载数据集失败: {str(e)}")
    
    def load_val_dataset(self):
        """保持向后兼容的验证集加载方法"""
        self.load_datasets()
    
    def upload_image(self):
        """上传图片并预测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
            
        try:
            # 显示图片
            self.display_image(file_path)
            self.current_image_path = file_path
            
            # 开始预测
            self.statusBar().showMessage("正在预测...")
            self.prediction_thread = PredictionThread(self.model, file_path)
            self.prediction_thread.finished.connect(self.update_prediction_result)
            self.prediction_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片失败: {str(e)}")
            self.statusBar().showMessage(f"处理图片失败: {str(e)}")
    
    def random_val_image(self):
        """随机选择指定数据集图片并预测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        # 获取选择的数据集
        selected_dataset = self.dataset_combo.currentText()
        dataset = None
        
        if selected_dataset == "训练集":
            dataset = self.train_dataset
        elif selected_dataset == "验证集":
            dataset = self.val_dataset
        elif selected_dataset == "测试集":
            dataset = self.test_dataset
        
        # 检查数据集是否存在
        if dataset is None:
            self.load_datasets()
            
            # 重新获取数据集
            if selected_dataset == "训练集":
                dataset = self.train_dataset
            elif selected_dataset == "验证集":
                dataset = self.val_dataset
            elif selected_dataset == "测试集":
                dataset = self.test_dataset
            
            if dataset is None:
                QMessageBox.warning(self, "警告", f"无法加载{selected_dataset}数据")
                return
        
        # 随机选择一张图片
        idx = random.randint(0, len(dataset) - 1)
        img_path = dataset.imgs[idx][0]
        
        # 显示图片
        self.display_image(img_path)
        self.current_image_path = img_path
        
        # 开始预测
        self.statusBar().showMessage(f"正在预测{selected_dataset}图片...")
        self.prediction_thread = PredictionThread(self.model, img_path)
        self.prediction_thread.finished.connect(self.update_prediction_result)
        self.prediction_thread.start()
    
    def display_image(self, image_path):
        """显示图片"""
        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.image_label.setText("无法加载图片")
                return
                
            # 调整图片大小以适应标签
            pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            
        except Exception as e:
            self.image_label.setText(f"显示图片失败: {str(e)}")
    
    def update_prediction_result(self, pred_idx, confidence, true_label):
        """更新预测结果"""
        if pred_idx < 0:
            self.result_label.setText(f"预测失败: {true_label}")
            self.confidence_bar.setValue(0)
            self.confidence_label.setText("置信度: 0%")
            self.true_label.setText("真实标签: 未知")
            self.statusBar().showMessage("预测失败")
            return
            
        # 更新预测结果标签 - 添加预测成功标识
        class_name = CLASS_NAMES[pred_idx]
        class_desc = CLASS_DESCRIPTIONS[class_name]
        
        # 判断预测是否成功
        prediction_success = true_label in CLASS_NAMES and true_label == class_name
        
        # 在预测结果后添加"(预测成功)"字样
        success_text = " (预测成功)" if prediction_success else ""
        self.result_label.setText(f"预测结果：{class_desc}{success_text}")
        
        # 根据预测结果设置不同颜色（保持字体大小为30pt）
        if pred_idx == 0:  # 无积灰
            self.result_label.setStyleSheet("color: #27AE60; font-size: 30px; font-weight: bold; font-family: 'Microsoft YaHei', 'Arial'; margin: 10px;")
        elif pred_idx == 1:  # 轻微积灰
            self.result_label.setStyleSheet("color: #F39C12; font-size: 30px; font-weight: bold; font-family: 'Microsoft YaHei', 'Arial'; margin: 10px;")
        else:  # 严重积灰
            self.result_label.setStyleSheet("color: #E74C3C; font-size: 30px; font-weight: bold; font-family: 'Microsoft YaHei', 'Arial'; margin: 10px;")
        
        # 更新置信度 - 根据置信度大小设置颜色
        confidence_value = int(confidence * 100)
        self.confidence_bar.setValue(confidence_value)
        self.confidence_label.setText(f"置信度: {confidence_value}%")
        
        # 根据置信度设置进度条颜色
        if confidence_value >= 90:  # 90%以上绿色
            confidence_color = "#27AE60"
        elif confidence_value >= 70:  # 70%-90%黄色
            confidence_color = "#F39C12"
        else:  # 70%以下红色
            confidence_color = "#E74C3C"
        
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{ 
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #F5F5F5;
            }}
            QProgressBar::chunk {{ 
                background-color: {confidence_color}; 
                border-radius: 4px;
            }}
        """)
        
        # 更新真实标签
        if true_label in CLASS_NAMES:
            true_desc = CLASS_DESCRIPTIONS[true_label]
            self.true_label.setText(f"真实标签: {true_desc}")
        else:
            self.true_label.setText("真实标签: 未知")
        
        self.statusBar().showMessage(f"预测完成 | 类别: {class_desc} ({class_name}) | 置信度: {confidence:.2f}")
        # 主函数
def main():
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格
    
    # 设置全局字体
    font = QFont(CHINESE_FONT)
    app.setFont(font)
    
    # 创建窗口
    window = PVPanelClassifier()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 确保模型缓存目录存在
    model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = model_cache_dir
    main()

