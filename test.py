import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

# 导入模型定义
from model import SimpleCNN
from torchvision import models
import torch.nn as nn

# 定义不使用预训练权重的模型创建函数
def get_resnet18_no_pretrain(num_classes=3):
    model = models.resnet18(weights=None)  # 不使用预训练权重
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50_no_pretrain(num_classes=3):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_mobilenet_v2_no_pretrain(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_mobilenet_v3_no_pretrain(num_classes=3):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def get_alexnet_no_pretrain(num_classes=3):
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def get_googlenet_no_pretrain(num_classes=3):
    model = models.googlenet(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 类别标签
CLASS_NAMES = ['0_ashless', '1_little_ashes', '2_all_ashes']

# 模型配置
MODEL_CONFIGS = {
    'SimpleCNN': {
        'loader': SimpleCNN,
        'file_pattern': 'simplecnn'
    },
    'ResNet18': {
        'loader': get_resnet18_no_pretrain,
        'file_pattern': 'resnet18'
    },
    'ResNet50': {
        'loader': get_resnet50_no_pretrain,
        'file_pattern': 'resnet50'
    },
    'MobileNetV2': {
        'loader': get_mobilenet_v2_no_pretrain,
        'file_pattern': 'mobilenetv2'
    },
    'MobileNetV3': {
        'loader': get_mobilenet_v3_no_pretrain,
        'file_pattern': 'mobilenetv3'
    },
    'GoogLeNet': {
        'loader': get_googlenet_no_pretrain,
        'file_pattern': 'googlenet'
    },
    'AlexNet': {
        'loader': get_alexnet_no_pretrain,
        'file_pattern': 'alexnet'
    }
}

def load_model(model_name, model_path):
    """加载训练好的模型"""
    try:
        config = MODEL_CONFIGS[model_name]
        model = config['loader'](num_classes=3)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"加载模型 {model_name} 失败: {e}")
        return None

def get_test_dataloader(data_dir, batch_size=32):
    """创建测试集数据加载器"""
    # 使用与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4973815083503723, 0.5052969455718994, 0.5267759561538696], 
                           [0.2140694409608841, 0.18967010080814362, 0.16519778966903687])
    ])
    
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估中"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各种指标
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    # 计算F1分数
    f1_scores = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 计算每个类别的详细指标
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    
    results = {
        'test_loss': avg_loss,
        'test_acc': accuracy,
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        '0_ashless_f1': f1_scores[0],
        '1_little_ashes_f1': f1_scores[1],
        '2_all_ashes_f1': f1_scores[2],
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
    
    return results

def find_model_files(checkpoint_dir):
    """查找所有模型文件"""
    model_files = {}
    
    if not os.path.exists(checkpoint_dir):
        print(f"检查点目录不存在: {checkpoint_dir}")
        return model_files
    
    # 遍历检查点目录中的所有文件
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            # 根据文件名匹配模型类型
            filename_lower = filename.lower()
            for model_name, config in MODEL_CONFIGS.items():
                pattern = config['file_pattern'].lower()
                if pattern in filename_lower:
                    model_files[model_name] = os.path.join(checkpoint_dir, filename)
                    break
    
    return model_files

def main():
    """主函数"""
    data_dir = "./data"
    checkpoint_dir = "./checkpoint"
    
    # 检查测试集是否存在
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        print(f"测试集目录不存在: {test_dir}")
        return
    
    # 创建测试集数据加载器
    print("加载测试集数据...")
    test_loader, class_names = get_test_dataloader(data_dir)
    print(f"测试集包含 {len(test_loader.dataset)} 张图片")
    print(f"类别: {class_names}")
    
    # 查找所有模型文件
    model_files = find_model_files(checkpoint_dir)
    
    if not model_files:
        print("未找到任何模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for model_name, file_path in model_files.items():
        print(f"  - {model_name}: {file_path}")
    
    # 评估每个模型
    results_data = []
    
    for model_name, model_path in model_files.items():
        print(f"\n正在评估 {model_name}...")
        
        # 加载模型
        model = load_model(model_name, model_path)
        if model is None:
            print(f"跳过 {model_name}")
            continue
        
        # 评估模型
        try:
            results = evaluate_model(model, test_loader)
            results['Model'] = model_name
            results_data.append(results)
            
            print(f"{model_name} 评估完成:")
            print(f"  - 测试准确率: {results['test_acc']:.2f}%")
            print(f"  - 测试损失: {results['test_loss']:.4f}")
            print(f"  - Macro F1: {results['macro_f1']:.4f}")
            print(f"  - Weighted F1: {results['weighted_f1']:.4f}")
            
        except Exception as e:
            print(f"评估 {model_name} 时出错: {e}")
            continue
        
        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存结果
    if results_data:
        df = pd.DataFrame(results_data)
        
        # 重新排列列的顺序
        columns_order = [
            'Model', 'test_loss', 'test_acc', 'accuracy',
            '0_ashless_f1', '1_little_ashes_f1', '2_all_ashes_f1',
            'macro_f1', 'weighted_f1'
        ]
        df = df[columns_order]
        
        # 保存为Excel文件
        output_file = "test_results.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n结果已保存到 {output_file}")
        
        # 显示结果摘要
        print("\n=== 测试结果摘要 ===")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # 找出最佳模型
        best_accuracy_model = df.loc[df['test_acc'].idxmax()]
        best_f1_model = df.loc[df['macro_f1'].idxmax()]
        
        print(f"\n最高准确率模型: {best_accuracy_model['Model']} ({best_accuracy_model['test_acc']:.2f}%)")
        print(f"最高Macro F1模型: {best_f1_model['Model']} ({best_f1_model['macro_f1']:.4f})")
        
    else:
        print("没有成功评估任何模型")

if __name__ == "__main__":
    main()
