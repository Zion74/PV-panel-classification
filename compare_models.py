import os
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import classification_report
import torch
from model import (
    SimpleCNN, get_resnet18, get_resnet50, get_mobilenet_v2, 
    get_mobilenet_v3, get_alexnet, get_googlenet
)
from dataset import get_dataloaders

def load_model(model_class, checkpoint_path, device):
    """加载训练好的模型"""
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model_metrics(model, val_loader, device, classes):
    """评估模型并返回详细指标"""
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
    
    # 计算准确率
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    
    # 获取分类报告
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # 提取每个类别的精确度、召回率和F1分数
    metrics = {}
    for i, cls in enumerate(classes):
        metrics[f"{cls}_precision"] = report[cls]['precision']
        metrics[f"{cls}_recall"] = report[cls]['recall']
        metrics[f"{cls}_f1"] = report[cls]['f1-score']
    
    # 添加宏平均和加权平均
    metrics["macro_precision"] = report['macro avg']['precision']
    metrics["macro_recall"] = report['macro avg']['recall']
    metrics["macro_f1"] = report['macro avg']['f1-score']
    metrics["weighted_precision"] = report['weighted avg']['precision']
    metrics["weighted_recall"] = report['weighted avg']['recall']
    metrics["weighted_f1"] = report['weighted avg']['f1-score']
    
    # 添加总体准确率
    metrics["accuracy"] = accuracy
    
    return metrics

def get_final_metrics(results_path):
    """从训练结果Excel文件中获取最终指标和训练时间信息"""
    metrics = {
        "final_train_loss": np.nan,
        "final_train_acc": np.nan,
        "final_val_loss": np.nan,
        "final_val_acc": np.nan,
        "epochs": np.nan,
        "total_train_time": np.nan,
        "avg_epoch_time": np.nan
    }
    
    try:
        # 读取训练指标
        df = pd.read_excel(results_path, sheet_name=0)
        # 获取最后一个epoch的指标
        last_row = df.iloc[-1]
        metrics.update({
            "final_train_loss": last_row["Train Loss"],
            "final_train_acc": last_row["Train Accuracy"],
            "final_val_loss": last_row["Validation Loss"],
            "final_val_acc": last_row["Validation Accuracy"],
            "epochs": last_row["Epoch"]
        })
        
        # 尝试读取时间信息（如果存在）
        try:
            time_df = pd.read_excel(results_path, sheet_name="Time Info")
            if not time_df.empty:
                metrics.update({
                    "total_train_time": time_df["总训练时间(秒)"].iloc[0],
                    "avg_epoch_time": time_df["平均每个Epoch时间(秒)"].iloc[0]
                })
        except Exception as e:
            print(f"读取{results_path}的时间信息时出错: {e}")
            
    except Exception as e:
        print(f"读取{results_path}时出错: {e}")
        
    return metrics

def compare_models():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    data_dir = r"/openbayes/home/PV-panel-classification/data"
    _, val_loader, classes = get_dataloaders(data_dir, batch_size=32, use_calculated_stats=True)
    
    # 模型类映射
    model_classes = {
        "SimpleCNN": SimpleCNN,
        "ResNet18": get_resnet18,
        "ResNet50": get_resnet50,
        "MobileNetV2": get_mobilenet_v2,
        "MobileNetV3": get_mobilenet_v3,
        "AlexNet": get_alexnet,
        "GoogLeNet": get_googlenet
    }
    
    # 查找所有模型检查点
    checkpoint_dir = "checkpoint"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "best_model_*.pth"))
    
    # 创建结果列表
    results = []
    
    for checkpoint in checkpoints:
        model_name = os.path.basename(checkpoint).replace("best_model_", "").replace(".pth", "")
        print(f"\n评估模型: {model_name}")
        
        # 检查模型类是否存在
        if model_name not in model_classes:
            print(f"警告: 找不到模型类 {model_name}，跳过评估")
            continue
        
        # 加载模型
        try:
            model = load_model(model_classes[model_name], checkpoint, device)
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")
            continue
        
        # 评估模型
        metrics = evaluate_model_metrics(model, val_loader, device, classes)
        
        # 获取训练结果
        results_path = f"results/{model_name}/training_results.xlsx"
        if os.path.exists(results_path):
            training_metrics = get_final_metrics(results_path)
            metrics.update(training_metrics)
        
        # 添加模型名称
        metrics["model_name"] = model_name
        
        # 添加到结果列表
        results.append(metrics)
    
    # 创建DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # 重新排列列，使模型名称在第一列
        cols = df.columns.tolist()
        cols.remove("model_name")
        cols = ["model_name"] + cols
        df = df[cols]
        
        # 保存到Excel
        output_path = "model_comparison.xlsx"
        df.to_excel(output_path, index=False)
        print(f"\n✅ 模型比较结果已保存至: {output_path}")
        
        # 打印简要比较
        print("\n模型准确率和训练时间比较:")
        comparison_cols = ["model_name", "accuracy", "macro_f1", "weighted_f1"]
        
        # 如果有训练时间信息，也包含在比较中
        if "total_train_time" in df.columns and not df["total_train_time"].isna().all():
            comparison_cols.extend(["total_train_time", "avg_epoch_time"])
            
        comparison = df[comparison_cols].sort_values("accuracy", ascending=False)
        print(comparison)
    else:
        print("没有找到可评估的模型")

if __name__ == "__main__":
    compare_models()