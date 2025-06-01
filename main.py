import torch
import torch.optim as optim
import pandas as pd
from dataset import get_dataloaders
from model import (
    SimpleCNN, get_resnet18, get_resnet50, get_mobilenet_v2, 
    get_mobilenet_v3, get_alexnet, get_googlenet
)
from train import train
import os
import subprocess

if __name__ == "__main__":
    data_dir = r"./data"
    # 使用计算得到的均值和标准差创建数据加载器
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=32, use_calculated_stats=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 确保checkpoint目录存在
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 定义要训练的模型
    models = {
        "SimpleCNN": SimpleCNN(),
        "ResNet18": get_resnet18(),
        "ResNet50": get_resnet50(),
        "MobileNetV2": get_mobilenet_v2(),
        "MobileNetV3": get_mobilenet_v3(),
        "AlexNet": get_alexnet(),
        "GoogLeNet": get_googlenet()
    }

    # 选择要训练的模型（可以注释掉不需要训练的模型）
    selected_models = {
        "SimpleCNN": models["SimpleCNN"],
        "ResNet18": models["ResNet18"],
        "MobileNetV2": models["MobileNetV2"],
        "ResNet50": models["ResNet50"],
        "MobileNetV3": models["MobileNetV3"],
        "AlexNet": models["AlexNet"],
        "GoogLeNet": models["GoogLeNet"]
    }

    # 创建一个列表来存储每个模型的训练时间信息
    time_results = []
    
    for name, model in selected_models.items():
        print(f"\n{'='*50}\n训练模型: {name}\n{'='*50}")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc, total_train_time, avg_epoch_time = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, model_name=name)
        
        # 将时间信息添加到结果列表中
        time_results.append({
            'model_name': name,
            'total_train_time': total_train_time,
            'avg_epoch_time': avg_epoch_time
        })
    
    # 将所有模型的训练时间信息保存到Excel文件
    time_df = pd.DataFrame(time_results)
    time_excel_path = "model_training_times.xlsx"
    time_df.to_excel(time_excel_path, index=False)
    print(f"\n✅ 所有模型的训练时间信息已保存至: {time_excel_path}")
    
    # 所有模型训练完成后，运行模型比较脚本
    print("\n所有模型训练完成，开始比较模型性能...")
    try:
        # 方法1：使用subprocess运行脚本
        subprocess.run(["python", "compare_models.py"], check=True)
    except Exception as e:
        print(f"运行模型比较脚本时出错: {e}")
        # 方法2：如果subprocess失败，直接导入并运行
        try:
            from compare_models import compare_models
            compare_models()
        except Exception as e2:
            print(f"导入并运行模型比较函数时出错: {e2}")
    
    print("\n程序执行完毕！请查看以下文件获取结果：")
    print("1. model_comparison.xlsx - 所有模型的性能比较结果")
    print("2. model_training_times.xlsx - 所有模型的训练时间信息")
    print("3. 各模型results目录下的training_results.xlsx - 每个模型的详细训练过程和时间信息")

