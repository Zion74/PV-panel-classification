import torch
import torch.optim as optim
from dataset import get_dataloaders
from model import (
    SimpleCNN, get_resnet18, get_resnet50, get_mobilenet_v2, 
    get_mobilenet_v3, get_alexnet, get_googlenet
)
from train import train
import os

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
        # 取消注释以训练其他模型
        # "ResNet50": models["ResNet50"],
        # "MobileNetV3": models["MobileNetV3"],
        # "AlexNet": models["AlexNet"],
        # "GoogLeNet": models["GoogLeNet"]
    }

    for name, model in selected_models.items():
        print(f"\n{'='*50}\n训练模型: {name}\n{'='*50}")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, model_name=name)

