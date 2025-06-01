import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

def calculate_mean_std(data_dir):
    """
    计算训练集图像的均值和标准差
    """
    # 创建一个简单的转换，只调整大小和转换为张量，不进行标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 加载训练集
    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 初始化变量
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    # 第一次遍历计算均值
    for images, _ in train_loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    
    # 第二次遍历计算标准差
    for images, _ in train_loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        std += ((images - mean.unsqueeze(1).unsqueeze(2))**2).sum([0, 2])
    
    std = torch.sqrt(std / (total_images * 224 * 224))
    
    print(f"计算得到的均值: {mean.tolist()}")
    print(f"计算得到的标准差: {std.tolist()}")
    
    return mean.tolist(), std.tolist()

def get_dataloaders(data_dir, batch_size=32, use_calculated_stats=True):
    """
    创建数据加载器，可选择是否使用计算得到的均值和标准差
    """
    # 如果选择使用计算得到的统计数据
    if use_calculated_stats:
        # 检查是否已经计算过均值和标准差
        mean, std = calculate_mean_std(data_dir)
    else:
        # 使用默认值 [0.5, 0.5, 0.5]
        mean, std = [0.5]*3, [0.5]*3
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        train_ds.classes
    )
