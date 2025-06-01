import torch
import numpy as np
from torchvision import transforms, datasets
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from model import (
    SimpleCNN, get_resnet18, get_resnet50, get_mobilenet_v2, 
    get_mobilenet_v3, get_alexnet, get_googlenet
)
import os
import random
from dataset import calculate_mean_std

# 类别标签
CLASS_NAMES = ['0_ashless', '1_little_ashes', '2_all_ashes']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算训练集的均值和标准差
data_dir = "./data"
try:
    mean, std = calculate_mean_std(data_dir)
    print(f"使用计算得到的均值: {mean} 和标准差: {std}")
except Exception as e:
    print(f"计算均值和标准差失败，使用默认值: {str(e)}")
    mean, std = [0.5]*3, [0.5]*3

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # 使用计算得到的均值和标准差
])

# 初始化界面
root = tk.Tk()
root.title("光伏板沾灰识别")
root.geometry("600x700") # 调整窗口大小
root.resizable(False, False) # 禁止调整窗口大小
root.configure(bg="#f0f0f0") # 设置背景色

# 标题
title_label = tk.Label(root, text="光伏板积灰程度智能识别系统", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333333")
title_label.pack(pady=20)

img_label = tk.Label(root, bg="#ffffff", bd=2, relief="solid") # 添加边框和背景
img_label.pack(pady=10)

result_label = tk.Label(root, text="请选择模型和图片进行识别", font=("Helvetica", 16), bg="#f0f0f0", fg="#007bff")
result_label.pack(pady=20)

model = None  # 当前加载的模型对象
model_name = None  # 当前模型名称
val_dataset = None  # 验证集数据

def load_model(default_path=None):
    global model, model_name

    file_path = default_path
    if not file_path:
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    
    if not file_path:
        return

    filename = os.path.basename(file_path).lower()
    
    # 根据文件名判断模型类型
    if "resnet18" in filename:
        model = get_resnet18(num_classes=3)
        model_name = "ResNet18"
    elif "resnet50" in filename:
        model = get_resnet50(num_classes=3)
        model_name = "ResNet50"
    elif "mobilenetv2" in filename or "mobilenet_v2" in filename:
        model = get_mobilenet_v2(num_classes=3)
        model_name = "MobileNetV2"
    elif "mobilenetv3" in filename or "mobilenet_v3" in filename:
        model = get_mobilenet_v3(num_classes=3)
        model_name = "MobileNetV3"
    elif "alexnet" in filename:
        model = get_alexnet(num_classes=3)
        model_name = "AlexNet"
    elif "googlenet" in filename:
        model = get_googlenet(num_classes=3)
        model_name = "GoogLeNet"
    elif "simplecnn" in filename:
        model = SimpleCNN(num_classes=3)
        model_name = "SimpleCNN"
    else:
        messagebox.showerror("错误", "无法识别模型类型，请确保文件名包含模型名称")
        return

    try:
        model.load_state_dict(torch.load(file_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        result_label.config(text=f"✅ 已加载模型：{model_name}", fg="#28a745") # 成功加载模型后改变颜色
        # 加载验证集数据
        load_val_dataset()
    except Exception as e:
        messagebox.showerror("错误", f"加载模型失败: {str(e)}")

def load_val_dataset():
    global val_dataset
    try:
        data_dir = "./data"
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
        print(f"✅ 已加载验证集，共 {len(val_dataset)} 张图片")
    except Exception as e:
        print(f"❌ 加载验证集失败: {e}")

def load_and_predict():
    global model
    if model is None:
        messagebox.showwarning("提示", "请先加载模型文件（.pth）")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    # 显示图像
    img = Image.open(file_path).convert("RGB")
    img_display = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img_display)
    img_label.configure(image=tk_img)
    img_label.image = tk_img

    # 图像预处理并预测
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    result_label.configure(
        text=f"🖼️ 预测结果: {CLASS_NAMES[pred]} \n🔥 置信度: {confidence:.2f}",
        fg="#dc3545" if pred == 2 else ("#ffc107" if pred == 1 else "#28a745"), # 根据预测结果改变颜色
        font=("Helvetica", 18, "bold") # 预测结果字体加粗
    )

def random_val_predict():
    global model, val_dataset
    if model is None:
        messagebox.showwarning("提示", "请先加载模型文件（.pth）")
        return
        
    if val_dataset is None:
        load_val_dataset()
        if val_dataset is None:
            messagebox.showerror("错误", "无法加载验证集数据，请确保数据路径正确")
            return
    
    # 随机选择一张验证集图片
    idx = random.randint(0, len(val_dataset) - 1)
    img, label = val_dataset[idx]
    
    # 获取原始图像路径
    img_path = val_dataset.imgs[idx][0]
    true_class = val_dataset.classes[label]
    
    # 显示图像
    original_img = Image.open(img_path).convert("RGB")
    img_display = original_img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img_display)
    img_label.configure(image=tk_img)
    img_label.image = tk_img
    
    # 预测
    img = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    # 显示结果，包括真实标签和预测标签
    result_text = f"🖼️ 预测结果: {CLASS_NAMES[pred]} \n🔥 置信度: {confidence:.2f} \n📋 真实标签: {true_class}"
    result_color = "#28a745" if pred == label else "#dc3545"  # 正确为绿色，错误为红色
    
    result_label.configure(
        text=result_text,
        fg=result_color,
        font=("Helvetica", 18, "bold")
    )

# 按钮样式
button_style = {
    "font": ("Helvetica", 12, "bold"),
    "bg": "#007bff",
    "fg": "white",
    "activebackground": "#0056b3",
    "activeforeground": "white",
    "bd": 0,
    "relief": "flat",
    "padx": 15,
    "pady": 8
}

# 按钮框架
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

# 按钮
model_btn = tk.Button(button_frame, text="选择模型文件", command=load_model, **button_style)
model_btn.pack(side=tk.LEFT, padx=10)

img_btn = tk.Button(button_frame, text="选择图片识别", command=load_and_predict, **button_style)
img_btn.pack(side=tk.LEFT, padx=10)

random_btn = tk.Button(button_frame, text="随机验证集图片", command=random_val_predict, **button_style)
random_btn.pack(side=tk.LEFT, padx=10)

# 启动时尝试加载默认模型
root.after(100, lambda: load_model(default_path="checkpoint/best_model_ResNet18.pth"))

root.mainloop()

