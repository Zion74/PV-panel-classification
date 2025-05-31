import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from model import get_resnet, get_mobilenet, SimpleCNN
import os

# 类别标签
CLASS_NAMES = ['0_ashless', '1_little_ashes', '2_all_ashes']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 初始化界面
root = tk.Tk()
root.title("光伏板沾灰识别")
root.geometry("550x600")

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="请选择模型和图片进行识别", font=("Helvetica", 14))
result_label.pack(pady=20)

model = None  # 当前加载的模型对象
model_name = None  # 当前模型名称

def load_model():
    global model, model_name

    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    if not file_path:
        return

    filename = os.path.basename(file_path).lower()
    if "resnet" in filename:
        model = get_resnet(num_classes=3)
        model_name = "ResNet18"
    elif "mobilenet" in filename:
        model = get_mobilenet(num_classes=3)
        model_name = "MobileNetV2"
    elif "simplecnn" in filename:
        model = SimpleCNN(num_classes=3)
        model_name = "SimpleCNN"
    else:
        messagebox.showerror("错误", "无法识别模型类型，请确保文件名包含：resnet / mobilenet / simplecnn")
        return

    model.load_state_dict(torch.load(file_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    result_label.config(text=f"✅ 已加载模型：{model_name}")

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
        fg="green", font=("Helvetica", 16)
    )

# 按钮
model_btn = tk.Button(root, text="选择模型文件", command=load_model, font=("Helvetica", 12))
model_btn.pack(pady=5)

img_btn = tk.Button(root, text="选择图片识别", command=load_and_predict, font=("Helvetica", 12))
img_btn.pack(pady=5)

root.mainloop()

