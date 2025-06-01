import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, roc_curve, auc

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    # 计算准确率
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct / len(y_true)
    
    # 打印分类报告
    if len(y_true) > 0:
        print(classification_report(y_true, y_pred, target_names=["无灰尘", "有些灰尘", "布满灰尘"]))
    
    return accuracy


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, model_name="model"):
    model.to(device)
    best_acc = 0.0  # 初始化最佳准确率
    
    # 创建保存结果的文件夹
    results_dir = f"results/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化记录训练过程的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_list = []
    epoch_times = []  # 记录每个epoch的训练时间
    
    # 记录总训练开始时间
    total_start_time = time.time()

    for epoch in range(epochs):
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 计算训练准确率
            predicted = torch.argmax(out, dim = 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        # 计算平均训练损失和准确率
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = evaluate_model(model, val_loader, device)   

        # 记录结果
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)
        
        # 计算并记录本epoch的训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"Model {model_name}, Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")

        # 保存最优模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_path = f"checkpoint/best_model_{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ 新最佳模型已保存: {save_path}，验证准确率: {val_accuracy:.4f}")
    
    # 计算总训练时间和平均每个epoch的时间
    total_train_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    print(f"总训练时间: {total_train_time:.2f}s, 平均每个epoch时间: {avg_epoch_time:.2f}s")
    
    # 保存训练结果为Excel
    results_df = pd.DataFrame({
        'Epoch': epochs_list,
        'Train Loss': train_losses,
        'Train Accuracy': train_accs,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accs
    })
    
    # 创建时间信息的DataFrame
    time_df = pd.DataFrame({
        'Epoch': epochs_list,
        'Epoch时间(秒)': epoch_times
    })
    
    # 创建总结时间信息的DataFrame
    time_summary_df = pd.DataFrame({
        '总训练时间(秒)': [total_train_time],
        '平均每个Epoch时间(秒)': [avg_epoch_time]
    })
    
    excel_path = f"{results_dir}/training_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        results_df.to_excel(writer, sheet_name='Metrics', index=False)
        time_df.to_excel(writer, sheet_name='Time Details', index=False)
        time_summary_df.to_excel(writer, sheet_name='Time Info', index=False)
    print(f"✅ 训练结果已保存至: {excel_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    # 设置字体以避免中文乱码和Times New Roman字体缺失问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica'] # 或者其他通用字体，如 'Arial'
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    # 1. Loss曲线（使用默认配色）
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs_list, val_losses, label='Validation Loss', marker='s', markersize=4)
    plt.title(f'{model_name} - Loss Curves', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Accuracy曲线（使用默认配色）
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_accs, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(epochs_list, val_accs, label='Validation Accuracy', marker='s', markersize=4)
    plt.title(f'{model_name} - Accuracy Curves', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"results/{model_name}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存至: {results_dir}/training_curves.png")
    
    return best_acc, total_train_time, avg_epoch_time

