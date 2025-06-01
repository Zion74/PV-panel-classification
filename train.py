import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
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

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        # 计算平均训练损失和准确率
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
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
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 保存最优模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_path = f"checkpoint/best_model_{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ 新最佳模型已保存: {save_path}，验证准确率: {val_accuracy:.4f}")
    
    # 保存训练结果为Excel
    results_df = pd.DataFrame({
        'Epoch': epochs_list,
        'Train Loss': train_losses,
        'Train Accuracy': train_accs,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accs
    })
    
    excel_path = f"{results_dir}/training_results.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"✅ 训练结果已保存至: {excel_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='训练损失')
    plt.plot(epochs_list, val_losses, 'r-', label='验证损失')
    plt.title(f'{model_name} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs_list, val_accs, 'r-', label='验证准确率')
    plt.title(f'{model_name} - 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png")
    plt.close()
    print(f"✅ 训练曲线已保存至: {results_dir}/training_curves.png")
    
    return best_acc

