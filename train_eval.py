import torch
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, model_name="model"):
    model.to(device)
    best_acc = 0.0  # 初始化最佳准确率

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        # 每轮评估一次验证集，并保存最优模型
        acc = evaluate_model(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            save_path = f"best_model_{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ 新最佳模型已保存: {save_path}，验证准确率: {acc:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # 分类报告
    print(classification_report(y_true, y_pred, target_names=["无灰尘", "有些灰尘", "布满灰尘"]))

    # ROC 曲线
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], [x[i] for x in y_probs])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # 计算准确率并返回（用于保存模型）
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    return correct / len(y_true)
