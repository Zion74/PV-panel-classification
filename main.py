import torch
import torch.optim as optim
from dataset import get_dataloaders
from model import SimpleCNN, get_resnet, get_mobilenet
from train_eval import train_model

if __name__ == "__main__":
    data_dir = r"./data"
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "SimpleCNN": SimpleCNN(),
        "ResNet18": get_resnet(),
        "MobileNetV2": get_mobilenet()
    }

    for name, model in models.items():
        print(f"\nTraining: {name}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5, model_name=name)

