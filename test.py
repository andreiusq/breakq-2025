import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from medmnist import RetinaMNIST
import matplotlib.pyplot as plt
from torchvision import models


class DataHandler:
    def __init__(self, download=True, as_rgb=True):
        self.images = {}
        self.labels = {}

        for split in ["train", "val", "test"]:
            ds = RetinaMNIST(split=split, download=download)
            imgs = ds.imgs
            labels = ds.labels.squeeze()

            if imgs.ndim == 3:
                if as_rgb:
                    imgs = np.stack([imgs]*3, axis=-1)
                else:
                    imgs = imgs[..., None]

            self.images[split] = imgs.astype(np.uint8)
            self.labels[split] = labels.astype(int)

    def resize(self, size):
        for split in ["train", "val", "test"]:
            resized = []
            for img in self.images[split]:
                if img.shape[-1] == 1:
                    pil = Image.fromarray(img.squeeze(), mode="L")
                else:
                    pil = Image.fromarray(img)
                pil = pil.resize(size, Image.BILINEAR)
                arr = np.array(pil)
                if arr.ndim == 2:
                    arr = arr[..., None]
                resized.append(arr)
            self.images[split] = np.stack(resized, axis=0)



def build_dataloaders(dh, batch_size=32):
    loaders = {}

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    for split in ["train", "val", "test"]:
        imgs = torch.from_numpy(dh.images[split]).float()
        imgs = imgs.permute(0,3,1,2) / 255.0
        imgs = (imgs - mean) / std

        labels = torch.from_numpy(dh.labels[split]).long()
        dataset = TensorDataset(imgs, labels)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2
        )

    return loaders



class RetinaResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



def train_model(model, loaders, epochs=30, lr=0.0003, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loaders["train"])
        history["train_loss"].append(avg_loss)

        # VALIDATION
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history



def plot_training(history):
    epochs = range(1, len(history['train_loss'])+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history['val_acc'])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def show_predictions(model, loader, device="cpu", max_images=16):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    plt.figure(figsize=(12,12))

    for i in range(min(max_images, len(images))):
        img = images[i].permute(1,2,0).cpu()
        img = img * torch.tensor([0.229,0.224,0.225]) + torch.tensor([0.485,0.456,0.406])
        img = img.numpy().clip(0,1)

        gt = labels[i].item()
        pred = preds[i].item()
        prob = probs[i][pred].item()

        color = 'green' if gt == pred else 'red'

        plt.subplot(4,4,i+1)
        plt.imshow(img)
        plt.title(f"GT:{gt} P:{pred}\n{prob:.2f}", color=color)
        plt.axis('off')

    plt.suptitle("Predicții ResNet18 pe RetinaMNIST")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device folosit: {device}")

    dh = DataHandler()
    dh.resize((224, 224))

    loaders = build_dataloaders(dh, batch_size=32)

    num_classes = len(np.unique(dh.labels['train']))
    model = RetinaResNet(num_classes).to(device)

    print("\n===== ANTRENARE MODEL RESNET18 =====")
    history = train_model(model, loaders, epochs=30, lr=0.0003, device=device)

    plot_training(history)

    print("\n===== VIZUALIZARE PREDICȚII =====")
    show_predictions(model, loaders['val'], device=device)