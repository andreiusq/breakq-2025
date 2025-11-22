import numpy as np
from PIL import Image
from medmnist import RetinaMNIST
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import torchvision.transforms as transforms


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

    def select_train_percentage(self, pct, shuffle=True, seed=None):
        assert 0 < pct <= 1
        N = len(self.labels["train"])
        n_keep = int(N * pct)

        rng = np.random.default_rng(seed)
        classes, counts = np.unique(self.labels["train"], return_counts=True)

        desired = counts * pct
        per_class = np.floor(desired).astype(int)

        remainder = n_keep - per_class.sum()
        if remainder > 0:
            fracs = desired - per_class
            order = np.argsort(-fracs)
            for i in order[:remainder]:
                per_class[i] += 1

        keep_idx_list = []
        for cls, k in zip(classes, per_class):
            cls_idx = np.where(self.labels["train"] == cls)[0].copy()
            if shuffle:
                rng.shuffle(cls_idx)
            if k > 0:
                keep_idx_list.append(cls_idx[:k])

        keep_idx = np.concatenate(keep_idx_list) if keep_idx_list else np.array([], dtype=int)
        if shuffle and keep_idx.size:
            rng.shuffle(keep_idx)

        self.images["train"] = self.images["train"][keep_idx]
        self.labels["train"] = self.labels["train"][keep_idx]


def build_dataloaders(dh, batch_size=16, augment=True):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.ToTensor(),
    ]) if augment else transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    for split in ["train", "val", "test"]:
        imgs_list = []
        for img in dh.images[split]:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            if split == "train" and augment:
                img = train_transform(img)
            else:
                img = test_transform(img)
            imgs_list.append(img)

        imgs = torch.stack(imgs_list)
        labels = torch.from_numpy(dh.labels[split]).long()

        dataset = TensorDataset(imgs, labels)
        shuffle = (split == "train")
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return loaders


class SimpleQuantum(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        if x.shape[1] > self.n_qubits:
            x = x[:, :self.n_qubits]
        elif x.shape[1] < self.n_qubits:
            padding = torch.zeros(x.shape[0], self.n_qubits - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = torch.tanh(x)
        return self.qlayer(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class QuantumCNN(nn.Module):
    def __init__(self, num_classes, n_qubits=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, n_qubits),
            nn.Tanh()
        )

        self.quantum = SimpleQuantum(n_qubits=n_qubits, n_layers=2)

        self.post_quantum = nn.Sequential(
            nn.Linear(128 + n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x_flat = torch.flatten(x, 1)

        x_q_in = self.pre_quantum(x)
        x_q_out = self.quantum(x_q_in)

        x_combined = torch.cat([x_flat, x_q_out], dim=1)
        output = self.post_quantum(x_combined)

        return output


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds_all.append(outputs.cpu())
            labels_all.append(labels.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    probs = torch.from_numpy(preds_all).softmax(dim=1).numpy()
    acc = accuracy_score(labels_all, probs.argmax(axis=1))

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, auc


def train_model(model_name, model, loaders, device, epochs=150, lr=0.001):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 25
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, loaders['val'], criterion, device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*70}\n")

    return {
        'test_acc': test_acc,
        'test_auc': test_auc,
        'best_val_acc': best_val_acc,
        'n_params': n_params
    }


def main():
    print("\n" + "="*80)
    print(" SIMPLE & EFFECTIVE QUANTUM ML")
    print(" BreaQ 2025")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"\nDataset: RetinaMNIST")
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(dh.labels['train'])}")

    results = {}

    for pct in [0.5, 1.0]:
        print(f"\n{'='*80}")
        print(f" TRAINING WITH {int(pct*100)}% OF DATA")
        print(f"{'='*80}")

        dh_copy = DataHandler(download=False, as_rgb=True)
        dh_copy.images = {k: v.copy() for k, v in dh.images.items()}
        dh_copy.labels = {k: v.copy() for k, v in dh.labels.items()}

        if pct < 1.0:
            dh_copy.select_train_percentage(pct, seed=42)

        loaders = build_dataloaders(dh_copy, batch_size=16, augment=True)

        print(f"\nActual training samples: {len(dh_copy.labels['train'])}")

        for model_name, model_fn in [
            ('Simple CNN', lambda: SimpleCNN(num_classes)),
            ('Quantum CNN', lambda: QuantumCNN(num_classes, n_qubits=4))
        ]:
            model = model_fn().to(device)
            res = train_model(model_name, model, loaders, device, epochs=150, lr=0.001)

            key = f"{model_name}_{int(pct*100)}%"
            results[key] = res

    print("\n" + "="*80)
    print(" FINAL SUMMARY")
    print("="*80)

    for key, res in results.items():
        print(f"\n{key}:")
        print(f"  Test Accuracy: {res['test_acc']:.4f}")
        print(f"  Test AUC: {res['test_auc']:.4f}")
        print(f"  Best Val Acc: {res['best_val_acc']:.4f}")

    print("\n" + "="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = ['Simple CNN', 'Quantum CNN']
    percentages = ['50%', '100%']

    for idx, metric in enumerate(['test_acc', 'test_auc']):
        ax = axes[idx]

        for model in models:
            values = [results[f"{model}_{pct}"][metric] for pct in percentages]
            marker = 'D' if 'Quantum' in model else 'o'
            linewidth = 3 if 'Quantum' in model else 2
            ax.plot([50, 100], values, marker=marker, label=model, linewidth=linewidth, markersize=10)

        ax.set_xlabel('Training Data (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test ' + ('Accuracy' if metric == 'test_acc' else 'AUC'), fontsize=13, fontweight='bold')
        ax.set_title(f'Test {"Accuracy" if metric == "test_acc" else "AUC"} Comparison', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([50, 100])

    plt.tight_layout()
    plt.savefig('quantum_simple_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to: quantum_simple_results.png")
    plt.show()


if __name__ == "__main__":
    main()
