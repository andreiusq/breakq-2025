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


def build_dataloaders(dh, batch_size=32, augment=True):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
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


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.tanh(x)

        results = []
        for i in range(batch_size):
            qnode = qml.QNode(self.circuit, self.dev, interface='torch', diff_method='backprop')
            out = qnode(x[i], self.weight)
            results.append(torch.stack(out))

        return torch.stack(results).float()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
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
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=2)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_qubits, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x_q_in = self.pre_quantum(x)
        x_quantum = self.quantum(x_q_in)
        output = self.post_quantum(x_quantum)
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


def train_model(model_name, model, loaders, device, class_weights, epochs=100, lr=0.001):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_acc = 0
    patience = 20
    patience_counter = 0
    start_time = time.time()

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

        if patience_counter >= patience and epoch > 30:
            print(f"Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"Parameters: {n_params:,} | Time: {training_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'test_acc': test_acc,
        'test_auc': test_auc,
        'best_val_acc': best_val_acc,
        'n_params': n_params,
        'time': training_time
    }


def main():
    print("\n" + "="*80)
    print(" SIMPLE & EFFECTIVE QUANTUM ML - BreaQ 2025")
    print(" Designed for Small Medical Imaging Datasets")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"\nDataset: RetinaMNIST")
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(dh.labels['train'])}")
    print(f"  Val: {len(dh.labels['val'])}")
    print(f"  Test: {len(dh.labels['test'])}")

    results = {}

    for pct in [1.0]:
        print(f"\n{'='*80}")
        print(f" TRAINING WITH {int(pct*100)}% OF DATA")
        print(f"{'='*80}")

        dh_copy = DataHandler(download=False, as_rgb=True)
        dh_copy.images = {k: v.copy() for k, v in dh.images.items()}
        dh_copy.labels = {k: v.copy() for k, v in dh.labels.items()}

        if pct < 1.0:
            dh_copy.select_train_percentage(pct, seed=42)

        targets = dh_copy.labels['train']
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)

        loaders = build_dataloaders(dh_copy, batch_size=32, augment=True)

        for model_name, model_fn in [
            ('Simple CNN', lambda: SimpleCNN(num_classes)),
            ('Quantum Hybrid CNN', lambda: QuantumCNN(num_classes, n_qubits=4))
        ]:
            model = model_fn().to(device)
            res = train_model(model_name, model, loaders, device, class_weights, epochs=100, lr=0.001)

            key = f"{model_name}"
            results[key] = res

    print("\n" + "="*80)
    print(" FINAL COMPARISON")
    print("="*80)

    for key, res in results.items():
        print(f"\n{key}:")
        print(f"  Test Accuracy: {res['test_acc']:.4f}")
        print(f"  Test AUC: {res['test_auc']:.4f}")
        print(f"  Best Val Acc: {res['best_val_acc']:.4f}")
        print(f"  Parameters: {res['n_params']:,}")
        print(f"  Training Time: {res['time']:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    accs = [results[m]['test_acc'] for m in models]
    aucs = [results[m]['test_auc'] for m in models]

    ax = axes[0]
    colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in models]
    bars = ax.bar(range(len(models)), accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Test Accuracy Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(range(len(models)), aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.set_title('Test AUC Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Results saved to: final_results.png")
    plt.show()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
