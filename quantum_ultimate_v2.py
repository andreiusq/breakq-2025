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

    def get_class_distribution(self, split='train'):
        unique, counts = np.unique(self.labels[split], return_counts=True)
        return dict(zip(unique, counts))


def build_dataloaders(dh, batch_size=16, img_size=96):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for split in ["train", "val", "test"]:
        imgs_list = []
        for img in dh.images[split]:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            if split == "train":
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


class AdvancedQuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, inputs, weights):
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                idx = i % len(inputs)
                qml.RY(inputs[idx] * np.pi, wires=i)
                qml.RZ(weights[layer, i, 0], wires=i)

            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)

            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])

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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SEBlock(48),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            SEBlock(96),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            SEBlock(192),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class QuantumHybridCNN(nn.Module):
    def __init__(self, num_classes, n_qubits=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SEBlock(48),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            SEBlock(96),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(96, n_qubits),
            nn.Tanh()
        )

        self.quantum = AdvancedQuantumLayer(n_qubits=n_qubits, n_layers=4)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x_q_in = self.pre_quantum(x)
        x_quantum = self.quantum(x_q_in)
        output = self.post_quantum(x_quantum)
        return output


class MixupDataset:
    def __init__(self, loader, alpha=0.4):
        self.loader = loader
        self.alpha = alpha

    def __iter__(self):
        for images, labels in self.loader:
            if self.alpha > 0 and np.random.rand() < 0.5:
                lam = np.random.beta(self.alpha, self.alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size)

                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                yield mixed_images, labels_a, labels_b, lam
            else:
                yield images, labels, labels, 1.0


def train_epoch_mixup(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    mixup_loader = MixupDataset(loader, alpha=0.4)

    for imgs, labels_a, labels_b, lam in mixup_loader:
        imgs = imgs.to(device)
        labels_a = labels_a.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        total += labels_a.size(0)

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

    return running_loss / len(loader.dataset), acc, auc, probs


def train_model(model_name, model, loaders, device, class_weights, epochs=80, lr=0.001):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*5, epochs=epochs,
        steps_per_epoch=len(loaders['train']), pct_start=0.3
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)

    best_val_acc = 0
    best_model_state = None
    patience = 25
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_mixup(model, loaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_auc, _ = evaluate(model, loaders['val'], criterion, device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience and epoch > 30:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    test_loss, test_acc, test_auc, test_probs = evaluate(model, loaders['test'], criterion, device)
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
        'time': training_time,
        'model': model,
        'test_probs': test_probs
    }


def ensemble_predict(models, loader, device):
    all_probs = []
    labels_all = []

    for model in models:
        model.eval()
        probs_list = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())

                if len(labels_all) == 0:
                    labels_all.append(labels.cpu())

        all_probs.append(torch.cat(probs_list).numpy())

    labels_all = torch.cat(labels_all).numpy() if labels_all else None

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(labels_all, ensemble_preds)
    try:
        auc = roc_auc_score(labels_all, ensemble_probs, multi_class='ovr')
    except:
        auc = 0.0

    return acc, auc


def main():
    print("\n" + "="*80)
    print(" ULTIMATE QUANTUM ML SOLUTION - BreaQ 2025")
    print(" Advanced Techniques for Small Medical Datasets")
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

    print(f"\nClass Distribution:")
    dist = dh.get_class_distribution('train')
    for cls, count in dist.items():
        print(f"  Class {cls}: {count} samples ({count/len(dh.labels['train'])*100:.1f}%)")

    targets = dh.labels['train']
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    loaders = build_dataloaders(dh, batch_size=16, img_size=96)

    results = {}
    trained_models = []

    for model_name, model_fn, lr in [
        ('Improved CNN', lambda: ImprovedCNN(num_classes), 0.002),
        ('Quantum Hybrid CNN', lambda: QuantumHybridCNN(num_classes, n_qubits=6), 0.001),
    ]:
        model = model_fn().to(device)
        res = train_model(model_name, model, loaders, device, class_weights, epochs=80, lr=lr)

        results[model_name] = res
        trained_models.append(res['model'])

    print("\n" + "="*80)
    print(" ENSEMBLE PREDICTION")
    print("="*80)

    ensemble_acc, ensemble_auc = ensemble_predict(trained_models, loaders['test'], device)
    print(f"\nEnsemble Test Acc: {ensemble_acc:.4f} | AUC: {ensemble_auc:.4f}")

    results['Ensemble'] = {
        'test_acc': ensemble_acc,
        'test_auc': ensemble_auc,
        'best_val_acc': 0,
        'n_params': sum(r['n_params'] for r in results.values()),
        'time': sum(r['time'] for r in results.values())
    }

    print("\n" + "="*80)
    print(" FINAL COMPARISON")
    print("="*80)

    for key, res in results.items():
        print(f"\n{key}:")
        print(f"  Test Accuracy: {res['test_acc']:.4f}")
        print(f"  Test AUC: {res['test_auc']:.4f}")
        if res['best_val_acc'] > 0:
            print(f"  Best Val Acc: {res['best_val_acc']:.4f}")
        print(f"  Parameters: {res['n_params']:,}")
        print(f"  Training Time: {res['time']:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(results.keys())
    accs = [results[m]['test_acc'] for m in models]
    aucs = [results[m]['test_auc'] for m in models]

    ax = axes[0]
    colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' if 'CNN' in m else '#95E1D3' for m in models]
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
    plt.savefig('ultimate_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: ultimate_results.png")
    plt.show()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
