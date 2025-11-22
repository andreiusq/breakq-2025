import numpy as np
from PIL import Image
from medmnist import RetinaMNIST
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
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


def build_dataloaders(dh, batch_size=24, img_size=64):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        if split == "train":
            class_counts = np.bincount(dh.labels[split])
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[dh.labels[split]]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loaders[split] = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        else:
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return loaders


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
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


class CompactCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=3)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
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

    per_class_acc = []
    for cls in range(probs.shape[1]):
        cls_mask = labels_all == cls
        if cls_mask.sum() > 0:
            cls_acc = (probs.argmax(axis=1)[cls_mask] == cls).mean()
            per_class_acc.append(cls_acc)
        else:
            per_class_acc.append(0.0)

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, auc, probs, per_class_acc


def train_model(model_name, model, loaders, device, epochs=60, lr=0.002):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_auc, _, val_per_class = evaluate(model, loaders['val'], criterion, device)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience and epoch > 25:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    test_loss, test_acc, test_auc, test_probs, test_per_class = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"Per-class Test Acc: {[f'{x:.3f}' for x in test_per_class]}")
    print(f"Parameters: {n_params:,} | Time: {training_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'test_acc': test_acc,
        'test_auc': test_auc,
        'best_val_acc': best_val_acc,
        'n_params': n_params,
        'time': training_time,
        'model': model,
        'test_probs': test_probs,
        'test_per_class': test_per_class
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

    per_class_acc = []
    for cls in range(ensemble_probs.shape[1]):
        cls_mask = labels_all == cls
        if cls_mask.sum() > 0:
            cls_acc = (ensemble_preds[cls_mask] == cls).mean()
            per_class_acc.append(cls_acc)
        else:
            per_class_acc.append(0.0)

    try:
        auc = roc_auc_score(labels_all, ensemble_probs, multi_class='ovr')
    except:
        auc = 0.0

    return acc, auc, per_class_acc


def main():
    print("\n" + "="*80)
    print(" OPTIMIZED QUANTUM ML - BreaQ 2025")
    print(" Specialized for Severely Imbalanced Small Datasets")
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
    class_counts = np.bincount(dh.labels['train'])
    for cls, count in enumerate(class_counts):
        print(f"  Class {cls}: {count:3d} samples ({count/len(dh.labels['train'])*100:.1f}%)")

    loaders = build_dataloaders(dh, batch_size=24, img_size=64)

    results = {}
    trained_models = []

    for model_name, model_fn, lr in [
        ('Compact CNN', lambda: CompactCNN(num_classes), 0.002),
        ('Quantum Hybrid', lambda: QuantumCNN(num_classes, n_qubits=4), 0.0015),
    ]:
        model = model_fn().to(device)
        res = train_model(model_name, model, loaders, device, epochs=60, lr=lr)

        results[model_name] = res
        trained_models.append(res['model'])

    print("\n" + "="*80)
    print(" ENSEMBLE PREDICTION")
    print("="*80)

    ensemble_acc, ensemble_auc, ensemble_per_class = ensemble_predict(trained_models, loaders['test'], device)
    print(f"\nEnsemble Test Acc: {ensemble_acc:.4f} | AUC: {ensemble_auc:.4f}")
    print(f"Ensemble Per-class Acc: {[f'{x:.3f}' for x in ensemble_per_class]}")

    results['Ensemble'] = {
        'test_acc': ensemble_acc,
        'test_auc': ensemble_auc,
        'best_val_acc': 0,
        'n_params': sum(r['n_params'] for r in results.values() if 'model' in r),
        'time': sum(r['time'] for r in results.values() if 'model' in r),
        'test_per_class': ensemble_per_class
    }

    print("\n" + "="*80)
    print(" FINAL COMPARISON")
    print("="*80)

    for key, res in results.items():
        print(f"\n{key}:")
        print(f"  Test Accuracy: {res['test_acc']:.4f}")
        print(f"  Test AUC: {res['test_auc']:.4f}")
        if 'test_per_class' in res:
            print(f"  Per-class Acc: {[f'{x:.2f}' for x in res['test_per_class']]}")
        if res.get('best_val_acc', 0) > 0:
            print(f"  Best Val Acc: {res['best_val_acc']:.4f}")
        print(f"  Parameters: {res['n_params']:,}")
        print(f"  Training Time: {res['time']:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    ax = axes[2]
    per_class_data = []
    for m in models:
        if 'test_per_class' in results[m]:
            per_class_data.append(results[m]['test_per_class'])

    if per_class_data:
        x = np.arange(num_classes)
        width = 0.25
        for i, (data, model_name) in enumerate(zip(per_class_data, models)):
            offset = (i - len(per_class_data)/2) * width + width/2
            ax.bar(x + offset, data, width, label=model_name, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Per-Class Accuracy', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in range(num_classes)])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('optimized_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: optimized_results.png")
    plt.show()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
