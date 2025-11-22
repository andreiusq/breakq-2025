import numpy as np
from medmnist import RetinaMNIST
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    def compute_normalization_stats(self):
        train_imgs = self.images['train'].astype(np.float32) / 255.0
        mean = train_imgs.mean(axis=(0, 1, 2))
        std = train_imgs.std(axis=(0, 1, 2))
        return mean, std


def build_dataloaders(dh, batch_size=32, img_size=28, mean=None, std=None):
    if mean is None:
        mean, std = dh.compute_normalization_stats()

    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
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


class VectorizedQuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def _circuit(inputs, weights):
            batch_size = inputs.shape[0] if inputs.ndim > 1 else 1

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    idx = i % inputs.shape[-1]
                    qml.RY(inputs[..., idx] * np.pi, wires=i)
                    qml.RZ(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)

                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = _circuit

    def forward(self, x):
        x = torch.tanh(x)
        out = self.qnode(x, self.weight)
        if isinstance(out, list):
            out = torch.stack(out, dim=-1)
        return out.float()


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
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 48),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(48),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class QuantumResidualCNN(nn.Module):
    def __init__(self, num_classes, n_qubits=6):
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

        self.classical_bypass = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_qubits)
        )

        self.quantum = VectorizedQuantumLayer(n_qubits=n_qubits, n_layers=3)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        x_classical = self.classical_bypass(x)

        x_q_in = self.pre_quantum(x)
        x_quantum = self.quantum(x_q_in)
        x_quantum_out = self.post_quantum(x_quantum)

        x_combined = x_classical + x_quantum_out

        output = self.classifier(x_combined)
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
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='macro')
    kappa = cohen_kappa_score(labels_all, preds, weights='quadratic')

    per_class_acc = []
    for cls in range(probs.shape[1]):
        cls_mask = labels_all == cls
        if cls_mask.sum() > 0:
            cls_acc = (preds[cls_mask] == cls).mean()
            per_class_acc.append(cls_acc)
        else:
            per_class_acc.append(0.0)

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, f1, kappa, auc, probs, per_class_acc


def train_model(model_name, model, loaders, device, epochs=50, lr=0.002):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_kappa = -1
    best_model_state = None
    patience = 18
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_kappa, val_auc, _, val_per_class = evaluate(model, loaders['val'], criterion, device)

        scheduler.step(val_kappa)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Kappa: {val_kappa:.4f} | Best: {best_val_kappa:.4f}")

        if patience_counter >= patience and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    test_loss, test_acc, test_f1, test_kappa, test_auc, test_probs, test_per_class = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Kappa: {test_kappa:.4f} | AUC: {test_auc:.4f}")
    print(f"Per-class Acc: {[f'{x:.3f}' for x in test_per_class]}")
    print(f"Parameters: {n_params:,} | Time: {training_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_kappa': test_kappa,
        'test_auc': test_auc,
        'best_val_kappa': best_val_kappa,
        'n_params': n_params,
        'time': training_time,
        'model': model,
        'test_probs': test_probs,
        'test_per_class': test_per_class
    }


def ensemble_predict(models, loader, device):
    all_probs = []
    labels_all = []

    for idx, model in enumerate(models):
        model.eval()
        probs_list = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())

                if idx == 0:
                    labels_all.append(labels.cpu())

        all_probs.append(torch.cat(probs_list).numpy())

    labels_all = torch.cat(labels_all).numpy()

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(labels_all, ensemble_preds)
    f1 = f1_score(labels_all, ensemble_preds, average='macro')
    kappa = cohen_kappa_score(labels_all, ensemble_preds, weights='quadratic')

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

    return acc, f1, kappa, auc, per_class_acc


def main():
    set_seed(42)

    print("\n" + "="*80)
    print(" QUANTUM-CLASSICAL HYBRID ML - BreaQ 2025")
    print(" Optimized for Medical Imaging with Severe Class Imbalance")
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

    loaders = build_dataloaders(dh, batch_size=32, img_size=28)

    results = {}
    trained_models = []

    for model_name, model_fn, lr in [
        ('Classical CNN', lambda: CompactCNN(num_classes), 0.002),
        ('Quantum-Classical Hybrid', lambda: QuantumResidualCNN(num_classes, n_qubits=6), 0.0015),
    ]:
        model = model_fn().to(device)
        res = train_model(model_name, model, loaders, device, epochs=50, lr=lr)

        results[model_name] = res
        trained_models.append(res['model'])

    print("\n" + "="*80)
    print(" ENSEMBLE PREDICTION")
    print("="*80)

    ensemble_acc, ensemble_f1, ensemble_kappa, ensemble_auc, ensemble_per_class = ensemble_predict(trained_models, loaders['test'], device)
    print(f"\nEnsemble Results:")
    print(f"  Acc: {ensemble_acc:.4f} | F1: {ensemble_f1:.4f} | Kappa: {ensemble_kappa:.4f} | AUC: {ensemble_auc:.4f}")
    print(f"  Per-class Acc: {[f'{x:.3f}' for x in ensemble_per_class]}")

    results['Ensemble'] = {
        'test_acc': ensemble_acc,
        'test_f1': ensemble_f1,
        'test_kappa': ensemble_kappa,
        'test_auc': ensemble_auc,
        'n_params': sum(r['n_params'] for r in results.values() if 'model' in r),
        'time': sum(r['time'] for r in results.values() if 'model' in r),
        'test_per_class': ensemble_per_class
    }

    print("\n" + "="*80)
    print(" FINAL RESULTS - RANKED BY KAPPA (Medical Imaging Standard)")
    print("="*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_kappa'], reverse=True)

    for key, res in sorted_results:
        print(f"\n{key}:")
        print(f"  Kappa: {res['test_kappa']:.4f} (primary metric)")
        print(f"  Accuracy: {res['test_acc']:.4f}")
        print(f"  F1-Macro: {res['test_f1']:.4f}")
        print(f"  AUC: {res['test_auc']:.4f}")
        if 'test_per_class' in res:
            print(f"  Per-class: {[f'{x:.2f}' for x in res['test_per_class']]}")
        print(f"  Params: {res['n_params']:,} | Time: {res['time']:.1f}s")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results.keys())
    accs = [results[m]['test_acc'] for m in models]
    kappas = [results[m]['test_kappa'] for m in models]
    f1s = [results[m]['test_f1'] for m in models]
    aucs = [results[m]['test_auc'] for m in models]

    colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' if 'Classical' in m else '#95E1D3' for m in models]

    for ax, metric_values, metric_name in zip(axes.flat[:3],
                                                [kappas, accs, f1s],
                                                ['Quadratic Kappa', 'Accuracy', 'F1-Macro']):
        bars = ax.bar(range(len(models)), metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9, fontweight='bold', rotation=15, ha='right')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1, 1]
    per_class_data = []
    model_names_filtered = []
    for m in models:
        if 'test_per_class' in results[m]:
            per_class_data.append(results[m]['test_per_class'])
            model_names_filtered.append(m)

    if per_class_data:
        x = np.arange(num_classes)
        width = 0.25
        for i, (data, model_name) in enumerate(zip(per_class_data, model_names_filtered)):
            offset = (i - len(per_class_data)/2) * width + width/2
            ax.bar(x + offset, data, width, label=model_name[:15], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in range(num_classes)])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: final_results.png")
    plt.show()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
