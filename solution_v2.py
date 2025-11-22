import numpy as np
from medmnist import RetinaMNIST
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torchvision.transforms as transforms
import torchvision.models as models


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        return focal_loss


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


def build_dataloaders(dh, batch_size=32, img_size=128, mean=None, std=None):
    if mean is None:
        mean, std = dh.compute_normalization_stats()

    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(img_size, scale=(0.82, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=10, translate=(0.07, 0.07), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25)], p=0.7),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4)),
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


class ImprovedQuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.weight = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def _circuit(inputs, weights):
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    idx = i % inputs.shape[-1]
                    qml.RY(inputs[..., idx] * np.pi, wires=i)

                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = _circuit

    def forward(self, x):
        x = torch.tanh(x)
        out = self.qnode(x, self.weight)
        if isinstance(out, list):
            out = torch.stack(out, dim=-1)
        return out.float()


class ResNetBackbone(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            weights = None

        self.backbone = models.resnet18(weights=weights if pretrained else None)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in list(self.backbone.layer3.parameters()) + list(self.backbone.layer4.parameters()):
            param.requires_grad = True

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class QuantumResNet(nn.Module):
    def __init__(self, num_classes, n_qubits=6, pretrained=True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            weights = None

        self.backbone = models.resnet18(weights=weights if pretrained else None)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in list(self.backbone.layer3.parameters()) + list(self.backbone.layer4.parameters()):
            param.requires_grad = True

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classical_path = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        self.quantum_path = nn.Sequential(
            nn.Linear(num_features, n_qubits),
            nn.Tanh()
        )

        self.quantum = ImprovedQuantumLayer(n_qubits=n_qubits, n_layers=3)

        self.quantum_post = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        x_classical = self.classical_path(features)

        x_q_in = self.quantum_path(features)
        x_quantum = self.quantum(x_q_in)
        x_quantum_out = self.quantum_post(x_quantum)

        x_combined = x_classical + x_quantum_out

        output = self.classifier(x_combined)
        return output


def mixup_batch(images, labels, alpha=0.3, prob=0.65):
    if alpha <= 0 or np.random.rand() > prob:
        return images, labels, labels, 1.0, False

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    labels_a, labels_b = labels, labels[perm]
    return mixed, labels_a, labels_b, lam, True


def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.3):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        imgs, labels_a, labels_b, lam, used_mixup = mixup_batch(imgs, labels, alpha=mixup_alpha)
        optimizer.zero_grad()
        outputs = model(imgs)
        if used_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        if used_mixup:
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
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

    ece = np.abs(probs.max(axis=1) - (preds == labels_all).astype(float)).mean()

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = 0.0

    cm = confusion_matrix(labels_all, preds)

    return running_loss / len(loader.dataset), acc, f1, kappa, auc, ece, probs, per_class_acc, cm


def train_model(model_name, model, loaders, device, class_weights, epochs=50, lr=0.002, mixup_alpha=0.3):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    criterion = FocalLoss(gamma=2, weight=class_weights, label_smoothing=0.1)

    best_val_kappa = -1
    best_model_state = None
    patience = 18
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, device, mixup_alpha=mixup_alpha)
        val_loss, val_acc, val_f1, val_kappa, val_auc, val_ece, _, val_per_class, _ = evaluate(model, loaders['val'], criterion, device)

        scheduler.step(val_kappa)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f} | Best: {best_val_kappa:.4f}")

        if patience_counter >= patience and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    test_loss, test_acc, test_f1, test_kappa, test_auc, test_ece, test_probs, test_per_class, test_cm = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Kappa: {test_kappa:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f} | ECE: {test_ece:.4f}")
    print(f"Per-class: {[f'{x:.3f}' for x in test_per_class]}")
    print(f"Params: {n_params:,} | Time: {training_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_kappa': test_kappa,
        'test_auc': test_auc,
        'test_ece': test_ece,
        'best_val_kappa': best_val_kappa,
        'n_params': n_params,
        'time': training_time,
        'model': model,
        'test_probs': test_probs,
        'test_per_class': test_per_class,
        'test_cm': test_cm
    }


def weighted_ensemble_predict(models, model_weights, loader, device):
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

    ensemble_probs = np.average(all_probs, axis=0, weights=model_weights)
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

    ece = np.abs(ensemble_probs.max(axis=1) - (ensemble_preds == labels_all).astype(float)).mean()

    try:
        auc = roc_auc_score(labels_all, ensemble_probs, multi_class='ovr')
    except:
        auc = 0.0

    cm = confusion_matrix(labels_all, ensemble_preds)

    return acc, f1, kappa, auc, ece, per_class_acc, cm


def main():
    set_seed(42)

    print("\n" + "="*80)
    print(" ADVANCED QUANTUM-CLASSICAL HYBRID - BreaQ 2025")
    print(" Focal Loss + ResNet18 + Optimized Quantum Layer")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"\nDataset: RetinaMNIST")
    print(f"  Train: {len(dh.labels['train'])} | Val: {len(dh.labels['val'])} | Test: {len(dh.labels['test'])}")

    print(f"\nClass Distribution:")
    class_counts = np.bincount(dh.labels['train'])
    for cls, count in enumerate(class_counts):
        print(f"  Class {cls}: {count:3d} ({count/len(dh.labels['train'])*100:.1f}%)")

    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-3)).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    loaders = build_dataloaders(dh, batch_size=32, img_size=128)

    results = {}
    trained_models = []
    model_kappas = []

    for model_name, model_fn, lr, mixup in [
        ('ResNet18-Baseline', lambda: ResNetBackbone(num_classes, pretrained=True), 0.0012, 0.35),
        ('Quantum-ResNet18', lambda: QuantumResNet(num_classes, n_qubits=6, pretrained=True), 0.0010, 0.30),
    ]:
        model = model_fn().to(device)
        res = train_model(model_name, model, loaders, device, class_weights, epochs=60, lr=lr, mixup_alpha=mixup)

        results[model_name] = res
        trained_models.append(res['model'])
        model_kappas.append(max(res['best_val_kappa'], 0.0))

    print("\n" + "="*80)
    print(" WEIGHTED ENSEMBLE (by validation Kappa)")
    print("="*80)

    kappa_sum = max(sum(model_kappas), 1e-6)
    ensemble_weights = [k / kappa_sum for k in model_kappas]
    print(f"Weights: {[f'{w:.3f}' for w in ensemble_weights]}")

    ens_acc, ens_f1, ens_kappa, ens_auc, ens_ece, ens_per_class, ens_cm = weighted_ensemble_predict(
        trained_models, ensemble_weights, loaders['test'], device
    )

    print(f"\nEnsemble: Kappa={ens_kappa:.4f} | Acc={ens_acc:.4f} | F1={ens_f1:.4f} | AUC={ens_auc:.4f} | ECE={ens_ece:.4f}")
    print(f"Per-class: {[f'{x:.3f}' for x in ens_per_class]}")

    results['Weighted-Ensemble'] = {
        'test_acc': ens_acc,
        'test_f1': ens_f1,
        'test_kappa': ens_kappa,
        'test_auc': ens_auc,
        'test_ece': ens_ece,
        'test_per_class': ens_per_class,
        'test_cm': ens_cm,
        'n_params': sum(r['n_params'] for r in results.values() if 'model' in r),
        'time': sum(r['time'] for r in results.values() if 'model' in r)
    }

    print("\n" + "="*80)
    print(" FINAL RESULTS (Ranked by Kappa)")
    print("="*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_kappa'], reverse=True)

    for key, res in sorted_results:
        print(f"\n{key}:")
        print(f"  Kappa: {res['test_kappa']:.4f} *")
        print(f"  Acc: {res['test_acc']:.4f} | F1: {res['test_f1']:.4f} | AUC: {res['test_auc']:.4f} | ECE: {res['test_ece']:.4f}")
        if 'test_per_class' in res:
            print(f"  Per-class: {[f'{x:.2f}' for x in res['test_per_class']]}")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    models = list(results.keys())
    kappas = [results[m]['test_kappa'] for m in models]
    accs = [results[m]['test_acc'] for m in models]
    f1s = [results[m]['test_f1'] for m in models]
    eces = [results[m]['test_ece'] for m in models]

    colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' if 'ResNet' in m else '#95E1D3' for m in models]

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(range(len(models)), kappas, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[:15] for m in models], fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('Quadratic Kappa', fontsize=11, fontweight='bold')
    ax.set_title('Primary Metric: Kappa', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(models)), accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[:15] for m in models], fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(len(models)), eces, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[:15] for m in models], fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('ECE (lower=better)', fontsize=11, fontweight='bold')
    ax.set_title('Expected Calibration Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, :])
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
            ax.bar(x + offset, data, width, label=model_name[:20], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Accuracy (Critical for Imbalanced Data)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}\n({class_counts[i]})' for i in range(num_classes)], fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

    for idx, (key, res) in enumerate(sorted_results[:2]):
        cm = res['test_cm']
        ax = fig.add_subplot(gs[2, idx])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{key[:20]}\nConfusion Matrix', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)

    plt.savefig('advanced_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: advanced_results.png")
    plt.show()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
