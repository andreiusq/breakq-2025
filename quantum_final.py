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
import torchvision.models


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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(96),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ]) if augment else transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(96),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(96),
        transforms.ToTensor(),
        normalize
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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ReuploadingQuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, inputs, weights):
        input_reshaped = inputs.reshape(self.n_layers, self.n_qubits)

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(input_reshaped[l, q] * torch.pi, wires=q)
                qml.RZ(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)

            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_layers, self.n_qubits)
        x = torch.atan(x)

        results = []
        for i in range(batch_size):
            qnode = qml.QNode(self.circuit, self.dev, interface='torch', diff_method='backprop')
            out = qnode(x[i], self.weight)
            results.append(torch.stack(out))

        return torch.stack(results).float()


class StrongCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class QuantumHybridNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.backbone.fc = nn.Identity()

        self.q_input_size = n_qubits * n_layers

        self.pre_quantum = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, self.q_input_size),
            nn.BatchNorm1d(self.q_input_size),
            nn.Tanh()
        )

        self.quantum = ReuploadingQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_qubits, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x_q_in = self.pre_quantum(x)
        x_quantum = self.quantum(x_q_in)
        output = self.post_quantum(x_quantum)
        return output


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup and np.random.rand() < 0.5:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds_all.append(outputs.detach().cpu())
        labels_all.append(labels.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    probs = torch.from_numpy(preds_all).softmax(dim=1).numpy()
    acc = accuracy_score(labels_all, probs.argmax(axis=1))

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = float('nan')

    return epoch_loss, acc, auc


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

    epoch_loss = running_loss / len(loader.dataset)
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    probs = torch.from_numpy(preds_all).softmax(dim=1).numpy()
    acc = accuracy_score(labels_all, probs.argmax(axis=1))

    try:
        auc = roc_auc_score(labels_all, probs, multi_class='ovr')
    except:
        auc = float('nan')

    return epoch_loss, acc, auc


class ExperimentOrchestrator:
    def __init__(self, device='cpu'):
        self.device = device
        self.results = defaultdict(lambda: defaultdict(list))

    def run_single_experiment(self, model_name, model, loaders, epochs=50, lr=0.001, class_weights=None):
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        start_time = time.time()
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, _ = train_epoch(model, loaders['train'], optimizer, criterion, self.device, use_mixup=False)
            val_loss, val_acc, _ = evaluate(model, loaders['val'], criterion, self.device)

            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train: {tr_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

            if patience_counter >= 15 and epoch > 20:
                print(f"Early stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time
        test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, self.device)
        n_params = sum(p.numel() for p in model.parameters())

        print(f"\nFINAL: Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f} | Params: {n_params:,} | Time: {training_time:.1f}s")

        return {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'best_val_acc': best_val_acc,
            'n_params': n_params,
            'time': training_time
        }

    def run_full_study(self, dh_base, num_classes, data_percentages=[0.5, 1.0], epochs=50):
        model_builders = {
            'ResNet18 Transfer': lambda: StrongCNN(num_classes, in_channels=3),
            'Quantum Hybrid (Data Re-upload)': lambda: QuantumHybridNet(num_classes, in_channels=3, n_qubits=4, n_layers=3),
        }

        results = defaultdict(lambda: defaultdict(list))

        for pct in data_percentages:
            print(f"\n{'='*80}")
            print(f" TRAINING WITH {int(pct*100)}% OF DATA")
            print(f"{'='*80}")

            dh = DataHandler(download=False, as_rgb=True)
            dh.images = {k: v.copy() for k, v in dh_base.images.items()}
            dh.labels = {k: v.copy() for k, v in dh_base.labels.items()}

            if pct < 1.0:
                dh.select_train_percentage(pct, seed=42)

            targets = dh.labels['train']
            class_counts = np.bincount(targets)
            class_weights = 1.0 / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            loaders = build_dataloaders(dh, batch_size=32, augment=True)

            for model_name, builder in model_builders.items():
                model = builder().to(self.device)
                res = self.run_single_experiment(model_name, model, loaders, epochs=epochs, lr=1e-3, class_weights=class_weights)

                results[model_name]['data_pct'].append(pct)
                results[model_name]['test_acc'].append(res['test_acc'])
                results[model_name]['test_auc'].append(res['test_auc'])
                results[model_name]['best_val_acc'].append(res['best_val_acc'])
                results[model_name]['n_params'].append(res['n_params'])
                results[model_name]['time'].append(res['time'])

        return results

    def plot_results(self, results, save_path='quantum_final_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            accs = results[model_name]['test_acc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            ax.plot(data_pcts, accs, marker=marker, label=model_name, linewidth=linewidth, markersize=10)
        ax.set_xlabel('Training Data (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Test Accuracy vs Training Data', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            aucs = results[model_name]['test_auc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            ax.plot(data_pcts, aucs, marker=marker, label=model_name, linewidth=linewidth, markersize=10)
        ax.set_xlabel('Training Data (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
        ax.set_title('Test AUC vs Training Data', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        model_names = list(results.keys())
        accs = [results[m]['test_acc'][-1] for m in model_names]
        params = [results[m]['n_params'][-1] for m in model_names]
        efficiency = [a / (p/1000) for a, p in zip(accs, params)]

        colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in model_names]
        bars = ax.bar(range(len(model_names)), efficiency, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        ax.set_ylabel('Acc / 1K Params', fontsize=13, fontweight='bold')
        ax.set_title('Parameter Efficiency', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 1]
        times = [results[m]['time'][-1] for m in model_names]
        bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        ax.set_ylabel('Training Time (s)', fontsize=13, fontweight='bold')
        ax.set_title('Training Time', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results saved to: {save_path}")
        plt.show()


def main():
    print("\n" + "="*80)
    print(" FINAL QUANTUM ML SOLUTION - BreaQ 2025")
    print(" Production-Ready Implementation")
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

    orchestrator = ExperimentOrchestrator(device=device)

    results = orchestrator.run_full_study(
        dh_base=dh,
        num_classes=num_classes,
        data_percentages=[0.5, 1.0],
        epochs=80
    )

    orchestrator.plot_results(results)

    print("\n" + "="*80)
    print(" FINAL RESULTS")
    print("="*80)

    for model_name in results.keys():
        print(f"\n{model_name}:")
        print(f"  Best Test Acc: {max(results[model_name]['test_acc']):.4f}")
        print(f"  Best Test AUC: {max(results[model_name]['test_auc']):.4f}")
        print(f"  Parameters: {results[model_name]['n_params'][-1]:,}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
