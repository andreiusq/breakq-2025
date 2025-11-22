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
from typing import Tuple, Dict, List
import time
from collections import defaultdict
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


def build_dataloaders(dh, batch_size=64, augment=True, num_workers=4):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    return loaders


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
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


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[n_qubits-1, 0])

                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 2], wires=i)

            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        batch_size = x.shape[0]

        n_features = x.shape[1]
        target_size = 2 ** self.n_qubits

        if n_features < target_size:
            padding = torch.zeros(batch_size, target_size - n_features, device=x.device)
            x = torch.cat([x, padding], dim=1)
        else:
            x = x[:, :target_size]

        x = F.normalize(x, p=2, dim=1)

        output = self.qlayer(x)

        if output.dim() > 2:
            output = output.reshape(batch_size, -1)

        if output.shape[1] < self.n_qubits:
            output = F.pad(output, (0, self.n_qubits - output.shape[1]))
        elif output.shape[1] > self.n_qubits:
            output = output[:, :self.n_qubits]

        return output


class PowerfulResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 4, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_se=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class QuantumHybridGPU(nn.Module):
    def __init__(self, num_classes, in_channels=3, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.pre_quantum = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        self.post_quantum = nn.Sequential(
            nn.Linear(256 + n_qubits, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_se=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_pool = self.avgpool(x)
        x_flat = torch.flatten(x_pool, 1)

        x_pre_q = self.pre_quantum(x_flat)
        x_quantum = self.quantum(x_pre_q)

        x_combined = torch.cat([x_flat, x_quantum], dim=1)
        x_out = self.post_quantum(x_combined)

        return x_out


class QuantumDualPath(nn.Module):
    def __init__(self, num_classes, in_channels=3, n_qubits=8):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64, use_se=True)
        )

        self.classical_path = nn.Sequential(
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.quantum_path = nn.Sequential(
            self._make_layer(64, 128, 2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=3)

        self.fusion = nn.Sequential(
            nn.Linear(256 + n_qubits, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_se=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_shared = self.conv_shared(x)

        x_classical = self.classical_path(x_shared)

        x_q_features = self.quantum_path(x_shared)
        x_quantum = self.quantum(x_q_features)

        x_fused = torch.cat([x_classical, x_quantum], dim=1)
        output = self.fusion(x_fused)

        return output


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
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
    def __init__(self, device='cuda'):
        self.device = device
        self.results = defaultdict(lambda: defaultdict(list))

    def run_single_experiment(self, model_name, model, loaders, epochs=100, lr=2e-3):
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 3,
            epochs=epochs,
            steps_per_epoch=len(loaders['train']),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.cuda.amp.GradScaler()

        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        start_time = time.time()
        best_val_acc = 0
        patience = 20
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_auc = train_epoch(model, loaders['train'], optimizer, criterion, self.device, scaler)
            val_loss, val_acc, val_auc = evaluate(model, loaders['val'], criterion, self.device)

            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['train_auc'].append(tr_auc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train: {tr_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= patience and epoch > 50:
                print(f"Early stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time

        test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, self.device)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {model_name}")
        print(f"{'='*70}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC:      {test_auc:.4f}")
        print(f"  Best Val Acc:  {best_val_acc:.4f}")
        print(f"  Parameters:    {n_params:,}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"{'='*70}")

        return {
            'history': history,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'best_val_acc': best_val_acc,
            'n_params': n_params,
            'training_time': training_time
        }

    def run_full_study(self, dh_base, num_classes, data_percentages=[0.25, 0.5, 1.0], epochs=100):
        print("\n" + "="*80)
        print(" GPU-ACCELERATED QUANTUM ML EXPERIMENT")
        print(" RTX 4050 Optimized Training")
        print("="*80)

        model_builders = {
            'Powerful ResNet': lambda: PowerfulResNet(num_classes, in_channels=3),
            'Quantum Hybrid GPU': lambda: QuantumHybridGPU(num_classes, in_channels=3, n_qubits=8, n_layers=4),
            'Quantum Dual-Path': lambda: QuantumDualPath(num_classes, in_channels=3, n_qubits=8),
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

            loaders = build_dataloaders(dh, batch_size=64, augment=True, num_workers=4)

            for model_name, builder in model_builders.items():
                torch.cuda.empty_cache()

                model = builder().to(self.device)
                res = self.run_single_experiment(model_name, model, loaders, epochs=epochs, lr=2e-3)

                results[model_name]['data_pct'].append(pct)
                results[model_name]['test_acc'].append(res['test_acc'])
                results[model_name]['test_auc'].append(res['test_auc'])
                results[model_name]['best_val_acc'].append(res['best_val_acc'])
                results[model_name]['n_params'].append(res['n_params'])
                results[model_name]['time'].append(res['training_time'])

                del model
                torch.cuda.empty_cache()

        return results

    def plot_results(self, results, save_path='quantum_gpu_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            accs = results[model_name]['test_acc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            markersize = 10 if 'Quantum' in model_name else 8
            ax.plot(data_pcts, accs, marker=marker, label=model_name, linewidth=linewidth, markersize=markersize)
        ax.set_xlabel('Training Data (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Test Accuracy vs Training Data Size', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])

        ax = axes[0, 1]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            aucs = results[model_name]['test_auc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            markersize = 10 if 'Quantum' in model_name else 8
            ax.plot(data_pcts, aucs, marker=marker, label=model_name, linewidth=linewidth, markersize=markersize)
        ax.set_xlabel('Training Data (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=14, fontweight='bold')
        ax.set_title('Test AUC vs Training Data Size', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        ax = axes[1, 0]
        model_names = list(results.keys())
        full_data_idx = -1
        accs = [results[m]['test_acc'][full_data_idx] for m in model_names]
        params = [results[m]['n_params'][full_data_idx] for m in model_names]
        efficiency = [a / (p/1000) for a, p in zip(accs, params)]

        colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in model_names]
        bars = ax.bar(range(len(model_names)), efficiency, color=colors, alpha=0.85, edgecolor='black', linewidth=2.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel('Accuracy / 1K Parameters', fontsize=14, fontweight='bold')
        ax.set_title('Parameter Efficiency', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, val) in enumerate(zip(bars, efficiency)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax = axes[1, 1]
        times = [results[m]['time'][full_data_idx] for m in model_names]
        colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in model_names]
        bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.85, edgecolor='black', linewidth=2.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('GPU Training Time Comparison', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, val) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results saved to: {save_path}")
        plt.show()


def main():
    print("\n" + "="*80)
    print(" GPU-ACCELERATED QUANTUM MACHINE LEARNING")
    print(" Optimized for NVIDIA RTX 4050")
    print(" BreaQ Hackathon 2025")
    print("="*80)

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available! Falling back to CPU")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"\n✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\nLoading RetinaMNIST dataset...")
    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Classes: {num_classes}")
    print(f"  - Training samples: {len(dh.labels['train'])}")
    print(f"  - Validation samples: {len(dh.labels['val'])}")
    print(f"  - Test samples: {len(dh.labels['test'])}")

    orchestrator = ExperimentOrchestrator(device=device)

    results = orchestrator.run_full_study(
        dh_base=dh,
        num_classes=num_classes,
        data_percentages=[0.25, 0.5, 1.0],
        epochs=100
    )

    orchestrator.plot_results(results)

    print("\n" + "="*80)
    print(" FINAL PERFORMANCE SUMMARY")
    print("="*80)

    for model_name in results.keys():
        print(f"\n{model_name}:")
        print(f"  Best Test Accuracy: {max(results[model_name]['test_acc']):.4f}")
        print(f"  Best Test AUC:      {max(results[model_name]['test_auc']):.4f}")
        print(f"  Best Val Accuracy:  {max(results[model_name]['best_val_acc']):.4f}")
        print(f"  Parameters:         {results[model_name]['n_params'][-1]:,}")
        print(f"  Avg Training Time:  {np.mean(results[model_name]['time']):.2f}s")

    print("\n" + "="*80)
    print(" Experiment Complete! Check quantum_gpu_results.png for visualizations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
