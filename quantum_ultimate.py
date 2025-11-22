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


def build_dataloaders(dh, batch_size=32, augment=True):
    loaders = {}

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]) if augment else transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
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


class QuantumConvFilter(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[n_qubits-1, 0])

                for i in range(n_qubits):
                    qml.RY(inputs[i] * weights[layer, i, 2], wires=i)

            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        if x.shape[-1] < self.n_qubits:
            padding = torch.zeros(*x.shape[:-1], self.n_qubits - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)
        else:
            x = x[..., :self.n_qubits]

        output = self.qlayer(x)

        if output.dim() > 2:
            output = output.reshape(output.shape[0], -1)

        if output.shape[-1] < self.n_qubits:
            output = F.pad(output, (0, self.n_qubits - output.shape[-1]))
        elif output.shape[-1] > self.n_qubits:
            output = output[..., :self.n_qubits]

        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UltimateClassicalCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class QuantumEnhancedResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.pre_quantum = nn.Sequential(
            nn.Linear(128, n_qubits * 2),
            nn.ReLU(),
            nn.Linear(n_qubits * 2, n_qubits),
            nn.Tanh()
        )

        self.quantum = QuantumConvFilter(n_qubits=n_qubits, n_layers=n_layers)

        self.layer3 = self._make_layer(128 + n_qubits, 256, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x_conv = self.layer2(x)

        x_pool = self.avgpool(x_conv)
        x_flat = torch.flatten(x_pool, 1)

        x_pre_q = self.pre_quantum(x_flat)
        x_quantum = self.quantum(x_pre_q)

        H, W = x_conv.shape[2], x_conv.shape[3]
        x_q_spatial = x_quantum.view(B, self.n_qubits, 1, 1).expand(B, self.n_qubits, H, W)

        x_fused = torch.cat([x_conv, x_q_spatial], dim=1)

        x_out = self.layer3(x_fused)
        x_out = self.classifier(x_out)

        return x_out


class QuantumMultiScale(nn.Module):
    def __init__(self, num_classes, in_channels=3, n_qubits=6):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.scale1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 128)
        )

        self.quantum1 = QuantumConvFilter(n_qubits=n_qubits, n_layers=2)
        self.quantum2 = QuantumConvFilter(n_qubits=n_qubits, n_layers=3)

        self.fusion = nn.Sequential(
            nn.Conv2d(128 + n_qubits * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.conv1(x)
        x1 = self.scale1(x)
        x2 = self.scale2(x1)

        H, W = x2.shape[2], x2.shape[3]

        q_in1 = F.adaptive_avg_pool2d(x1, 1).flatten(1)
        q_in1 = F.pad(q_in1, (0, max(0, self.n_qubits - q_in1.shape[1])))[:, :self.n_qubits]
        q_out1 = self.quantum1(q_in1)
        q_spatial1 = q_out1.view(B, self.n_qubits, 1, 1).expand(B, self.n_qubits, H, W)

        q_in2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        q_in2 = F.pad(q_in2, (0, max(0, self.n_qubits - q_in2.shape[1])))[:, :self.n_qubits]
        q_out2 = self.quantum2(q_in2)
        q_spatial2 = q_out2.view(B, self.n_qubits, 1, 1).expand(B, self.n_qubits, H, W)

        fused = torch.cat([x2, q_spatial1, q_spatial2], dim=1)

        features = self.fusion(fused)
        output = self.classifier(features)

        return output


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler and device == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
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

    def run_single_experiment(self, model_name, model, loaders, epochs=50, lr=1e-3):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 5,
            epochs=epochs,
            steps_per_epoch=len(loaders['train']),
            pct_start=0.3
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        start_time = time.time()
        best_val_acc = 0

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

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | Train: {tr_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        training_time = time.time() - start_time

        test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, self.device)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"\nResults:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Training Time: {training_time:.2f}s")

        return {
            'history': history,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'n_params': n_params,
            'training_time': training_time
        }

    def run_data_regime_study(self, dh_base, num_classes, data_percentages=[0.25, 0.5, 1.0], epochs=50):
        print("\n" + "="*80)
        print(" ULTIMATE QUANTUM ML SHOWDOWN")
        print("="*80)

        model_builders = {
            'ResNet Baseline': lambda: UltimateClassicalCNN(num_classes, in_channels=3),
            'Quantum-Enhanced ResNet': lambda: QuantumEnhancedResNet(num_classes, in_channels=3, n_qubits=8, n_layers=4),
            'Quantum Multi-Scale': lambda: QuantumMultiScale(num_classes, in_channels=3, n_qubits=6),
        }

        results = defaultdict(lambda: defaultdict(list))

        for pct in data_percentages:
            print(f"\n{'='*60}")
            print(f"Training with {int(pct*100)}% of data")
            print(f"{'='*60}")

            dh = DataHandler(download=False, as_rgb=True)
            dh.images = {k: v.copy() for k, v in dh_base.images.items()}
            dh.labels = {k: v.copy() for k, v in dh_base.labels.items()}

            if pct < 1.0:
                dh.select_train_percentage(pct, seed=42)

            loaders = build_dataloaders(dh, batch_size=16, augment=True)

            for model_name, builder in model_builders.items():
                model = builder().to(self.device)
                res = self.run_single_experiment(model_name, model, loaders, epochs=epochs, lr=1e-3)

                results[model_name]['data_pct'].append(pct)
                results[model_name]['test_acc'].append(res['test_acc'])
                results[model_name]['test_auc'].append(res['test_auc'])
                results[model_name]['n_params'].append(res['n_params'])
                results[model_name]['time'].append(res['training_time'])

        return results

    def plot_results(self, results, save_path='quantum_ultimate_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            accs = results[model_name]['test_acc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            ax.plot(data_pcts, accs, marker=marker, label=model_name, linewidth=linewidth, markersize=8)
        ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Training Data', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            aucs = results[model_name]['test_auc']
            marker = 'D' if 'Quantum' in model_name else 'o'
            linewidth = 3 if 'Quantum' in model_name else 2
            ax.plot(data_pcts, aucs, marker=marker, label=model_name, linewidth=linewidth, markersize=8)
        ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
        ax.set_title('AUC vs Training Data', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        model_names = list(results.keys())
        full_data_idx = -1
        accs = [results[m]['test_acc'][full_data_idx] for m in model_names]
        params = [results[m]['n_params'][full_data_idx] for m in model_names]
        efficiency = [a / (p/1000) for a, p in zip(accs, params)]

        colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in model_names]
        bars = ax.bar(range(len(model_names)), efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Accuracy / 1K Params', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 1]
        times = [results[m]['time'][full_data_idx] for m in model_names]
        colors = ['#FF6B6B' if 'Quantum' in m else '#4ECDC4' for m in model_names]
        bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")
        plt.show()


def main():
    print("\n" + "="*80)
    print(" ULTIMATE QUANTUM ML - BreaQ 2025")
    print(" Maximum Performance with Creative Quantum Integration")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"Classes: {num_classes}")
    print(f"Training samples: {len(dh.labels['train'])}")

    orchestrator = ExperimentOrchestrator(device=device)

    results = orchestrator.run_data_regime_study(
        dh_base=dh,
        num_classes=num_classes,
        data_percentages=[0.25, 0.5, 1.0],
        epochs=50
    )

    orchestrator.plot_results(results)

    print("\n" + "="*80)
    print(" FINAL RESULTS")
    print("="*80)

    for model_name in results.keys():
        print(f"\n{model_name}:")
        print(f"  Best Accuracy: {max(results[model_name]['test_acc']):.4f}")
        print(f"  Best AUC: {max(results[model_name]['test_auc']):.4f}")
        print(f"  Parameters: {results[model_name]['n_params'][-1]:,}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
