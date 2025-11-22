import numpy as np
from PIL import Image
from medmnist import RetinaMNIST
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import time
from collections import defaultdict


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

    def select_classes(self, classes):
        for split in ["train", "val", "test"]:
            cls_set = set(classes)
            mask = np.array([lbl in cls_set for lbl in self.labels[split]])
            self.images[split] = self.images[split][mask]
            self.labels[split] = self.labels[split][mask]

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

    def normalize(self, method="minmax"):
        for split in ["train", "val", "test"]:
            imgs = self.images[split].astype(np.float32)

            if method == "minmax":
                min_val = imgs.min(axis=(0,1,2), keepdims=True)
                max_val = imgs.max(axis=(0,1,2), keepdims=True)
                imgs = (imgs - min_val) / (max_val - min_val + 1e-7)
            elif method == "standard":
                mean = imgs.mean(axis=(0,1,2), keepdims=True)
                std = imgs.std(axis=(0,1,2), keepdims=True) + 1e-7
                imgs = (imgs - mean) / std

            self.images[split] = imgs

    def to_grayscale(self):
        for split in ["train", "val", "test"]:
            gray_list = []
            for img in self.images[split]:
                pil = Image.fromarray(img).convert("L")
                gray_list.append(np.array(pil))
            self.images[split] = np.stack(gray_list, axis=0)[..., None]

    def resize(self, size):
        for split in ["train", "val", "test"]:
            out = []
            for img in self.images[split]:
                if img.shape[-1] == 1:
                    pil = Image.fromarray(img.squeeze(), mode="L")
                else:
                    pil = Image.fromarray(img)
                pil = pil.resize(size, Image.BILINEAR)
                arr = np.array(pil)
                if arr.ndim == 2:
                    arr = arr[..., None]
                out.append(arr)
            self.images[split] = np.stack(out, axis=0)


def build_dataloaders(dh, batch_size=64, device='cpu'):
    loaders = {}
    for split in ["train", "val", "test"]:
        imgs = torch.from_numpy(dh.images[split]).float()
        imgs = imgs.permute(0,3,1,2)
        if imgs.max() > 1.0:
            imgs /= 255.0
        labels = torch.from_numpy(dh.labels[split]).long()
        dataset = TensorDataset(imgs, labels)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if device=='cuda' else False
        )
    return loaders


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        if x.shape[1] > self.n_qubits:
            x = x[:, :self.n_qubits]
        elif x.shape[1] < self.n_qubits:
            padding = torch.zeros(x.shape[0], self.n_qubits - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)

        output = self.qlayer(x)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        return output


class StrongClassicalCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class QuantumHybridStrong(nn.Module):
    def __init__(self, num_classes, in_channels=1, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv_features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.pre_quantum = nn.Sequential(
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_features(x)
        x = self.pre_quantum(x)
        x_q = self.quantum_layer(x)
        if x_q.shape[1] != self.n_qubits:
            x_q = x_q.view(x.shape[0], self.n_qubits)
        x = self.post_quantum(x_q)
        return x


class QuantumEarlyFusion(nn.Module):
    def __init__(self, num_classes, in_channels=1, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16 + n_qubits, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_conv = self.conv1(x)

        B, C, H, W = x_conv.shape
        x_pool = self.spatial_pool(x_conv)

        x_flat = x_pool.view(B, C, -1).mean(dim=2)
        x_quantum = self.quantum_layer(x_flat)

        if x_quantum.dim() > 2:
            x_quantum = x_quantum.view(B, -1)

        if x_quantum.shape[1] != self.n_qubits:
            x_quantum = x_quantum[:, :self.n_qubits] if x_quantum.shape[1] > self.n_qubits else torch.nn.functional.pad(x_quantum, (0, self.n_qubits - x_quantum.shape[1]))

        x_quantum_spatial = x_quantum.unsqueeze(-1).unsqueeze(-1).expand(B, self.n_qubits, H, W)

        x_fused = torch.cat([x_conv, x_quantum_spatial], dim=1)

        x = self.conv2(x_fused)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
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

    def run_single_experiment(self, model_name, model, loaders, epochs=30, lr=1e-3):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        start_time = time.time()
        best_val_acc = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_auc = train_epoch(model, loaders['train'], optimizer, criterion, self.device)
            val_loss, val_acc, val_auc = evaluate(model, loaders['val'], criterion, self.device)

            scheduler.step(val_acc)

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['train_auc'].append(tr_auc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | Train acc: {tr_acc:.4f} | Val acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

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

    def run_data_regime_study(self, dh_base, num_classes, in_channels,
                             data_percentages=[0.1, 0.25, 0.5, 1.0],
                             epochs=30):
        print("\n" + "="*80)
        print("COMPREHENSIVE QUANTUM ML STUDY")
        print("Strong models across different data regimes")
        print("="*80)

        model_builders = {
            'Classical CNN': lambda: StrongClassicalCNN(num_classes, in_channels),
            'Quantum Hybrid (Mid)': lambda: QuantumHybridStrong(num_classes, in_channels, n_qubits=8, n_layers=3),
            'Quantum Early Fusion': lambda: QuantumEarlyFusion(num_classes, in_channels, n_qubits=4, n_layers=2),
        }

        results = defaultdict(lambda: defaultdict(list))

        for pct in data_percentages:
            print(f"\n{'='*60}")
            print(f"Training with {int(pct*100)}% of data")
            print(f"{'='*60}")

            dh = DataHandler(download=False)
            dh.images = {k: v.copy() for k, v in dh_base.images.items()}
            dh.labels = {k: v.copy() for k, v in dh_base.labels.items()}

            if pct < 1.0:
                dh.select_train_percentage(pct, seed=42)

            loaders = build_dataloaders(dh, batch_size=32, device=self.device)

            for model_name, builder in model_builders.items():
                model = builder().to(self.device)
                res = self.run_single_experiment(model_name, model, loaders, epochs=epochs, lr=1e-3)

                results[model_name]['data_pct'].append(pct)
                results[model_name]['test_acc'].append(res['test_acc'])
                results[model_name]['test_auc'].append(res['test_auc'])
                results[model_name]['n_params'].append(res['n_params'])
                results[model_name]['time'].append(res['training_time'])

        return results

    def plot_data_regime_results(self, results, save_path='quantum_ml_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            accs = results[model_name]['test_acc']
            marker = 's' if 'Quantum' in model_name else 'o'
            linewidth = 2.5 if 'Quantum' in model_name else 1.5
            ax.plot(data_pcts, accs, marker=marker, label=model_name, linewidth=linewidth)
        ax.set_xlabel('Training Data (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Training Data Size', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for model_name in results.keys():
            data_pcts = [p*100 for p in results[model_name]['data_pct']]
            aucs = results[model_name]['test_auc']
            marker = 's' if 'Quantum' in model_name else 'o'
            linewidth = 2.5 if 'Quantum' in model_name else 1.5
            ax.plot(data_pcts, aucs, marker=marker, label=model_name, linewidth=linewidth)
        ax.set_xlabel('Training Data (%)', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('AUC vs Training Data Size', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        model_names = list(results.keys())
        full_data_idx = -1
        accs = [results[m]['test_acc'][full_data_idx] for m in model_names]
        params = [results[m]['n_params'][full_data_idx] for m in model_names]
        efficiency = [a / (p/1000) for a, p in zip(accs, params)]

        colors = ['red' if 'Quantum' in m else 'blue' for m in model_names]
        bars = ax.bar(range(len(model_names)), efficiency, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylabel('Accuracy / 1K Parameters', fontsize=12)
        ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 1]
        times = [results[m]['time'][full_data_idx] for m in model_names]
        colors = ['red' if 'Quantum' in m else 'blue' for m in model_names]
        bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")
        plt.show()

        return fig


def main():
    print("\n" + "="*80)
    print(" STRONG QUANTUM ML - BreaQ Hackathon 2025")
    print(" Proper architectures with meaningful quantum integration")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    print("\nLoading RetinaMNIST dataset...")
    dh = DataHandler(download=True, as_rgb=False)
    dh.to_grayscale()
    dh.normalize('minmax')

    num_classes = len(np.unique(dh.labels['train']))
    in_channels = dh.images['train'].shape[-1]

    print(f"Classes: {num_classes}")
    print(f"Input channels: {in_channels}")
    print(f"Training samples: {len(dh.labels['train'])}")

    orchestrator = ExperimentOrchestrator(device=device)

    results = orchestrator.run_data_regime_study(
        dh_base=dh,
        num_classes=num_classes,
        in_channels=in_channels,
        data_percentages=[0.1, 0.25, 0.5, 1.0],
        epochs=30
    )

    orchestrator.plot_data_regime_results(results, save_path='quantum_ml_strong_results.png')

    print("\n" + "="*80)
    print(" EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print("-" * 60)

    for model_name in results.keys():
        print(f"\n{model_name}:")
        print(f"  Final Test Accuracy: {results[model_name]['test_acc'][-1]:.4f}")
        print(f"  Final Test AUC: {results[model_name]['test_auc'][-1]:.4f}")
        print(f"  Parameters: {results[model_name]['n_params'][-1]:,}")

        acc_low = results[model_name]['test_acc'][0]
        acc_high = results[model_name]['test_acc'][-1]
        data_efficiency = (acc_low / acc_high) * 100
        print(f"  Data Efficiency (10% vs 100%): {data_efficiency:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
