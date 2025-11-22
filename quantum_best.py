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
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        normalize
    ]) if augment else transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
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


class EfficientCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        mobilenet = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

        for param in mobilenet.parameters():
            param.requires_grad = False

        for param in mobilenet.features[-3:].parameters():
            param.requires_grad = True

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class QuantumHybrid(nn.Module):
    def __init__(self, num_classes, n_qubits=4):
        super().__init__()

        mobilenet = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

        for param in mobilenet.parameters():
            param.requires_grad = False

        for param in mobilenet.features[-3:].parameters():
            param.requires_grad = True

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=2)

        self.post_quantum = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n_qubits, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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


def train_model(model_name, model, loaders, device, class_weights, epochs=60, lr=0.001):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)

    best_val_acc = 0
    patience = 12
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, loaders['val'], criterion, device)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    test_loss, test_acc, test_auc = evaluate(model, loaders['test'], criterion, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print(f"FINAL - {model_name}")
    print(f"Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"Trainable Params: {n_params:,} | Time: {training_time:.1f}s")
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
    print(" OPTIMIZED QUANTUM ML - BreaQ 2025")
    print(" MobileNetV2 + Quantum Hybrid")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dh = DataHandler(download=True, as_rgb=True)
    num_classes = len(np.unique(dh.labels['train']))

    print(f"\nDataset: RetinaMNIST")
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(dh.labels['train'])}")

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

        print(f"\nTraining samples: {len(dh_copy.labels['train'])}")

        for model_name, model_fn in [
            ('MobileNetV2 Transfer', lambda: EfficientCNN(num_classes)),
            ('Quantum Hybrid', lambda: QuantumHybrid(num_classes, n_qubits=4))
        ]:
            model = model_fn().to(device)
            res = train_model(model_name, model, loaders, device, class_weights, epochs=60, lr=0.001)

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
        print(f"  Trainable Params: {res['n_params']:,}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
