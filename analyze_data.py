import numpy as np
from medmnist import RetinaMNIST
import matplotlib.pyplot as plt
from PIL import Image

print("="*80)
print(" DATA QUALITY ANALYSIS - RetinaMNIST")
print("="*80)

train_ds = RetinaMNIST(split='train', download=True)
val_ds = RetinaMNIST(split='val', download=True)
test_ds = RetinaMNIST(split='test', download=True)

print(f"\nDataset Splits:")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val: {len(val_ds)} samples")
print(f"  Test: {len(test_ds)} samples")

train_labels = train_ds.labels.squeeze()
val_labels = val_ds.labels.squeeze()
test_labels = test_ds.labels.squeeze()

print(f"\nClass Distribution (Train):")
unique, counts = np.unique(train_labels, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:4d} samples ({count/len(train_labels)*100:5.1f}%)")

print(f"\nClass Distribution (Val):")
unique, counts = np.unique(val_labels, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:4d} samples ({count/len(val_labels)*100:5.1f}%)")

print(f"\nClass Distribution (Test):")
unique, counts = np.unique(test_labels, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:4d} samples ({count/len(test_labels)*100:5.1f}%)")

print(f"\nImage Statistics:")
train_imgs = train_ds.imgs
print(f"  Shape: {train_imgs.shape}")
print(f"  Dtype: {train_imgs.dtype}")
print(f"  Min: {train_imgs.min()}")
print(f"  Max: {train_imgs.max()}")
print(f"  Mean: {train_imgs.mean():.2f}")
print(f"  Std: {train_imgs.std():.2f}")

per_class_mean = []
per_class_std = []
for cls in range(5):
    cls_imgs = train_imgs[train_labels == cls]
    per_class_mean.append(cls_imgs.mean())
    per_class_std.append(cls_imgs.std())

print(f"\nPer-Class Statistics:")
for cls in range(5):
    print(f"  Class {cls}: Mean={per_class_mean[cls]:.2f}, Std={per_class_std[cls]:.2f}")

fig, axes = plt.subplots(5, 8, figsize=(16, 10))
fig.suptitle('Sample Images per Class (RetinaMNIST)', fontsize=16, fontweight='bold')

for cls in range(5):
    cls_indices = np.where(train_labels == cls)[0]
    sample_indices = np.random.choice(cls_indices, min(8, len(cls_indices)), replace=False)

    for i, idx in enumerate(sample_indices):
        img = train_imgs[idx]
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        axes[cls, i].imshow(img)
        axes[cls, i].axis('off')
        if i == 0:
            axes[cls, i].set_ylabel(f'Class {cls}\n({len(cls_indices)} samples)',
                                   fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: data_quality_analysis.png")

print("\n" + "="*80)
print(" POTENTIAL ISSUES IDENTIFIED")
print("="*80)

imbalance_ratio = counts.max() / counts.min()
print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 5:
    print("  ⚠ SEVERE IMBALANCE - Need strong class weighting & augmentation")
elif imbalance_ratio > 3:
    print("  ⚠ MODERATE IMBALANCE - Need class weighting")

if len(train_labels) < 1000:
    print(f"\n⚠ VERY SMALL DATASET ({len(train_labels)} samples)")
    print("  - Use heavy augmentation")
    print("  - Avoid deep architectures (overfitting risk)")
    print("  - Consider ensemble methods")

if per_class_std[0] / per_class_mean[0] > 0.5:
    print(f"\n⚠ HIGH VARIANCE in pixel values")
    print("  - Normalization is critical")

print("\n" + "="*80 + "\n")
