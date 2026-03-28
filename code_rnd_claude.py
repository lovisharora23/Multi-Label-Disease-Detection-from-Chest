# ============================================================
# MULTI-DISEASE CHEST X-RAY DETECTION
# Proposed Model: Attention-Guided CNN
# ============================================================
# CELL 1 — SETUP
# ============================================================

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install torch torchvision scikit-learn pandas tqdm grad-cam -q

import os
import tarfile
import glob
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)


# ============================================================
# CELL 2 — PATHS
# ============================================================

# Define base directory for relative paths if needed, otherwise assume current working directory
base_dir = os.getcwd()

nih_root = os.path.abspath(os.path.join(base_dir, 'nih'))
nih_images = os.path.join(nih_root, 'images')
dataset_source_dir = '/Users/lovisharora394/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
csv_path = os.path.join(dataset_source_dir, 'Data_Entry_2017.csv')
extract_dir = './nih_images'          # local fast storage on Colab
os.makedirs(extract_dir, exist_ok=True)


# ============================================================
# CELL 3 — EXTRACT ALL IMAGES (tar.gz AND plain folders)
# ============================================================
# This cell handles:
#   1. Any .tar.gz files under nih_root  (e.g. images_001.tar.gz)
#   2. Any plain image sub-folders       (e.g. nih/extra_images/)
# All images end up flat inside extract_dir for fast local reads.

print("Step 1: Extracting .tar.gz archives...")
tar_files = glob.glob(os.path.join(nih_root, '**', '*.tar.gz'), recursive=True)
if tar_files:
    for tf in tar_files:
        print(f"  Extracting {os.path.basename(tf)} ...")
        with tarfile.open(tf, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    member.name = os.path.basename(member.name)  # flatten path
                    tar.extract(member, extract_dir)
    print(f"  Done. Archives extracted to {extract_dir}")
else:
    print("  No .tar.gz files found — skipping.")

print("\nStep 2: Copying images from plain sub-folders...")
image_extensions = ('.png', '.jpg', '.jpeg')
copied = 0
for root_dir, dirs, files in os.walk(nih_root):
    # skip if this folder is the csv-only root
    for fname in files:
        if fname.lower().endswith(image_extensions):
            src  = os.path.abspath(os.path.join(root_dir, fname))
            dest = os.path.abspath(os.path.join(extract_dir, fname))
            if not os.path.exists(dest):        # don't overwrite already-extracted
                os.symlink(src, dest)           # symlink saves space & time
                copied += 1

print(f"  Linked {copied} images from plain folders → {extract_dir}")

print("\nStep 2: Copying images from Kaggle cache...")
linked_images_count = 0
# Recursively find all PNG files
for root_dir, dirs, files in os.walk(dataset_source_dir):
    # Skip the destination folder to prevent infinite symlink loops
    if os.path.abspath(root_dir) == os.path.abspath(extract_dir):
        continue
    for fname in files:
        if fname.lower().endswith(image_extensions):
            src = os.path.abspath(os.path.join(root_dir, fname))
            dest = os.path.abspath(os.path.join(extract_dir, fname))
            if not os.path.exists(dest):  # don't overwrite already-extracted
                os.symlink(src, dest)  # symlink saves space & time
                linked_images_count += 1
print(f"  Linked {linked_images_count} images from Kaggle cache → {extract_dir}")

print(f"\nTotal images available: {len(os.listdir(extract_dir))}")


# ============================================================
# CELL 4 — LOAD CSV & FILTER
# ============================================================

data = pd.read_csv(csv_path)
available_images = set(os.listdir(extract_dir))
data = data[data["Image Index"].isin(available_images)].reset_index(drop=True)
print("Matched dataset size:", len(data))

# ---- Optional: cap dataset for quicker runs ---------------
# data = data.sample(30000, random_state=42).reset_index(drop=True)
# -----------------------------------------------------------


# ============================================================
# CELL 5 — IMAGE VERIFICATION
# ============================================================

verified_indices = []
bad = 0
for idx in tqdm(data.index, desc="Verifying images"):
    img_path = os.path.join(extract_dir, data.loc[idx, 'Image Index'])
    try:
        with Image.open(img_path) as img:
            img.verify()
        verified_indices.append(idx)
    except Exception:
        bad += 1

data = data.loc[verified_indices].reset_index(drop=True)
print(f"Verified dataset: {len(data)}  |  Removed: {bad} corrupt files")


# ============================================================
# CELL 6 — MULTI-LABEL ENCODING
# ============================================================

all_labels = sorted(set(
    lbl
    for sublist in data['Finding Labels'].str.split('|')
    for lbl in sublist
    if lbl != 'No Finding'
))

for label in all_labels:
    data[label] = data['Finding Labels'].apply(lambda x: 1 if label in x else 0)

print(f"Disease classes ({len(all_labels)}): {all_labels}")


# ============================================================
# CELL 7 — COMPUTE CLASS WEIGHTS (for Weighted BCE)
# ============================================================

label_counts  = data[all_labels].sum(axis=0).values          # positives per class
total_samples = len(data)
# weight = total / (2 * positives)  — upweights rare diseases
class_weights = total_samples / (2.0 * (label_counts + 1e-6))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights (higher = rarer disease):")
for name, w in zip(all_labels, class_weights.cpu().numpy()):
    print(f"  {name:<25} {w:.2f}")


# ============================================================
# CELL 8 — TRAIN / VAL SPLIT
# ============================================================

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")


# ============================================================
# CELL 9 — DATASET CLASS
# ============================================================

class NIH_Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels    = all_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        label    = self.df.iloc[idx][self.labels].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


# ============================================================
# CELL 10 — TRANSFORMS  (stronger augmentation for training)
# ============================================================

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = NIH_Dataset(train_df, extract_dir, transform_train)
val_dataset   = NIH_Dataset(val_df,   extract_dir, transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=True)


# ============================================================
# CELL 11 — ATTENTION MODULE  ← YOUR NOVEL CONTRIBUTION
# ============================================================
# Spatial Attention: learns WHERE in the X-ray to focus.
# Channel Attention: learns WHICH feature maps matter.
# Together = "Dual Attention" — this is your proposed addition.

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale   = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention: highlights important lung regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_map, max_map], dim=1)
        scale    = self.sigmoid(self.conv(combined))
        return x * scale


class DualAttentionBlock(nn.Module):
    """Apply channel attention then spatial attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================
# CELL 12 — PROPOSED MODEL (Attention-Guided CNN)
# ============================================================

class AttentionDenseNet(nn.Module):
    """
    Proposed model:
      DenseNet121 feature extractor
      + Dual Attention Block  (novel contribution)
      + Global Average Pooling
      + Fully Connected classifier
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Use updated weights API (removes deprecation warning)
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Feature extractor — remove original classifier
        self.features = densenet.features          # output: (B, 1024, 7, 7)

        # Your contribution: dual attention on top of DenseNet features
        self.attention = DualAttentionBlock(in_channels=1024, reduction=16)

        # Pooling + classifier
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features  = self.features(x)          # (B, 1024, 7, 7)
        attended  = self.attention(features)  # (B, 1024, 7, 7) — attention applied
        pooled    = self.gap(attended)        # (B, 1024, 1, 1)
        flat      = pooled.view(pooled.size(0), -1)   # (B, 1024)
        out       = self.classifier(self.dropout(flat))
        return out


model = AttentionDenseNet(num_classes=len(all_labels), pretrained=True).to(device)
print("Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")


# ============================================================
# CELL 13 — LOSS, OPTIMIZER, SCHEDULER
# ============================================================

# Weighted BCE: each class gets its own weight to tackle imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Reduce LR if val loss plateaus for 2 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)


# ============================================================
# CELL 14 — TRAINING LOOP (15 epochs, with checkpointing)
# ============================================================

EPOCHS         = 15
best_val_loss  = float('inf')
checkpoint_path = os.path.join(nih_root, 'best_attention_model.pth')
train_losses   = []
val_losses     = []

for epoch in range(EPOCHS):

    # ---------- TRAIN ----------
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---------- VALIDATE ----------
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "):
            images = images.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels.to(device))
            val_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Compute mean AUC
    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    aucs = []
    for i in range(len(all_labels)):
        try:
            aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
        except Exception:
            pass
    mean_auc = np.mean(aucs) if aucs else 0.0

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]  "
          f"Train Loss: {avg_train_loss:.4f}  |  "
          f"Val Loss: {avg_val_loss:.4f}  |  "
          f"Mean AUC: {mean_auc:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch':       epoch + 1,
            'model_state': model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'val_loss':    best_val_loss,
            'mean_auc':    mean_auc,
        }, checkpoint_path)
        print(f"  ✔ Saved best model → {checkpoint_path}")

    # LR scheduler step
    scheduler.step(avg_val_loss)

print("\nTraining complete.")


# ============================================================
# CELL 15 — LOAD BEST MODEL & FINAL EVALUATION
# ============================================================

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val loss: {checkpoint['val_loss']:.4f})")

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Final evaluation"):
        images  = images.to(device)
        outputs = model(images)
        preds   = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())

all_preds   = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

print("\nValidation AUC per class:")
auc_scores = {}
for i, label in enumerate(all_labels):
    try:
        score = roc_auc_score(all_targets[:, i], all_preds[:, i])
        auc_scores[label] = score
        print(f"  {label:<25} {score:.4f}")
    except Exception:
        print(f"  {label:<25} N/A")

mean_auc = np.mean(list(auc_scores.values()))
print(f"\nMean AUC: {mean_auc:.4f}")


# ============================================================
# CELL 16 — TRAINING CURVE PLOT
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss',   marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss — Attention CNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(nih_root, 'training_curve.png'), dpi=150)
plt.close()


# ============================================================
# CELL 17 — ROC CURVES (ALL 14 CLASSES)
# ============================================================

from sklearn.metrics import roc_curve, auc as sk_auc

plt.figure(figsize=(12, 10))
colors = plt.cm.tab20.colors

for i, label in enumerate(all_labels):
    try:
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
        roc_val     = sk_auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 label=f"{label} ({roc_val:.2f})", lw=1.5)
    except Exception:
        pass

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All 14 Disease Classes (Proposed Attention CNN)")
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(nih_root, 'roc_curves.png'), dpi=150)
plt.close()


# ============================================================
# CELL 18 — AUC BAR CHART
# ============================================================

labels_sorted = sorted(auc_scores, key=auc_scores.get, reverse=True)
vals_sorted   = [auc_scores[l] for l in labels_sorted]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels_sorted, vals_sorted,
               color=['green' if v >= 0.8 else 'orange' if v >= 0.7 else 'red'
                      for v in vals_sorted])
plt.axhline(mean_auc, color='blue', linestyle='--', label=f'Mean AUC = {mean_auc:.3f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel("AUC")
plt.title("Per-Class AUC — Proposed Attention CNN")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(nih_root, 'auc_bar.png'), dpi=150)
plt.close()


# ============================================================
# CELL 19 — SAMPLE PREDICTIONS
# ============================================================

import random

model.eval()
sample_indices = random.sample(range(len(val_dataset)), 6)
plt.figure(figsize=(15, 8))

for i, idx in enumerate(sample_indices):
    image, label = val_dataset[idx]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs  = torch.sigmoid(output).cpu().numpy()[0]
    top_pred = [all_labels[j] for j in range(len(probs)) if probs[j] > 0.5] or ["No Finding"]

    img_np = image.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img_np, cmap='gray')
    plt.title(f"Predicted:\n{', '.join(top_pred)}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(nih_root, 'sample_predictions.png'), dpi=150)
plt.close()


# ============================================================
# CELL 20 — GRAD-CAM VISUALIZATION (BUG FIXED)
# ============================================================

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target: last conv layer inside DenseNet features
target_layer = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layer)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    image, label = val_dataset[i]
    input_tensor = image.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # FIX: correct std/mean — your old code had [0.229,0.456,0.406] which was wrong
    img_np = image.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1).astype(np.float32)

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    true_labels = [all_labels[j] for j in range(len(label)) if label[j] == 1] or ["No Finding"]
    axes[i].imshow(visualization)
    axes[i].set_title(f"True: {', '.join(true_labels)}", fontsize=8)
    axes[i].axis("off")

plt.suptitle("Grad-CAM Heatmaps — Attention CNN", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(nih_root, 'gradcam.png'), dpi=150)
plt.close()


# ============================================================
# CELL 21 — CONTRIBUTION SUMMARY TABLE (for your report)
# ============================================================

print("=" * 65)
print(f"{'Component':<30} {'Baseline':<18} {'Proposed (Ours)'}")
print("=" * 65)
rows = [
    ("Backbone",         "DenseNet121",   "DenseNet121"),
    ("Attention module", "None",          "Dual Attention Block ✔"),
    ("Loss function",    "Focal Loss",    "Weighted BCE ✔"),
    ("Class imbalance",  "Focal Loss",    "Per-class weights ✔"),
    ("Explainability",   "Grad-CAM",      "Grad-CAM"),
    ("Epochs",           "3",             "15"),
    ("Batch size",       "8",             "32"),
    ("LR scheduler",     "None",          "ReduceLROnPlateau ✔"),
    ("Checkpointing",    "None",          "Best model saved ✔"),
]
for name, base, prop in rows:
    print(f"  {name:<28} {base:<18} {prop}")
print("=" * 65)
print(f"\nFinal Mean AUC (Proposed): {mean_auc:.4f}")
