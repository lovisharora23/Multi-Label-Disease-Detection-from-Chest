import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc
import matplotlib.pyplot as plt

# ============================================================
# 1. SETUP & DOWNLOAD CHEXPERT
# ============================================================
print("Downloading/Locating CheXpert dataset via kagglehub...")
try:
    dataset_path = kagglehub.dataset_download('ashery/chexpert')
    print(f"Dataset sourced natively at: {dataset_path}")
except Exception as e:
    print(f"Error downloading via kagglehub ashery/chexpert: {e}")
    print("Falling back to another kaggle identifier...")
    dataset_path = kagglehub.dataset_download('stanfordmlgroup/chexpert')

valid_csv_candidates = glob.glob(os.path.join(dataset_path, "**", "valid.csv"), recursive=True)

if valid_csv_candidates:
    csv_path = valid_csv_candidates[0]
else:
    # If the validation CSV isn't found, fall back to train and take 500 samples
    train_candidates = glob.glob(os.path.join(dataset_path, "**", "train.csv"), recursive=True)
    if not train_candidates:
         raise FileNotFoundError("Could not find train.csv or valid.csv inside the downloaded Kaggle dataset.")
    csv_path = train_candidates[0]

# Extract the proper root directory containing all files
dataset_root = os.path.dirname(csv_path)

print(f"Using CSV: {csv_path}")
df = pd.read_csv(csv_path)

if 'train.csv' in csv_path:
    # Limit to 1000 for extremely quick validation if we couldn't find the explicit small valid set
    df = df.sample(1000, random_state=42).reset_index(drop=True)

# ============================================================
# 2. LABEL MAPPING (CheXpert -> NIH)
# ============================================================
CHEXPERT_TARGETS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Get exactly which neuron output indices map to the specific diseases in our trained model
NIH_TARGET_INDICES = [NIH_LABELS.index(lbl.replace('Pleural Effusion', 'Effusion')) for lbl in CHEXPERT_TARGETS]

print(f"Aligned Neural Target Indices: {NIH_TARGET_INDICES}")

# Clean CheXpert targets: NaN -> 0, -1 -> 1 (U-Ones approach standard for generalization tests)
for col in CHEXPERT_TARGETS:
    df[col] = df[col].fillna(0).replace(-1, 1).astype(float)

# ============================================================
# 3. DATASET CLASS
# ============================================================
class CheXpertDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        # Sometimes 'root_dir' acts as base, but 'Path' includes 'CheXpert-v1.0-small/'.
        # We will dynamically find what root connects to 'Path'
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['Path']
        
        # Test basic concat
        img_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(img_path):
            # Test bypassing wrapper folder
            img_path = os.path.join(self.root_dir, rel_path.split('/', 1)[-1])
            if not os.path.exists(img_path):
                # Test direct absolute Kaggle path
                img_path = os.path.join(dataset_path, rel_path)
                
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = self.df.iloc[idx][CHEXPERT_TARGETS].values.astype(np.float32)
        return image, torch.tensor(labels)

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = CheXpertDataset(df, dataset_root, transform_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ============================================================
# 4. INSTANTIATE NIH MODEL ARCHITECTURE
# ============================================================
class ChannelAttention(nn.Module):
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
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_map, max_map], dim=1)
        scale    = self.sigmoid(self.conv(combined))
        return x * scale

class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class AttentionDenseNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=False):
        super().__init__()
        densenet = models.densenet121(weights=None)
        self.features = densenet.features
        self.attention = DualAttentionBlock(in_channels=1024, reduction=16)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features  = self.features(x)
        attended  = self.attention(features)
        pooled    = self.gap(attended)
        flat      = pooled.view(pooled.size(0), -1)
        out       = self.classifier(self.dropout(flat))
        return out

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Initializing NIH Model on Acceleration Target {device}...")

model_path = "nih/best_attention_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError("Could not find the original NIH model at: " + model_path)

model = AttentionDenseNet(num_classes=14)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# ============================================================
# 5. ZERO-SHOT GENERALIZATION INFERENCE
# ============================================================
all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating NIH Features on CheXpert"):
        outputs = model(images.to(device))
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        # Filter predictions specifically to the 5 targeted overlap indices
        filtered_probs = probs[:, NIH_TARGET_INDICES]
        
        all_preds.append(filtered_probs)
        all_targets.append(targets.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# ============================================================
# 6. AUC CALCULATIONS & VISUALIZATIONS
# ============================================================
print("\n--- Zero-Shot Generalization Results ---")
auc_scores = {}
for i, label in enumerate(CHEXPERT_TARGETS):
    try:
        score = roc_auc_score(all_targets[:, i], all_preds[:, i])
        auc_scores[label] = score
        print(f"  {label:<20} {score:.4f}")
    except ValueError:
        print(f"  {label:<20} N/A (Missing positive samples in batch)")

mean_auc = np.mean(list(auc_scores.values()))
print(f"\nMean Generalization AUC: {mean_auc:.4f}")

# Plot Validation Generative Metrics
plt.figure(figsize=(9, 6))
labels_sorted = sorted(auc_scores, key=auc_scores.get, reverse=True)
vals_sorted   = [auc_scores[l] for l in labels_sorted]
bars = plt.bar(labels_sorted, vals_sorted, color='cornflowerblue')
plt.axhline(mean_auc, color='red', linestyle='--', label=f'CheXpert Mean AUC = {mean_auc:.3f}')
plt.xticks(rotation=20)
plt.ylabel("AUC Score")
plt.title("Zero-Shot Generalization: NIH AI on CheXpert Distribution")
plt.legend()
plt.tight_layout()
plt.savefig('nih/chexpert_generalization_auc.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
colors = plt.cm.tab10.colors
for i, label in enumerate(CHEXPERT_TARGETS):
    try:
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
        roc_val = sk_auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], label=f"{label} ({roc_val:.3f})")
    except ValueError:
        pass
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CheXpert Dataset - ROC Curves")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nih/chexpert_roc_curves.png', dpi=150)
plt.close()

print("\nEvaluation Successful. Graphics saved.")
