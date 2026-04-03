import os
import random
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import pandas as pd

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# --- Dual Attention Architecture ---
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
    def __init__(self, num_classes, pretrained=False):
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

all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
              'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
              'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

print("Loading model...")
model = AttentionDenseNet(num_classes=len(all_labels), pretrained=False).to(device)
checkpoint_path = os.path.join('nih', 'best_attention_model.pth')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

target_layer = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layer)

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset metadata...")
extract_dir = './nih_images'
dataset_source_dir = '/Users/lovisharora394/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
csv_path = os.path.join(dataset_source_dir, 'Data_Entry_2017.csv')

df = pd.read_csv(csv_path)
available_images = list(set(os.listdir(extract_dir)))
df = df[df["Image Index"].isin(available_images)].reset_index(drop=True)

# Generate 20 samples
samples = []
out_dir = "assets/demo"
os.makedirs(out_dir, exist_ok=True)

print("Pre-generating static Grad-CAM cache...")
for i in range(20):
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]
    img_name = row['Image Index']
    true_labels_str = row['Finding Labels']
    
    img_path = os.path.join(extract_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform_val(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    top_pred = [all_labels[j] for j in range(len(probs)) if probs[j] > 0.5]
    if not top_pred:
        top_pred = ["No Finding (Low Confidence)"]
        
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    img_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1).astype(np.float32)

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    viz_image = Image.fromarray(visualization)
    
    save_path = os.path.join(out_dir, f"patient_{i}.png")
    viz_image.save(save_path, format="PNG")
    
    samples.append({
        "image": save_path,
        "true_labels": true_labels_str.replace("|", ", "),
        "predicted_labels": ", ".join(top_pred)
    })
    print(f"Generated {i+1}/20")

with open('demo_data.json', 'w') as f:
    json.dump(samples, f)
print("Static pre-generation complete. Saved to demo_data.json")
