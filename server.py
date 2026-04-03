import os
import random
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='/')
CORS(app)

print("Loading dependencies...")
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

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
        # Ensure we have weights_only=False inside DenseNet creation not here
        weights = None
        densenet = models.densenet121(weights=weights)
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

print("Initializing Neural Network...")
model = AttentionDenseNet(num_classes=len(all_labels), pretrained=False).to(device)

checkpoint_path = os.path.join('nih', 'best_attention_model.pth')
print(f"Loading weights from {checkpoint_path}...")
try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model loaded successfully (Val Loss: {checkpoint['val_loss']:.4f})")
except Exception as e:
    print(f"Failed to load model weights: {e}")

model.eval()

# Initialize Grad-CAM
target_layer = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layer)

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset catalog...")
extract_dir = './nih_images'
dataset_source_dir = '/Users/lovisharora394/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
csv_path = os.path.join(dataset_source_dir, 'Data_Entry_2017.csv')

available_images = []
data = None
try:
    data = pd.read_csv(csv_path)
    available_images = list(set(os.listdir(extract_dir)))
    data = data[data["Image Index"].isin(available_images)].reset_index(drop=True)
    print(f"Loaded {len(data)} image metadata records.")
except Exception as e:
    print(f"Failed to load metadata CSV: {e}")


@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/random_gradcam', methods=['POST'])
def generate_random_gradcam():
    try:
        if data is None or len(data) == 0:
            return jsonify({"error": "Dataset not loaded, cannot pick a random image."}), 500
            
        idx = random.randint(0, len(data) - 1)
        row = data.iloc[idx]
        img_name = row['Image Index']
        true_labels_str = row['Finding Labels']
        
        img_path = os.path.join(extract_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform_val(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
        
        # Determine predicted classes > 0.5
        top_pred = [all_labels[j] for j in range(len(probs)) if probs[j] > 0.5]
        if not top_pred:
            top_pred = ["No Finding (Low Confidence)"]
            
        # Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        
        img_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Base64 enc
        viz_image = Image.fromarray(visualization)
        buffered = io.BytesIO()
        viz_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        response_data = {
            "image": "data:image/png;base64," + img_str,
            "true_labels": true_labels_str.replace("|", ", "),
            "predicted_labels": ", ".join(top_pred)
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
