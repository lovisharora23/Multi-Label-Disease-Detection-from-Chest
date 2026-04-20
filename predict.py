import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==============================================================================
# 1. ARCHITECTURE DEFINITION (Matches Training Code)
# ==============================================================================
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
    def __init__(self, num_classes):
        super().__init__()
        densenet = models.densenet121(weights=None)
        self.features = densenet.features
        self.attention = DualAttentionBlock(in_channels=1024, reduction=16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.features(x)
        attended = self.attention(features)
        pooled = self.gap(attended)
        flat = pooled.view(pooled.size(0), -1)
        out = self.classifier(self.dropout(flat))
        return out

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
CHECKPOINT_PATH = "nih/best_attention_model.pth"

# ==============================================================================
# 3. UTILITIES
# ==============================================================================
def load_model():
    model = AttentionDenseNet(num_classes=len(LABELS)).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model weights not found at {CHECKPOINT_PATH}")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✔ Successfully loaded model from epoch {checkpoint['epoch']}")
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return input_tensor, image

def run_prediction(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    return probs

def generate_heatmap(model, input_tensor, original_image, results_path="diagnostic_heatmap.png"):
    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    img_np = np.array(original_image.resize((224, 224))) / 255.0
    img_np = img_np.astype(np.float32)
    
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    plt.imsave(results_path, visualization)
    print(f"✔ AI Attention Heatmap saved to {results_path}")

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_xray_image>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    try:
        model = load_model()
        input_tensor, original_image = preprocess_image(img_path)
        probs = run_prediction(model, input_tensor)
        
        print("\n" + "="*40)
        print("  DIAGNOSTIC RESULTS  ")
        print("="*40)
        
        found_any = False
        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            if prob > 0.20:
                print(f"  [{label:<20}]  {prob*100:>5.2f}% confidence")
                found_any = True
        
        if not found_any:
            print("  No significant pathology detected (Normal).")
        
        print("="*40)
        
        generate_heatmap(model, input_tensor, original_image)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
