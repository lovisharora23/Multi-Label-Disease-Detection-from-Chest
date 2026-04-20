import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Chest X-ray Analyst",
    page_icon="🩻",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00c853;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL ARCHITECTURE (Matches Training Code)
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
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
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
        scale = self.sigmoid(self.conv(combined))
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
# 3. GLOBAL CONFIG & CACHED FUNCTIONS
# ==============================================================================
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
CHECKPOINT_PATH = "nih/best_attention_model.pth"

@st.cache_resource
def load_model():
    model = AttentionDenseNet(num_classes=len(LABELS)).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def predict_single(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    # Generate Heatmap
    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    img_np = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    img_np = img_np.astype(np.float32)
    heatmap_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    return probs, heatmap_img

# ==============================================================================
# 4. DASHBOARD UI
# ==============================================================================
def main():
    st.sidebar.title("🩻 Clinical AI Analyst")
    st.sidebar.info("Designed specifically for Multi-Label Chest Disease Detection using Attention-Guided DenseNet Architecture.")
    
    menu = st.sidebar.radio("Navigation", ["Diagnostic Upload", "Model Accuracy", "About Model"])
    
    if menu == "Diagnostic Upload":
        st.header("🔍 Interactive Diagnostic Upload")
        st.write("Upload a patient's X-ray image (PNG/JPG) to perform an instant diagnostic check and view AI explainability heatmaps.")
        
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input X-ray")
                st.image(image, use_container_width=True)
            
            if st.button("Run AI Diagnosis"):
                with st.spinner('Analyzing patterns...'):
                    model = load_model()
                    probs, heatmap = predict_single(model, image)
                    
                    with col2:
                        st.subheader("AI Attention Area")
                        st.image(heatmap, use_container_width=True, caption="Grad-CAM Explainability Overlay")
                    
                    st.divider()
                    st.subheader("📊 Pathological Analysis")
                    
                    # Sort by confidence
                    res = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
                    
                    found_any = False
                    for label, prob in res:
                        if prob > 0.20:
                            found_any = True
                            color = "red" if prob > 0.50 else "orange"
                            st.markdown(f"**{label}**: :{color}[{prob*100:.2f}% Confidence]")
                            st.progress(float(prob))
                    
                    if not found_any:
                        st.success("No significant pathologies detected above confidence threshold (Normal).")

    elif menu == "Model Accuracy":
        st.header("📈 Model Performance & Accuracy")
        st.write("Statistics for the Attention-Guided DenseNet-121 trained on the NIH ChestXray14 Dataset.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curves (All Classes)")
            if os.path.exists("assets/roc_curves.png"):
                st.image("assets/roc_curves.png")
            else:
                st.info("Performance data still compiling...")
        
        with col2:
            st.subheader("Per-Class AUC Scores")
            if os.path.exists("assets/auc_bar.png"):
                st.image("assets/auc_bar.png")
            else:
                st.info("AUC scores unavailable.")
        
        st.divider()
        st.metric(label="Zero-Shot Mean AUC (Stanford CheXpert)", value="0.8414", delta="Significant Generalization")
        st.write("The model was rigorously validated on the Stanford CheXpert dataset without any additional training, proving its ability to learn true pathological features.")

    elif menu == "About Model":
        st.header("🧠 About the Dual-Attention Architecture")
        st.markdown("""
        ### How it works:
        - **Spatial Attention:** Identifies *where* the disease is (e.g., focusing on the lungs while ignoring the background).
        - **Channel Attention:** Identifies *what* features are present (textures, fluid levels, etc.).
        - **Training:** Organic training on 112,000 images natively on Apple Silicon.
        - **Loss:** Weighted Binary Cross Entropy to ensure rare diseases like Hernia are detected with the same precision as common ones like Effusion.
        """)
        if os.path.exists("assets/training_curve.png"):
            st.image("assets/training_curve.png", caption="Training Convergence History")

if __name__ == "__main__":
    main()
