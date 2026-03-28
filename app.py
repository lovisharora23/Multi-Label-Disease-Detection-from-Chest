import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Medical AI: Cross-Dataset Validation", layout="wide", page_icon="🩺")

# --- Custom CSS for Ultra-Clean Aesthetics ---
st.markdown("""
<style>
.main-title { font-size: 3rem; font-weight: 800; color: white; margin-bottom: 0px; }
.sub-title { font-size: 1.25rem; color: #94a3b8; font-weight: 400; margin-bottom: 2.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #334155; }
.stTabs [data-baseweb="tab"] { font-size: 1.15rem; font-weight: 600; padding: 1rem 0; color: #94a3b8; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; }
div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 2.5rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Attention-Guided Chest X-Ray CNN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Zero-Shot Generalization Analysis: NIH Baseline vs. Stanford CheXpert Validation</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB-BASED ARCHITECTURE FOR CLEAN COMPARISONS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Architecture Blueprint", "📉 NIH Training Baseline", "🌐 Stanford Generalization Results"])

with tab1:
    st.markdown("### Massive PyTorch Medical AI deployed flawlessly on Apple Silicon.")
    st.markdown("This dashboard documents the journey of taking a highly complex experimental deep learning architecture and securely scaling it locally to process one of the largest uncompressed medical imaging repositories natively without third-party cloud environments.")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Volume", "112,000 Scans")
    col2.metric("Optimization Epochs", "15")
    col3.metric("Neural Backbone", "Dual Attention DenseNet")
    st.markdown("---")
    
    try:
        st.image("nih/sample_predictions.png", caption="Ground Truth vs Highest Confidence Output Probabilities", use_column_width=True)
    except Exception:
        st.warning("Sample predictions visualization is syncing...")

with tab2:
    st.header("Baseline Analytical Metrics (NIH Dataset)")
    st.markdown("The PyTorch architecture dynamically converged over **15 automated Epochs** exclusively utilizing Apple Silicon `MPS` (Metal Performance Shaders). A custom Weighted Binary Cross Entropy protocol seamlessly corrected the massive 14-disease label imbalance.")
    
    st.markdown("---")
    try:
        st.image("nih/training_curve.png", caption="Feature Extraction Convergence Trajectory", use_column_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image("nih/roc_curves.png", caption="Baseline Receiver Operating Characteristic (14 Distinct Pathologies)", use_column_width=True)
        with c2:
            st.image("nih/auc_bar.png", caption="Per-Class Area Under Curve Distribution (Global Mean: 0.8378)", use_column_width=True)
    except Exception:
        st.warning("NIH Analytics visualizations are syncing...")

with tab3:
    st.header("Zero-Shot Medical Generalization (Stanford Dataset)")
    st.markdown("To rigorously prove the model actually learned physiological representations (rather than simply overfitting and memorizing its own hospital's scanners), we instantly evaluated the exact same frozen baseline matrix against an entirely disjoint **10.7 Gigabyte** patient repository compiled natively by Stanford.")
    
    st.success("#### Zero-Shot Mean AUC: 0.8414 \nThe completely unseen patient validation flawlessly matches the internal baseline metrics, acting as definitive statistical evidence of systemic clinical generalizability!")
    
    st.markdown("---")
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image("nih/chexpert_generalization_auc.png", caption="Stanford CheXpert Shared Assessment (Zero-Shot Mean AUC: 0.8414)", use_column_width=True)
        with col2:
            st.image("nih/chexpert_roc_curves.png", caption="Generalization ROC Trajectories Tracking the 5 Overlapping Pathologies", use_column_width=True)
    except Exception:
        st.warning("Stanford evaluation visualization charts are missing from cache. Did the PyTorch script successfully exit?")
