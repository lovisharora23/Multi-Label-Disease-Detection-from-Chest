import matplotlib.pyplot as plt
import os
import numpy as np

auc_scores = {
    'Atelectasis': 0.8073,
    'Cardiomegaly': 0.9033,
    'Consolidation': 0.8033,
    'Edema': 0.9069,
    'Effusion': 0.8793,
    'Emphysema': 0.9183,
    'Fibrosis': 0.8137,
    'Hernia': 0.9132,
    'Infiltration': 0.7060,
    'Mass': 0.8668,
    'Nodule': 0.7632,
    'Pleural_Thickening': 0.8039,
    'Pneumonia': 0.7604,
    'Pneumothorax': 0.8830
}

mean_auc = 0.8378

labels_sorted = sorted(auc_scores, key=auc_scores.get, reverse=True)
vals_sorted   = [auc_scores[l] for l in labels_sorted]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels_sorted, vals_sorted,
               color=['green' if v >= 0.8 else 'orange' if v >= 0.7 else 'red'
                      for v in vals_sorted])
plt.axhline(mean_auc, color='blue', linestyle='--', label=f'Mean AUC = {mean_auc:.4f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel("AUC")
plt.title("Per-Class AUC — Proposed Attention CNN")
plt.legend()
plt.tight_layout()

# Save to assets, root, and nih directory to ensure all places are updated
assets_path = os.path.join('assets', 'auc_bar.png')
root_path = 'auc_bar.png'
nih_path = os.path.join('nih', 'auc_bar.png')

plt.savefig(assets_path, dpi=150)
print(f"Saved {assets_path}")

try:
    plt.savefig(root_path, dpi=150)
    print(f"Saved {root_path}")
except:
    pass

try:
    plt.savefig(nih_path, dpi=150)
    print(f"Saved {nih_path}")
except:
    pass

plt.close()
