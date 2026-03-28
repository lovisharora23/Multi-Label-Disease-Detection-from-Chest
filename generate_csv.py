import os
import csv
import random

image_dirs = ['./nih/images', './nih/images_002/images']
image_files = []
for d in image_dirs:
    if os.path.isdir(d):
        image_files.extend(os.listdir(d))

labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']

csv_path = './nih/Data_Entry_2017.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image Index', 'Finding Labels'])
    for img in image_files:
        if img.endswith('.png'):
            finding = random.choice(labels)
            if finding != 'No Finding' and random.random() > 0.5:
                finding += '|' + random.choice(labels)
            writer.writerow([img, finding])

print(f"Generated {csv_path} with {len((image_files))} entries")
