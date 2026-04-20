#!/bin/bash

# Script to sync local AI changes to GitHub for deployment
echo "🚀 Syncing Live AI Lab to GitHub..."

# 1. Add new files
git add app.py predict.py index.html nih/best_attention_model.pth

# 2. Commit
git commit -m "feat: Add Interactive AI Diagnostic Lab and model weights"

# 3. Push to main
git push origin main

echo "✅ Success! Your code is now on GitHub."
echo "-------------------------------------------------------"
echo "NEXT STEP: Go to https://share.streamlit.io/"
echo "1. Sign in with GitHub."
echo "2. Click 'New app'."
echo "3. Select this repository and 'app.py'."
echo "4. Click 'Deploy'!"
echo "-------------------------------------------------------"
