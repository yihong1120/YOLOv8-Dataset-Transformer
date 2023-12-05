#!/bin/bash

# 確保腳本在出現錯誤時終止
set -e

# 安裝所需的依賴項
echo "Installing required packages..."
pip install -r requirements.txt

# 準備數據集
echo "Preparing dataset..."
python dataset_preparation.py --markers train20X20 --irrelevant irrelevant --output output --total_images 7000 --train_ratio 0.9

# 訓練模型
echo "Training model..."
python train.py --data_config output/data.yaml --epochs 100 --model_name yolov8n.pt

echo "Script completed successfully."
