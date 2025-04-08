# ViT-for-Facial-Recognition

This project implements **Vision Transformers (ViT)** for facial recognition as a component of a Deep Learning course. We leverage transfer learning using the pretrained `vit_b_32` architecture provided by `torchvision`. The model is trained on a publicly available dataset of celebrity faces to recognize and classify individual identities.

## 🧠 Overview

Vision Transformers have shown impressive performance on various computer vision tasks. This project adapts ViT for the specific task of facial recognition, using a deep learning approach built with PyTorch.

We utilize a **celebrity face recognition dataset** from Kaggle and apply **transfer learning** with a pretrained ViT model to classify images based on identity.

## 📂 Dataset

- **Name**: Celebrity Face Recognition Dataset  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset)
- The dataset contains folders of celebrity images labeled with their names.

## 🧰 Tech Stack

- Python 3.x  
- PyTorch  
- Torchvision (for `vit_b_32` model)  
- Jupyter Notebook  
- PIL / OpenCV for image processing  
- Matplotlib for visualization  

## 🔍 Model Details

- **Architecture**: `vit_b_32` from `torchvision.models`
- **Training Strategy**: Transfer learning on the pretrained ViT model
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam Optimizer
- **Metrics**: Accuracy and Loss Curve

