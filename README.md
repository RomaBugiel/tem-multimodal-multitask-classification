# TEM Multimodal Multi-Task Classification (ACCORDS Project)

This repository contains a PyTorch pipeline for the **multi-task and multimodal classification of TEM images** developed as part of the **ACCORDS project**.  
The goal is to automatically categorize transmission electron microscopy (TEM) images using convolutional neural networks (CNNs) combined with metadata (magnification values).  

---

## ✨ Project Overview

Manual classification of TEM images is time-consuming and subjective.  
This project introduces a machine learning workflow that:

- Uses **convolutional backbones** (ResNet18/34, EfficientNet-B0, ConvNeXt-Tiny) pretrained on ImageNet,  
- Incorporates **metadata (magnification)** via a dedicated fully connected branch,  
- Performs **multi-task classification** across five categorical dimensions simultaneously:
  1. Particle count  
  2. Material composition  
  3. Morphology  
  4. Nanostructures  
  5. Usability  

The multimodal design enables the model to learn magnification-specific priors, improving classification consistency across tasks.

---

## 📂 Dataset

- **Input data:** TEM images (`.tif`, `.jpg`, `.png`) paired with structured label sheets (`.csv`, `.xlsm`).  
- Each image was manually annotated with **five textual descriptors**, later encoded into integer classes.  
- A total of **185 images** were used in the initial experiments.  
- Stratified splitting ensured balanced training/validation distributions across categories.  

---

## 🧠 Model Architecture

- **Backbone:**  
  Pretrained CNNs (ResNet18/34, EfficientNet-B0, ConvNeXt-Tiny) serve as image feature extractors.  

- **Metadata branch:**  
  A fully connected layer (`1 → 32 units + ReLU`) encodes magnification values.  

- **Fusion:**  
  Image and metadata features are concatenated into a joint representation.  

- **Task heads:**  
  Separate fully connected layers predict each of the five categorical outputs.  

This architecture allows efficient transfer learning from natural images to microscopy while adapting to the unique challenges of TEM data.

---

## ⚙️ Training Pipeline

1. **Preprocessing**
   - Resize images to a fixed size.  
   - Apply data augmentation (flips, rotations, color jitter).  
   - Normalize using ImageNet statistics.  
   - Encode categorical labels as integers.  

2. **Training**
   - Stratified train/validation split (80/20).  
   - DataLoader for batched training and evaluation.  
   - Cross-entropy loss for each task.  
   - Optimized with AdamW.  

3. **Evaluation**
   - Accuracy reported per task and backbone.  
   - Results visualized as summary tables and bar charts.  

---

## 📊 Results

- **Best models:** ConvNeXt-Tiny and EfficientNet-B0 achieved the highest overall validation accuracy (~77%).  
- **Task-level performance:**
  - Usability: >96%  
  - Particle count & Material composition: ~90%  
  - Nanostructures: 55–61%  
  - Morphology: <45% (most challenging, reflecting annotation ambiguity).  

These results highlight the **effectiveness of transfer learning** for TEM image classification, while also showing the limitations imposed by manual labeling quality and dataset size.

---

## 🔮 Next Steps

- Expand the dataset with additional annotated TEM images.  
- Refine label definitions, especially for morphology and nanostructures.  
- Involve domain experts to reduce labeling ambiguity.  
- Explore semi-supervised or self-supervised approaches to leverage unlabeled TEM data.  

---

## 📌 Project Context

This work was developed in the framework of the **ACCORDS project**, which focuses on applying deep learning and AI methods to the characterization of advanced carbon-based materials using electron microscopy.  

---

## 🚀 How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/tem-multimodal-multitask-classification.git
   cd tem-multimodal-multitask-classification
