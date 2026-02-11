# Brain Tumour Classification (MRI) — Deep Learning (CNN)

A deep learning project that classifies brain MRI scans into **four classes**:

- **Glioma**
- **Meningioma**
- **Pituitary tumour**
- **No tumour**

This work was developed as an academic project (LSBU Deep Learning module) and includes multiple CNN architectures, evaluation, and a **Streamlit demo app** for real-time inference. :contentReference[oaicite:1]{index=1}

---

## Project Goals

Brain tumour diagnosis from MRI is time-consuming and can vary between clinicians.  
This project aims to build a **CNN-based decision-support prototype** that predicts the tumour type from an input MRI image and displays predicted probabilities.

> **Important:** This is a research/educational prototype — **not** a clinically approved diagnostic tool. :contentReference[oaicite:2]{index=2}

---

## Dataset Summary

The dataset is organized into predefined splits (Train / Validation / Test) with class folders.

After duplicate handling, the dataset sizes are: :contentReference[oaicite:3]{index=3}

| Split | Images |
|------|--------|
| Train | 4,420 |
| Validation | 571 |
| Test | 576 |
| **Total** | **5,567** |

### Duplicate Handling (Important)
- A hash-based scan found duplicates grouped into **182 duplicate groups**.
- Duplicates were removed from **Train/Validation** (257 images removed).
- Duplicates in **Test** were kept to avoid severe class imbalance and maintain fair evaluation. :contentReference[oaicite:4]{index=4}

---

## Data Preparation

Key preprocessing steps: :contentReference[oaicite:5]{index=5}

- Resize all images to **224 × 224**
- Normalize pixel values to **[0, 1]** by dividing by **255**
- Standardize channels (dataset included RGB / grayscale / rare RGBA)
- Apply **data augmentation** on training data:
  - rotation, shifts, zoom, shear, horizontal flip

---

## Models

Three CNN models were developed and compared:

### Model 1 — Baseline CNN
- 3 Conv + MaxPool blocks
- Dense + Dropout
- Trained ~10 epochs

### Model 2 — Deeper CNN + Regularization
- Convolutional blocks with:
  - L2 regularization, BatchNorm, Dropout
- Used callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Class weighting tested to improve meningioma performance

### Model 3 — Five-Layer CNN (Final)
- 5 Conv + MaxPool stages (deeper feature extraction)
- Selected as final due to best overall performance and improved meningioma classification :contentReference[oaicite:6]{index=6}

---

## Results (Test Set)

Final reported test accuracies: :contentReference[oaicite:7]{index=7}

- **Baseline CNN:** ~80%
- **Deeper CNN:** ~82%
- **Five-layer CNN (Final):** **~87%**

The final model improved meningioma performance substantially (reduced meningioma → “no tumour” misclassification compared to earlier models). :contentReference[oaicite:8]{index=8}

---

## Streamlit App (Demo Deployment)

A Streamlit web app is used for interactive inference:

Workflow: :contentReference[oaicite:9]{index=9}
1. Load trained model
2. Upload an MRI image
3. Resize + normalize
4. Predict tumour class + probability distribution
5. Display predicted class + confidence + bar chart
6. Show an educational “clinical suggestion” panel (non-medical)

### Exception Handling
The app is designed to accept **brain MRI-like grayscale inputs** and reject unrelated coloured/natural images to reduce misleading predictions. :contentReference[oaicite:10]{index=10}

---

## How to Run Locally

### 1) Create environment & install dependencies
```bash
pip install -r requirements.txt

Run the Streamlit app

Make sure the following files are in the **same folder**:

- `brain_train_tumor_final_model.h5`
- `brain_tumor_detection.py`
- `class_labels.json`

Then open **Anaconda Prompt** (or any terminal), navigate to that folder, and run:

```bash
streamlit run brain_tumor_detection.py
