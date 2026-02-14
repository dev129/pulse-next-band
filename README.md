# 🫀 Deep Learning from PPG Signals using Deep Learning with Activity Tracking.

## 📌 Overview

This project implements a deep learning pipeline to classify whether a person has **Myocardial Infarction (MI)** using **Photoplethysmogram (PPG)** time-series signals.

The model uses a **1D Convolutional Neural Network (CNN)** to learn morphological patterns in pulse waveforms that indicate cardiac abnormalities.

This is a **binary time-series classification problem**:

* **Input:** Fixed-length PPG signal (2000 time steps)
* **Output:**

  * `0` → Normal
  * `1` → Myocardial Infarction (MI)

---

## 🧠 Why This Matters

Myocardial Infarction (heart attack) detection traditionally requires ECG and clinical evaluation.
PPG signals from wearable devices offer a non-invasive alternative for early detection.

This project explores the feasibility of automated MI detection using PPG.

---

## 📂 Dataset

* Each sample is a **segmented PPG signal**
* Shape per sample: `2000 time steps`
* No timestamp column required (implicit time axis)
* Final column contains the class label

Dataset format:

```
Sample 1: [x0, x1, ..., x1999] → Label
Sample 2: [x0, x1, ..., x1999] → Label
```

---

## ⚙️ Preprocessing Pipeline

1. Extract features and labels
2. Stratified train–test split
3. Per-signal normalization
4. Reshape for deep learning

Normalization formula:

```
(X − mean) / std
```

Reshaped to:

```
(samples, timesteps, channels)
```

---

## 🤖 Model Architecture

### 1D CNN for Time-Series Classification

* Convolution layers detect waveform patterns
* Pooling reduces dimensionality
* Global Average Pooling aggregates features
* Dropout prevents overfitting
* Sigmoid output for binary classification

Architecture summary:

```
Conv1D → MaxPooling → Conv1D → MaxPooling → Conv1D
→ GlobalAveragePooling → Dropout → Dense(sigmoid)
```

---

## 🏋️ Training Strategy

* Loss: Binary Crossentropy
* Optimizer: Adam
* Early stopping to prevent overfitting
* Validation split during training

---

## 📊 Evaluation Metrics

Since this is a medical classification task, multiple metrics are used:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Special focus on:

> Recall for MI class (avoiding false negatives)

---

## 🚀 Results

The model learns discriminative patterns in PPG signals and can classify MI vs Normal signals with strong performance on unseen data.

---

## 🧩 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

1. Clone the repository

```
git clone https://github.com/yourusername/your-repo-name.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the notebook or training script

---

## 📌 Future Improvements

* Signal filtering (bandpass)
* Cross-validation
* Patient-wise splitting
* Explainability methods (Grad-CAM for 1D)
* Real-time deployment on wearable devices

---

## 📜 License

MIT License

---

## ✨ Author

Devansh Vishwa & Saloni Naithani
