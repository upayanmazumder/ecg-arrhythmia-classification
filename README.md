# ECG Arrhythmia Classification

A Machine Learning project to detect and classify heart arrhythmias from ECG signals.

This project uses the [ECG Arrhythmia Classification Dataset on Kaggle](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset?resource=download) to train models that can automatically distinguish between normal and abnormal heartbeats. Built as part of an academic project on Artificial Intelligence & Machine Learning.

---

## Overview

Electrocardiograms (ECGs) are recordings of the heart’s electrical activity.
By analyzing these signals, we can identify irregular heartbeat patterns — **arrhythmias** — which can indicate potential cardiac problems.

This project focuses on:

- Preprocessing ECG data
- Extracting key time-domain and frequency-domain features
- Training machine learning and deep learning models for classification
- Evaluating performance and visualizing results

---

## Project Structure

```
ecg-arrhythmia-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                # Original dataset (unmodified)
│   ├── processed/          # Cleaned & split data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── baseline_model.pkl
│   └── cnn_model.h5
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.json
│   └── training_logs/
│
└── reports/
    ├── final_report.pdf
    └── presentation.pptx
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/upayanmazumder/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Workflow

1. **Data Exploration** → Understand ECG signal structure
2. **Preprocessing** → Filtering, normalization, segmentation
3. **Feature Extraction** → Derive HRV, QRS intervals, etc.
4. **Model Training** → Try ML (RandomForest, SVM) and DL (CNN/LSTM)
5. **Evaluation** → Accuracy, F1-score, confusion matrix
6. **Visualization** → Compare results and insights

---

## Quick Start

Train the model using the prepared dataset:

```bash
python src/train.py
```

Evaluate the trained model:

```bash
python src/evaluate.py
```

View results in the `results/` folder.

---

## Tech Stack

- **Python 3.10+**
- **NumPy**, **Pandas**, **Matplotlib**
- **Scikit-learn**
- **TensorFlow / PyTorch**
- **Jupyter Notebooks**

---

## Results

| Model         | Accuracy           | F1 Score           |
| ------------- | ------------------ | ------------------ |
| Random Forest | _yet to be filled_ | _yet to be filled_ |
| CNN           | _yet to be filled_ | _yet to be filled_ |

_(Final results will be updated after training completion.)_

![Confusion Matrix](results/confusion_matrix.png)

---

## Future Work

- Add LSTM/1D-CNN hybrid architecture
- Improve feature extraction pipeline
- Implement real-time ECG classification
- Deploy using Streamlit or FastAPI

---

## Contributors

**Upayan Mazumder**
[GitHub](https://github.com/upayanmazumder) · [LinkedIn](https://www.linkedin.com/in/upayanmazumder/)

---

## License

This project is licensed under the **GNU Lesser General Public License (LGPL)**.
