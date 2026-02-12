# 📰 News Text Classification Pipeline

An end-to-end NLP machine learning pipeline for classifying news articles into categories using **TF-IDF vectorization** and **Linear Support Vector Machine (LinearSVC)**.

This project demonstrates a clean, modular, and production-style ML workflow including preprocessing, feature engineering, training, evaluation, and configuration management.

---

## 🚀 Project Overview

The goal of this project is to build a scalable and reproducible text classification system using the **AG News Dataset**.

The pipeline processes raw news data and classifies each article into its respective category using traditional NLP and machine learning techniques.

This project emphasizes:

- Clean architecture
- Modular design
- Reproducibility
- Maintainability
- Clear ML workflow separation

---

## 🧠 Problem Statement

Given a dataset containing news articles with titles and descriptions, classify each article into one of multiple predefined categories.

This is a **multi-class text classification problem**.

---

## 🏗️ Project Architecture

The system follows a structured ML pipeline:

Raw Dataset
↓
Data Preprocessing
↓
Cleaned Dataset
↓
TF-IDF Feature Engineering
↓
Train/Test Split (Stratified)
↓
Model Training (Linear SVM)
↓
Model Serialization
↓
Model Evaluation
↓
Metrics Storage


Each stage is modular and separated into dedicated files for maintainability.

---

## 📂 Project Structure

classification_project/
│
├── data/
│ ├── raw/
│ │ └── train.csv
│ └── processed/
│ └── cleaned.csv
│
├── models/
│ └── news_classifier.pkl
│
├── results/
│ └── metrics.txt
│
├── src/
│ ├── config.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md


---

## 🔁 Workflow Explanation

### 1️⃣ Data Preprocessing

File: `data_preprocessing.py`

- Load raw dataset
- Combine `Title` and `Description`
- Convert text to lowercase
- Remove special characters using regex
- Remove English stopwords (NLTK)
- Save cleaned dataset

Output:
data/processed/cleaned.csv


---

### 2️⃣ Feature Engineering

File: `feature_engineering.py`

- Load cleaned dataset
- Separate features and labels
- Convert text into numerical vectors using **TF-IDF**
- Limit vocabulary to 5000 features
- Perform **stratified train-test split**
- Ensure reproducibility using `random_state`

Why TF-IDF?

- Reduces impact of common words
- Improves discriminative power
- Works efficiently with linear models

---

### 3️⃣ Model Training

File: `train.py`

Model used:
LinearSVC (Support Vector Machine)


Why Linear SVM?

- Performs well on high-dimensional sparse data
- Fast and memory efficient
- Strong generalization for text classification

The following artifacts are saved using `joblib`:

- Trained model
- Fitted TF-IDF vectorizer
- Test dataset

Output:
models/news_classifier.pkl


---

### 4️⃣ Model Evaluation

File: `evaluate.py`

Metrics calculated:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

Results are stored in:

results/metrics.txt


Example performance:
Accuracy: ~90–92%


---

### 5️⃣ Entry Point

File: `main.py`

Running:

```bash
python main.py
Executes the full pipeline:

Preprocessing

Feature Engineering

Training

Evaluation

🛠️ Technologies & Frameworks
Python 3.x

Pandas

Scikit-learn

NLTK

Joblib

Regex

⚙️ Configuration Management
All parameters and file paths are centralized in:

src/config.py
Includes:

File paths

TEST_SIZE

RANDOM_STATE

MAX_FEATURES

This improves:

Maintainability

Reproducibility

Scalability

📊 Model Performance
The model achieves approximately:

Accuracy: ~91%
With strong precision, recall, and F1-scores across all classes.
