data set link:- https://drive.google.com/file/d/127JqP3WGjBVihR-ZcUR86T3wwy3_g63v/view

# 💳 Online Payment Fraud Detection using Machine Learning in Python

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

This project focuses on building a **machine learning-based fraud detection system** for online payments. It uses classification algorithms to predict whether a transaction is fraudulent or legitimate, based on historical transaction data.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 About the Project

Online payment fraud is a major issue in digital transactions. This project leverages machine learning models to accurately detect and classify fraudulent transactions in real-time.

**Objectives:**
- Explore and preprocess the dataset.
- Apply multiple machine learning models (Logistic Regression, Random Forest, XGBoost, etc.).
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Visualize key results.

---

## 📂 Dataset

We used a **publicly available dataset** that includes thousands of online transaction records with features like transaction type, amount, old balance, new balance, etc.

- Source: [Kaggle or other dataset link]
- Features include:
  - `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, etc.
  - Target: `isFraud` (0 = legitimate, 1 = fraud)

---

## 🧰 Tech Stack

- Python 3.10+
- GOOGLE COLLAB Notebook
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb


📊 Modeling
The following models were trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

XGBoost

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Evaluation Metrics:

Confusion Matrix

Accuracy

Precision

Recall

F1 Score

ROC-AUC Curve

✅ Results
Model	Accuracy	Precision	Recall	F1 Score
Random Forest	99.7%	96%	92%	94%
XGBoost	99.8%	97%	93%	95%
Logistic Regression	95.1%	65%	60%	62%

Results may vary depending on data preprocessing steps.

🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

