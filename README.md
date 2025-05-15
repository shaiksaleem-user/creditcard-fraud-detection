# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using various machine learning models, with an emphasis on handling class imbalance and maximizing precision-recall metrics.

---

## 📁 Project Structure

```
creditcard_fraud_detection_project/
│
├── data/                    # Contains the dataset
│   └── creditcard.csv
│
├── eda/                     # Exploratory Data Analysis
│   └── eda.ipynb
│
├── modeling/                # Modeling and evaluation
│   └── modeling.ipynb
│
├── models/                  # Trained ML model files
│   └── xgboost_fraud_model.pkl
│
├── reports/                 # Reports and documentation
│   └── report.md
│
├── utils/                   # Utility scripts (optional)
│   └── preprocessing.py
│
├── README.md                # Project documentation
└── requirements.txt         # Required Python packages
```

---

## 🎯 Objective

The goal of this project is to detect fraudulent transactions using historical transaction data. Given the extreme class imbalance (fraud cases ~0.17%), the focus is on optimizing recall and precision rather than just accuracy.

---

## 📊 Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent cases**: 492 (≈ 0.172%)
- **Features**: V1–V28 (PCA components), `Time`, `Amount`, and `Class` (target)

---

## 🛠️ Tools & Technologies

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA)

- Class imbalance visualized
- Distribution analysis of transaction amounts and time
- Correlation heatmaps
- PCA components examined for outlier behavior

---

## 🤖 Machine Learning Models Used

| Model            | Precision (Fraud) | Recall (Fraud) | ROC-AUC | PR-AUC |
|------------------|-------------------|----------------|---------|--------|
| Logistic Reg     | 0.06              | 0.92           | 0.97    | -      |
| Random Forest    | 0.96              | 0.74           | 0.95    | -      |
| XGBoost (Best)   | 0.85              | 0.84           | 0.98    | 0.87   |

> 🔥 **XGBoost** delivered the best balance between precision and recall.

---

## 📈 Feature Importance (XGBoost)

![Feature Importance](reports/feature_importance.png)

---

## ✅ Conclusion

- **XGBoost** is the most effective model, with high ROC-AUC and PR-AUC.
- Precision-recall tradeoff is critical in fraud detection due to class imbalance.
- Model has been saved using `joblib` for future inference.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/creditcard_fraud_detection_project.git
   cd creditcard_fraud_detection_project
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Run `eda/eda.ipynb` followed by `modeling/modeling.ipynb`.

---

## 📬 Contact

**Author**: Shaik Saleem  
**Email**: [shaiksaleembasha541@gmail.com]  
**License**: MIT
