# 🏦 Loan Approval Prediction System & Fintech Underwriting Portal

An end-to-end Machine Learning system and modern banking portal that evaluates retail loan applications based on applicant credit profile, debt servicing capacity, collateral haircuts, and CIBIL bureau scores.

---

## 📌 Project Overview

In commercial banking, assessing credit risk and loan default probability is critical. This project implements a complete, leak-free Scikit-Learn machine learning pipeline featuring:
1. **Automated Data Cleaning**: Robust ingestion with string and column whitespace stripping.
2. **Domain-Specific Feature Engineering**: Custom financial ratios (Total Collateral Assets, Loan-to-Income, Loan-to-Asset) safely handling zero-asset edge cases.
3. **Advanced Banking Financial Metrics**:
   - **Monthly EMI**: Reducing-balance installment formula at 9.50% p.a. benchmark rate.
   - **Fixed Obligation to Income Ratio (FOIR)**: Debt servicing capacity metric with health tier thresholds.
   - **Regulatory Collateral Haircuts**: Conservative collateral appraisal (80% Residential, 75% Commercial, 90% Bank Deposits, 50% Luxury).
4. **SHAP Model Explainability**: Local decision factor attribution via SHAP `TreeExplainer` on Gradient Boosting trees.
5. **Multi-Model Benchmark**: Evaluates **Logistic Regression**, **Random Forest**, and **Gradient Boosting**, selecting the best performer based on F1-score and ROC-AUC.
6. **Serialization Safety**: Shared `features.py` module preventing pickle `AttributeError` during production inference.
7. **Executive Memo Export**: One-click download of an audit-ready Markdown Credit Risk & Underwriting Memorandum.
8. **Interactive Fintech Dashboard**: Modern Streamlit portal styled with navy/slate banking aesthetics and minimal toolbar.

---

## 📂 Directory Structure

```text
loan-approval-prediction/
├── .streamlit/
│   └── config.toml                # Banking theme & minimal toolbar configuration
├── .venv/                         # Python virtual environment
├── data/
│   └── loan_approval_dataset.csv  # Kaggle loan dataset (4,269 records)
├── models/
│   └── loan_pipeline.joblib       # Persisted preprocessing + best ML model pipeline
├── screenshots/
│   ├── 1_target_distribution.png  # EDA: Countplot of loan_status
│   ├── 2_cibil_vs_status.png      # EDA: Boxplot of cibil_score vs loan_status
│   ├── 3_income_vs_loan.png       # EDA: Scatterplot of income_annum vs loan_amount
│   ├── 4_correlation_heatmap.png  # EDA: Correlation heatmap of numeric features
│   ├── 5_confusion_matrix.png     # Evaluation: Confusion matrix of best model
│   └── metrics_summary.txt        # Evaluation metrics table across all 3 models
├── features.py                    # Custom feature engineering transformer (shared module)
├── train.py                       # Data cleaning, EDA plots, model training & evaluation
├── app.py                         # Interactive Streamlit fintech application
├── requirements.txt               # Pinned dependencies (including SHAP)
└── README.md                      # Project documentation and execution instructions
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Pip

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### 1. Run Data Preprocessing & Model Training
Execute `train.py` to clean the dataset, generate the 5 EDA and evaluation plots in `screenshots/`, train all 3 candidate models, and serialize the best pipeline to `models/loan_pipeline.joblib`:

```bash
python train.py
```

### 2. Launch the Streamlit Web Application
Run the interactive dashboard:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 Financial Calculations & Feature Engineering

### 1. Reducing-Balance Monthly EMI
$$\text{EMI} = \frac{P \cdot r \cdot (1 + r)^n}{(1 + r)^n - 1}$$
*Where $P = \text{Loan Amount}$, $r = \frac{9.5\%}{12} = \text{monthly interest rate}$, $n = \text{tenure in months}$.*

### 2. Fixed Obligation to Income Ratio (FOIR)
$$\text{FOIR} = \frac{\text{Monthly EMI}}{\text{Monthly Gross Income}} \times 100\%$$
- **$\le 40\%$**: Comfortable Debt Load
- **$40\% - 55\%$**: Moderate Debt Load
- **$> 55\%$**: Critical Debt Burden

### 3. Regulatory Collateral Haircuts
$$\text{Effective Collateral} = 0.80 \times \text{Res} + 0.75 \times \text{Comm} + 0.90 \times \text{Bank} + 0.50 \times \text{Lux}$$

### 4. Custom Feature Engineering Ratios (`features.py`)
- **Total Assets Value**: $\sum \text{Assets}$
- **Loan-to-Income (LTI)**: $\frac{\text{Loan Amount}}{\text{Annual Income}}$
- **Loan-to-Asset (LTA)**: $\frac{\text{Loan Amount}}{\text{Total Assets}}$ (with $0.0$ division safeguard)

---

## 📊 Evaluation & Model Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 90.75% | 91.54% | 93.79% | 0.9265 | 0.9727 |
| **Random Forest** | 99.88% | 99.81% | 100.0% | 0.9991 | 1.0000 |
| **Gradient Boosting (Best)** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
