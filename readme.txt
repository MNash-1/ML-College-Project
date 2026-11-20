Build a complete end-to-end Machine Learning project for "Loan Approval Prediction" using the dataset located at `data/loan_approval_dataset.csv`.

Follow this exact directory structure and specifications:

1. **Exact Folder Structure:**
   ```text
   loan-approval-prediction/
   ├── .venv/                         # Python virtual environment
   ├── data/
   │   └── loan_approval_dataset.csv  # Original Kaggle dataset (already present)
   ├── models/
   │   └── loan_pipeline.joblib       # Saved preprocessing + best ML model pipeline
   ├── screenshots/
   │   ├── 1_target_distribution.png  # EDA: Countplot of loan_status
   │   ├── 2_cibil_vs_status.png      # EDA: Boxplot of cibil_score vs loan_status
   │   ├── 3_income_vs_loan.png       # EDA: Scatterplot of income_annum vs loan_amount
   │   ├── 4_correlation_heatmap.png  # EDA: Correlation heatmap of numeric features
   │   ├── 5_confusion_matrix.png     # Evaluation: Confusion matrix of best model
   │   └── metrics_summary.txt        # Evaluation metrics table for all 3 models
   ├── train.py                       # Data cleaning, EDA plots, training & evaluation
   ├── app.py                         # Interactive Streamlit web application
   ├── requirements.txt               # Project dependencies
   └── README.md                      # Setup, training, and execution instructions