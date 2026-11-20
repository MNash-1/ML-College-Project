"""Training script for Loan Approval Prediction.

Performs:
1. Data loading and cleaning (whitespace stripping, column standardization).
2. Exploratory Data Analysis (EDA) and saving required plots to screenshots/.
3. Feature engineering and pipeline building with LoanFeatureEngineer.
4. Model training and evaluation across 3 algorithms:
   - Logistic Regression (Baseline)
   - Random Forest Classifier
   - Gradient Boosting Classifier
5. Exporting metrics summary table to screenshots/metrics_summary.txt.
6. Exporting best pipeline to models/loan_pipeline.joblib.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from features import LoanFeatureEngineer


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load dataset and clean column names and string values."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Clean whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Clean whitespace from all string/object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop identifier column if present
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: str):
    """Generate and save the 4 required EDA plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'figure.autolayout': True})

    # 1. Target Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='loan_status', palette=['#2ecc71', '#e74c3c'])
    plt.title('1. Target Distribution: Loan Approval Status', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Loan Status', fontsize=12)
    plt.ylabel('Applicant Count', fontsize=12)
    for p in ax.patches:
        count = int(p.get_height())
        pct = f"{count / len(df) * 100:.1f}%"
        ax.annotate(f"{count} ({pct})", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plot1_path = os.path.join(output_dir, '1_target_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. CIBIL vs Status
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='loan_status', y='cibil_score', palette=['#2ecc71', '#e74c3c'])
    plt.title('2. CIBIL Score vs Loan Status', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Loan Status', fontsize=12)
    plt.ylabel('CIBIL Score', fontsize=12)
    plot2_path = os.path.join(output_dir, '2_cibil_vs_status.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Income vs Loan Amount
    plt.figure(figsize=(8, 5.5))
    sns.scatterplot(
        data=df, x='income_annum', y='loan_amount',
        hue='loan_status', palette=['#2ecc71', '#e74c3c'],
        alpha=0.6, s=35
    )
    plt.title('3. Income Annum vs Loan Amount by Status', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Annual Income (₹)', fontsize=12)
    plt.ylabel('Loan Amount (₹)', fontsize=12)
    plt.legend(title='Loan Status')
    plot3_path = os.path.join(output_dir, '3_income_vs_loan.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title('4. Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold', pad=12)
    plot4_path = os.path.join(output_dir, '4_correlation_heatmap.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")


def train_and_evaluate(df: pd.DataFrame, screenshots_dir: str, models_dir: str):
    """Train 3 models, evaluate, export metrics, confusion matrix, and best pipeline."""
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Encode target: Approved = 1, Rejected = 0
    y = (df['loan_status'] == 'Approved').astype(int)
    X = df.drop(columns=['loan_status'])

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Feature definitions (including the 3 engineered features from features.py)
    numeric_features = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value',
        'total_assets_value', 'loan_to_income_ratio', 'loan_to_asset_ratio'
    ]
    categorical_features = ['education', 'self_employed']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = []
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('engineer', LoanFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc
        })
        trained_pipelines[name] = (pipeline, y_pred, f1)
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

    results_df = pd.DataFrame(results)

    # 5. Format & save metrics summary table
    summary_text = "=" * 80 + "\n"
    summary_text += " " * 22 + "LOAN APPROVAL PREDICTION - MODEL EVALUATION\n"
    summary_text += "=" * 80 + "\n\n"
    summary_text += results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n"
    
    # Identify best model by F1-Score
    best_row = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    summary_text += f"Winning Model: {best_model_name} (F1-Score: {best_row['F1-Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f})\n"
    summary_text += "=" * 80 + "\n"

    metrics_file = os.path.join(screenshots_dir, 'metrics_summary.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSaved metrics summary to: {metrics_file}")
    print(summary_text)

    # 6. Confusion Matrix of best model
    best_pipeline, best_y_pred, _ = trained_pipelines[best_model_name]
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'5. Confusion Matrix: {best_model_name}', fontsize=13, fontweight='bold', pad=12)
    plt.grid(False)
    plot5_path = os.path.join(screenshots_dir, '5_confusion_matrix.png')
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot5_path}")

    # 7. Save best pipeline
    model_save_path = os.path.join(models_dir, 'loan_pipeline.joblib')
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best pipeline ({best_model_name}) to: {model_save_path}")


def main():
    data_path = os.path.join('data', 'loan_approval_dataset.csv')
    screenshots_dir = 'screenshots'
    models_dir = 'models'

    df = load_and_clean_data(data_path)
    generate_eda_plots(df, screenshots_dir)
    train_and_evaluate(df, screenshots_dir, models_dir)
    print("\nTraining and artifact generation completed successfully!")


if __name__ == '__main__':
    main()
