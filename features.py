"""Feature engineering module for Loan Approval Prediction.

This module defines the custom transformer and helper functions for computing
domain-specific financial ratios:
1. Total Assets Value = residential + commercial + luxury + bank assets
2. Loan-to-Income (LTI) Ratio = loan_amount / income_annum (safe division)
3. Loan-to-Asset (LTA) Ratio = loan_amount / total_assets_value (safe division avoiding div by 0)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for loan feature engineering.
    
    Safe for pickling and joblib serialization across train.py and app.py.
    """
    
    def __init__(self, add_total_assets=True, add_lti_ratio=True, add_lta_ratio=True):
        self.add_total_assets = add_total_assets
        self.add_lti_ratio = add_lti_ratio
        self.add_lta_ratio = add_lta_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to DataFrame if input is numpy array or dict
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            raise ValueError("Input to LoanFeatureEngineer must be a pandas DataFrame or dict")

        # Strip whitespace from columns if present
        df.columns = [col.strip() for col in df.columns]

        # Ensure required columns exist
        res = df['residential_assets_value'].astype(float)
        comm = df['commercial_assets_value'].astype(float)
        lux = df['luxury_assets_value'].astype(float)
        bank = df['bank_asset_value'].astype(float)
        income = df['income_annum'].astype(float)
        loan = df['loan_amount'].astype(float)

        # 1. Total assets value
        total_assets = res + comm + lux + bank
        if self.add_total_assets:
            df['total_assets_value'] = total_assets

        # 2. Loan-to-Income (LTI) ratio: safe division
        if self.add_lti_ratio:
            df['loan_to_income_ratio'] = np.where(income > 0, loan / income, 0.0)

        # 3. Loan-to-Asset (LTA) ratio: safely handle total_assets == 0
        if self.add_lta_ratio:
            df['loan_to_asset_ratio'] = np.where(total_assets > 0, loan / total_assets, 0.0)

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to apply feature engineering directly to a DataFrame."""
    transformer = LoanFeatureEngineer()
    return transformer.transform(df)
