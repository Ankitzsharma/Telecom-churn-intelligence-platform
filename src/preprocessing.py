import pandas as pd
import numpy as np

def clean_data(df):
    """
    Perform data cleaning steps:
    1. Convert TotalCharges to numeric
    2. Handle missing values
    3. Remove identifiers
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Check for missing values in TotalCharges
    missing_tc = df['TotalCharges'].isnull().sum()
    if missing_tc > 0:
        print(f"Handling {missing_tc} missing values in TotalCharges. Strategy: Fill with 0 (assuming new customers).")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Map SeniorCitizen to Yes/No for consistency if needed, but it's already int 0/1 which is fine.
    # Actually, usually better to keep as 0/1.
    
    return df

def encode_data(df):
    """
    Encode categorical variables.
    - Binary variables: Label Encoding (Yes/No -> 1/0)
    - Multi-class variables: One-Hot Encoding
    """
    df = df.copy()
    
    # Identify binary columns (2 unique values) and multi-class columns
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    multi_cols = [col for col in df.columns if df[col].nunique() > 2 and df[col].dtype == 'object']
    
    # Binary encoding
    for col in binary_cols:
        # Assuming Yes/No or Male/Female
        if set(df[col].unique()) == {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        elif set(df[col].unique()) == {'Male', 'Female'}:
            df[col] = df[col].map({'Male': 1, 'Female': 0})
        else:
            # Fallback for other binary
            df[col] = pd.factorize(df[col])[0]
            
    # One-Hot Encoding for multi-class
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
    return df

def check_imbalance(df, target_col='Churn'):
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        print("\nClass Distribution:")
        print(counts)
        print("\nClass Ratio:")
        print(counts / len(df))
