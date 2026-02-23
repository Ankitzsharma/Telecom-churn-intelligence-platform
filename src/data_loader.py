import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Load dataset from the specified filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def validate_data(df):
    """
    Perform data validation steps as requested.
    """
    print("========== DATA VALIDATION ==========")
    print(f"Shape: {df.shape}")
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values Summary:")
    print(df.isnull().sum())
    
    if 'Churn' in df.columns:
        print("\nUnique values in Churn:")
        print(df['Churn'].unique())
        print("\nChurn Distribution:")
        print(df['Churn'].value_counts(normalize=True))
    else:
        print("\nWARNING: 'Churn' column not found!")

    # Check TotalCharges
    if 'TotalCharges' in df.columns:
        # Check if it's numeric
        is_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce').notnull().all()
        print(f"\nIs 'TotalCharges' fully numeric? {is_numeric}")
        
        # Check for empty strings or spaces which might be hiding non-numeric values
        non_numeric_count = pd.to_numeric(df['TotalCharges'], errors='coerce').isnull().sum()
        print(f"Number of non-numeric 'TotalCharges' values: {non_numeric_count}")
    else:
        print("\nWARNING: 'TotalCharges' column not found!")

    print("\nSummary Statistics:")
    print(df.describe(include='all'))
    print("=====================================")

if __name__ == "__main__":
    filepath = "Data/Telco-Customer-Churn-Dataset.csv"
    try:
        df = load_data(filepath)
        validate_data(df)
    except Exception as e:
        print(f"Error: {e}")
