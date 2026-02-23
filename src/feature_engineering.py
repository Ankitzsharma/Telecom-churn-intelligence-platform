import pandas as pd
import numpy as np

def create_features(df):
    """
    Create advanced business features:
    1. Engagement Score
    2. Recency Proxy
    3. Revenue Intensity
    4. Contract Risk Score
    5. Estimated CLV
    6. Customer Value Segment
    """
    df = df.copy()
    
    # 1. Engagement Score: Count of active services subscribed
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # We need to handle how these are encoded. 
    # If this runs BEFORE encoding (on raw-ish data), it's easier.
    # If AFTER encoding, we need to sum the relevant columns.
    # Let's assume this runs on the CLEANED but NOT FULLY ENCODED data, 
    # or we handle the encoding inside.
    # Actually, let's look at the raw values.
    # 'Yes', 'No', 'No internet service', 'No phone service'
    
    # Helper to count 'Yes'
    def count_services(row):
        count = 0
        for service in services:
            if service in row.index:
                val = row[service]
                if val == 'Yes':
                    count += 1
                # If columns are already encoded as 1/0
                elif isinstance(val, (int, float)) and val == 1:
                    count += 1
        return count

    df['EngagementScore'] = df.apply(count_services, axis=1)
    
    # 2. Recency Proxy: Derived from tenure and contract type
    # Logic: High tenure + Long term contract = High Recency (Loyal)
    # Low tenure + Month-to-month = Low Recency (New/Risk)
    # Let's define a simple score. 
    # But wait, "Recency" usually means "Time since last purchase". 
    # In subscription, it might mean "Time since last renewal" or just "Tenure".
    # User says "Derived from tenure and contract type".
    # Let's normalize tenure and weight by contract.
    contract_weight = df['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})
    # If Contract is already encoded (e.g. dummy variables), we need to reconstruct or check columns.
    # To be safe, let's assume this function receives data BEFORE One-Hot Encoding but AFTER basic cleaning.
    
    if 'Contract' in df.columns and df['Contract'].dtype == 'object':
        df['RecencyProxy'] = df['tenure'] * contract_weight
    else:
        # If contract is encoded, we might skip or approximate.
        # Ideally, feature engineering happens before OHE.
        pass

    # 3. Revenue Intensity: MonthlyCharges / number_of_services
    # Avoid division by zero
    df['RevenueIntensity'] = df['MonthlyCharges'] / (df['EngagementScore'] + 1)
    
    # 4. Contract Risk Score: Short-term contracts weighted higher risk
    if 'Contract' in df.columns:
        df['ContractRiskScore'] = df['Contract'].map({'Month-to-month': 3, 'One year': 2, 'Two year': 1})
    
    # 5. Estimated CLV: tenure * MonthlyCharges
    df['EstimatedCLV'] = df['tenure'] * df['MonthlyCharges']
    
    # 6. Customer Value Segment: Categorize CLV into High / Medium / Low
    # Using quantiles
    q1 = df['EstimatedCLV'].quantile(0.33)
    q2 = df['EstimatedCLV'].quantile(0.66)
    
    def segment_clv(clv):
        if clv <= q1: return 'Low'
        elif clv <= q2: return 'Medium'
        else: return 'High'
        
    df['CustomerValueSegment'] = df['EstimatedCLV'].apply(segment_clv)
    
    return df
