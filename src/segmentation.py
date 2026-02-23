import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def perform_segmentation(df, n_clusters=4):
    """
    Perform KMeans clustering for customer segmentation.
    Returns: df, scaler, kmeans, cluster_names
    """
    # Select numerical features for clustering
    features_for_clustering = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'EngagementScore', 'RevenueIntensity', 'EstimatedCLV'
    ]
    
    # Check if these exist
    available_features = [col for col in features_for_clustering if col in df.columns]
    
    X = df[available_features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    # Analyze clusters to assign names
    # We calculate mean of key features for each cluster
    cluster_summary = df.groupby('Cluster')[available_features].mean()
    
    # Add churn rate if available (just for profiling, not for clustering)
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
             churn_numeric = df['Churn'].map({'Yes': 1, 'No': 0})
             cluster_summary['ChurnRate'] = df.groupby('Cluster').apply(lambda x: (x['Churn'] == 'Yes').mean() if x['Churn'].dtype=='object' else x['Churn'].mean())
        else:
             cluster_summary['ChurnRate'] = df.groupby('Cluster')['Churn'].mean()
             
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Logic to map clusters to names:
    cluster_names = {}
    
    # Sort by Tenure to find "New Customers" (Lowest Tenure)
    new_cust_cluster = cluster_summary['tenure'].idxmin()
    
    # Sort by CLV to find "High Value"
    high_val_cluster = cluster_summary['EstimatedCLV'].idxmax()
    
    # Exclude already assigned
    remaining = set(cluster_summary.index) - {new_cust_cluster, high_val_cluster}
    
    # Of the remaining, finding "Low Engagement" (Lowest EngagementScore or MonthlyCharges)
    if remaining:
        low_eng_cluster = cluster_summary.loc[list(remaining), 'EngagementScore'].idxmin()
        remaining.remove(low_eng_cluster)
    else:
        low_eng_cluster = None
        
    # The last one is "High Revenue At Risk"
    if remaining:
        risk_cluster = list(remaining)[0]
    else:
        risk_cluster = None
        
    # Assign names
    cluster_names[new_cust_cluster] = "New Customers"
    cluster_names[high_val_cluster] = "High Value Loyal" 
    if low_eng_cluster is not None:
        cluster_names[low_eng_cluster] = "Low Engagement"
    if risk_cluster is not None:
        cluster_names[risk_cluster] = "High Revenue At Risk"
        
    df['Cluster_Label'] = df['Cluster'].map(cluster_names)
    
    print("\nCluster Labels Assignment:")
    print(cluster_names)
    
    print("\nCluster Distribution:")
    print(df['Cluster_Label'].value_counts())
    
    # Return artifacts for production
    return df, scaler, kmeans, cluster_names
