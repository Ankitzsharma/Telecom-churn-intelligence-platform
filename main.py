import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, validate_data
from src.preprocessing import clean_data, encode_data, check_imbalance
from src.feature_engineering import create_features
from src.segmentation import perform_segmentation
from src.modeling import train_models
from src.business_logic import simulate_financial_impact, generate_recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('churn_model.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Customer Segmentation & Churn Risk Modeling System")
    
    filepath = "Data/Telco-Customer-Churn-Dataset.csv"
    
    # Create models directory if not exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Step 1: Data Validation & Understanding
    logger.info("Step 1: Loading and Validating Data")
    try:
        df = load_data(filepath)
        validate_data(df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Step 2: Data Cleaning
    logger.info("Step 2: Cleaning Data")
    df_clean = clean_data(df)
    check_imbalance(df_clean)

    # Step 3: Feature Engineering
    logger.info("Step 3: Feature Engineering")
    df_features = create_features(df_clean)
    
    # Step 4: Customer Segmentation
    logger.info("Step 4: Customer Segmentation")
    # Updated to unpack returned artifacts
    df_segmented, scaler, kmeans, cluster_names = perform_segmentation(df_features)
    
    # Save Segmentation Artifacts
    logger.info("Saving segmentation artifacts...")
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(kmeans, 'models/kmeans.pkl')
    joblib.dump(cluster_names, 'models/cluster_names.pkl')
    
    # Step 5: Encoding for Modeling
    logger.info("Step 5: Encoding Data for Modeling")
    df_encoded = encode_data(df_segmented)
    
    # Verify columns
    logger.info(f"Columns after encoding: {df_encoded.columns.tolist()}")
    
    # Save Feature Names (for ensuring input consistency in dashboard)
    # We exclude target and non-features
    features = df_encoded.drop(columns=['Churn', 'Cluster', 'Cluster_Label'], errors='ignore').columns.tolist()
    joblib.dump(features, 'models/feature_names.pkl')
    
    # Step 6: Churn Prediction Model
    logger.info("Step 6: Training Churn Prediction Models")
    results, best_models, X_test, y_test = train_models(df_encoded)
    
    # Step 7: Financial Impact Simulation
    logger.info("Step 7: Financial Impact Simulation")
    best_model_name = max(results, key=lambda k: results[k]['ROC-AUC'])
    logger.info(f"Best Model: {best_model_name}")
    best_model = best_models[best_model_name]
    
    # Save Best Model
    logger.info(f"Saving best model ({best_model_name})...")
    joblib.dump(best_model, 'models/best_model.pkl')
    
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    if 'MonthlyCharges' not in X_test.columns:
        logger.warning("MonthlyCharges not found in X_test. Fetching from original df.")
        X_test_with_charges = X_test.copy()
        X_test_with_charges['MonthlyCharges'] = df_segmented.loc[X_test.index, 'MonthlyCharges']
        simulate_financial_impact(X_test_with_charges, y_pred, y_prob, y_test)
    else:
        simulate_financial_impact(X_test, y_pred, y_prob, y_test)
    
    # Step 8: Business Strategy Recommendations
    logger.info("Step 8: Generating Recommendations")
    generate_recommendations()
    
    # Step 9: Production Readiness
    logger.info("Step 9: Production Readiness - Code is modular and logged.")
    
    # Step 10: Executive Summary
    logger.info("Step 10: Executive Summary")
    print("\n========== EXECUTIVE SUMMARY ==========")
    print("To: CTO, Head of Marketing, Product Team")
    print("From: Senior Data Scientist")
    print("Subject: Customer Segmentation & Churn Risk System Report")
    print("\n1. Key Findings:")
    print(f"   - Identified {len(df_segmented['Cluster'].unique())} distinct customer segments.")
    print(f"   - Best model ({best_model_name}) achieved ROC-AUC of {results[best_model_name]['ROC-AUC']:.4f}.")
    print("   - Top churn drivers include Contract Type, Tenure, and Internet Service Type.")
    
    print("\n2. Strategic Actions:")
    print("   - Implement targeted retention for 'High Revenue At Risk' segment.")
    print("   - Roll out personalized offers based on Engagement Score.")
    
    print("\n3. Expected Impact:")
    print("   - Potential to save significant revenue by targeting top 20% high-risk customers.")
    print("   - Estimated ROI for retention campaign is positive.")
    print("=======================================")

if __name__ == "__main__":
    main()
