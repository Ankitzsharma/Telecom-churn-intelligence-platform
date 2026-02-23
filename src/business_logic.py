import pandas as pd
import numpy as np

def simulate_financial_impact(df, churn_preds, churn_probs, y_true):
    """
    Calculate financial impact of churn and potential savings.
    """
    print("\n========== FINANCIAL IMPACT SIMULATION ==========")
    
    # Calculate average revenue (MonthlyCharges) for churned customers
    churned_customers = df[y_true == 1]
    avg_revenue_churned = churned_customers['MonthlyCharges'].mean()
    total_churned = len(churned_customers)
    total_revenue_loss = churned_customers['MonthlyCharges'].sum()
    
    print(f"Average Revenue per Churned Customer: ${avg_revenue_churned:.2f}")
    print(f"Total Monthly Revenue Loss due to Churn (in test set): ${total_revenue_loss:.2f}")
    
    # Identify top 20% high-risk customers
    # We use probabilities
    risk_df = pd.DataFrame({
        'ChurnProb': churn_probs,
        'MonthlyCharges': df.loc[y_true.index, 'MonthlyCharges'],
        'ActualChurn': y_true
    })
    
    threshold = risk_df['ChurnProb'].quantile(0.80)
    high_risk_segment = risk_df[risk_df['ChurnProb'] >= threshold]
    
    # Potential Revenue Saved
    # Assuming retention campaign success rate (e.g., 30% or 50%)
    # Let's assume 30% success rate for high risk customers
    retention_success_rate = 0.30
    potential_saved_revenue = high_risk_segment['MonthlyCharges'].sum() * retention_success_rate
    
    print(f"Top 20% High Risk Threshold (Prob): {threshold:.4f}")
    print(f"Number of High Risk Customers Identified: {len(high_risk_segment)}")
    print(f"Potential Revenue Saved (Assuming 30% retention success): ${potential_saved_revenue:.2f}")
    
    # ROI Calculation
    # Assume cost per retention offer is $10 (discount or marketing cost)
    cost_per_offer = 10
    total_campaign_cost = len(high_risk_segment) * cost_per_offer
    roi = (potential_saved_revenue - total_campaign_cost) / total_campaign_cost * 100
    
    print(f"Estimated Campaign Cost: ${total_campaign_cost:.2f}")
    print(f"Estimated ROI of Retention Campaign: {roi:.2f}%")
    print("=================================================")
    
    return {
        'avg_revenue_churned': avg_revenue_churned,
        'total_revenue_loss': total_revenue_loss,
        'potential_saved_revenue': potential_saved_revenue,
        'roi': roi
    }

def generate_recommendations():
    """
    Generate structured business recommendations.
    """
    print("\n========== BUSINESS STRATEGY RECOMMENDATIONS ==========")
    print("1. Targeted Retention Strategy:")
    print("   - Focus on High-Risk, High-Value customers identified by the model.")
    print("   - Offer personalized incentives like contract upgrades or discounts.")
    
    print("\n2. Personalized Offer Strategy:")
    print("   - Use 'EngagementScore' to tailor offers.")
    print("   - Low engagement users might need service usage tutorials or bundle upgrades.")
    
    print("\n3. Pricing Strategy:")
    print("   - Analyze price sensitivity in 'High Revenue At Risk' segment.")
    print("   - Consider loyalty discounts for long-term contracts to reduce churn.")
    
    print("\n4. High-CLV Protection Strategy:")
    print("   - Prioritize customer service for 'High Value Loyal' segment.")
    print("   - Implement a VIP support line.")
    
    print("\n5. Marketing Budget Optimization:")
    print("   - Reallocate acquisition budget to retention for high-risk segments.")
    print("   - Stop targeting 'Likely to Churn Regardless' segment with expensive offers.")
    
    print("\n6. Product Improvement Insights:")
    print("   - Investigate features driving churn (e.g., specific internet service types).")
    print("   - Enhance 'TechSupport' quality if it's a key retention driver.")
    print("=======================================================")
