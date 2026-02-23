import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(df, target_col='Churn'):
    """
    Train and evaluate Logistic Regression and XGBoost models.
    """
    # Prepare data
    X = df.drop(columns=[target_col, 'customerID', 'Cluster_Label', 'Cluster'], errors='ignore')
    # If Churn is Yes/No, map to 1/0
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }
    
    results = {}
    best_estimators = {}
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_estimators[name] = best_model
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        results[name] = metrics
        print(f"Best Params: {grid.best_params_}")
        print(f"Metrics: {metrics}")
        
        # Model Interpretation
        if name == 'Logistic Regression':
            coeffs = pd.DataFrame({'Feature': X.columns, 'Coefficient': best_model.coef_[0]})
            coeffs['AbsCoeff'] = coeffs['Coefficient'].abs()
            print("\nTop 10 Churn Drivers (Logistic Regression):")
            print(coeffs.sort_values(by='Coefficient', ascending=False).head(10))
            print("\nTop 10 Retention Drivers (Logistic Regression):")
            print(coeffs.sort_values(by='Coefficient', ascending=True).head(10))
            
        elif name == 'XGBoost':
            importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_})
            print("\nTop 10 Churn Predictors (XGBoost):")
            print(importance.sort_values(by='Importance', ascending=False).head(10))
            
    return results, best_estimators, X_test, y_test

