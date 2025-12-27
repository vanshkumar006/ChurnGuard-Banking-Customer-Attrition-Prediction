import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Modeling
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    classification_report, average_precision_score
)
from imblearn.over_sampling import SMOTE

# Configuration
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)

FILE_PATH = "BankChurners.csv"

def load_and_clean_data(filepath):
    """
    Loads the dataset and removes unnecessary columns (IDs and data leakage).
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: '{filepath}' not found in the current directory.")
        sys.exit(1)

    print("Step 1: Ingesting and cleaning data...")
    df = pd.read_csv(filepath)
    
    # Identify columns to drop
    # 1. CLIENTNUM is an ID, not a feature
    # 2. Columns containing 'Naive_Bayes' are artefacts from the original dataset source (Data Leakage)
    drop_cols = ['CLIENTNUM'] + [col for col in df.columns if 'Naive_Bayes' in col]
    df = df.drop(columns=drop_cols)
    
    print(f"   -> Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def preprocess_data(df):
    """
    Handles encoding, splitting, and scaling.
    """
    print("Step 2: Preprocessing and Feature Engineering...")
    
    data = df.copy()
    
    # 1. Encode Target (Existing=0, Attrited=1)
    data['Attrition_Flag'] = data['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
    
    # 2. Split X and y
    X = data.drop("Attrition_Flag", axis=1)
    y = data["Attrition_Flag"]
    
    # 3. One-Hot Encoding for categorical variables
    # dtype=int ensures we get 0/1 instead of False/True
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    
    # 4. Check Imbalance
    churn_rate = (y.sum() / len(y)) * 100
    print(f"   -> Churn Rate: {churn_rate:.2f}% (Imbalanced Dataset)")
    
    # 5. Split Data (Stratified to maintain churn rate in test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 6. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_models(X_train, y_train):
    """
    Trains Logistic Regression (SMOTE) and XGBoost (Weighted).
    """
    print("Step 3: Training models...")
    
    # --- Model A: Logistic Regression w/ SMOTE ---
    print("   -> Training Logistic Regression (with SMOTE)...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_res, y_train_res)
    
    # --- Model B: XGBoost w/ Class Weights ---
    print("   -> Training XGBoost (Weighted)...")
    # Calculate scale_pos_weight: sum(negative instances) / sum(positive instances)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=pos_weight, # Handles imbalance
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    return lr_model, xgb_model

def evaluate_and_visualize(models, X_test, y_test, feature_names):
    """
    Evaluates models, prints metrics, and plots confusion matrices + feature importance.
    """
    print("Step 4: Evaluation...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    results = {}
    
    for i, (name, model) in enumerate(models.items()):
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc = roc_auc_score(y_test, y_proba)
        pr = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n📊 {name} Results:")
        print(f"   ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1 Score: {f1:.4f}")
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted (0=Stay, 1=Churn)')
        axes[i].set_ylabel('Actual')
        
        results[name] = model

    plt.tight_layout()
    plt.show()
    
    # --- Feature Importance (XGBoost only) ---
    print("\nStep 5: Extracting Insights...")
    xgb_model = results['XGBoost']
    
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Predictors of Customer Churn', fontsize=14)
    plt.barh(range(10), importances[indices], align='center', color='#4c72b0')
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis() # Highest importance on top
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load
    df = load_and_clean_data(FILE_PATH)
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test, feat_names = preprocess_data(df)
    
    # 3. Train
    lr, xgb = train_models(X_train, y_train)
    
    # 4. Evaluate
    model_dict = {
        "Logistic Regression": lr,
        "XGBoost": xgb
    }
    evaluate_and_visualize(model_dict, X_test, y_test, feat_names)
    
    print("\n✅ Analysis Complete.")