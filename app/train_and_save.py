import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def main():
    print("Starting model training and serialization script...")
    
    # Load dataset
    data_path = os.path.join("data", "UCI_Credit_Card.csv")
    df = pd.read_csv(data_path)
    
    # Clean data (mirroring notebook cleaning exactly)
    df_clean = df.copy()
    
    # ID column is not a feature
    if 'ID' in df_clean.columns:
        df_clean = df_clean.set_index('ID')
        
    df_clean['EDUCATION'] = df_clean['EDUCATION'].replace([0, 5, 6], 4)
    df_clean['MARRIAGE'] = df_clean['MARRIAGE'].replace([0], 3)
    
    # Feature Engineering (mirroring notebook feature engineering exactly)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    pay_status_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    
    # 1. Average Credit Utilisation Rate
    util_rates = []
    for i in range(1, 7):
        bill = df_clean[f'BILL_AMT{i}']
        limit = df_clean['LIMIT_BAL']
        # Utilisation = bill / limit (set negative bill to 0 for utilisation calculation)
        util = np.where(bill < 0, 0, bill) / limit
        util_rates.append(util)
    df_clean['AVG_UTIL_RATE'] = np.mean(util_rates, axis=0)
    
    # 2. Average Payment Ratio
    pay_ratios = []
    for i in range(1, 7):
        bill = df_clean[f'BILL_AMT{i}']
        pay = df_clean[f'PAY_AMT{i}']
        # ratio = pay / bill if bill > 0, else NaN
        ratio = np.where(bill > 0, pay / bill, np.nan)
        pay_ratios.append(ratio)
    # Ignore NaN values when calculating average across columns
    df_clean['AVG_PAY_RATIO'] = np.nanmean(pay_ratios, axis=0)
    # Replace remaining NaNs (for clients with zero/negative bills all 6 months) with 0
    df_clean['AVG_PAY_RATIO'] = df_clean['AVG_PAY_RATIO'].fillna(0)
    
    # 3. Total Delinquency Duration (months delayed)
    delay_months = []
    for col in pay_status_cols:
        # A payment is delayed if status > 0
        delay = np.where(df_clean[col] > 0, 1, 0)
        delay_months.append(delay)
    df_clean['TOTAL_DELAY_MONTHS'] = np.sum(delay_months, axis=0)
    
    # Ensure target name is consistent
    if 'default.payment.next.month' in df_clean.columns:
        df_clean = df_clean.rename(columns={'default.payment.next.month': 'default'})
        
    # Drop AGE_BAND if exists
    if 'AGE_BAND' in df_clean.columns:
        df_clean = df_clean.drop('AGE_BAND', axis=1)
        
    # One-hot encoding
    df_encoded = pd.get_dummies(df_clean, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
    
    # Convert boolean to int
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Features and Target
    X = df_encoded.drop('default', axis=1)
    y = df_encoded['default']
    
    # Train / Test split (using random state 42 like in notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    
    # Train XGBoost Champion Model
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    spw = neg_count / pos_count
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    print("Model trained successfully!")
    
    # Ensure output dir exists
    output_dir = "app"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model, scaler, and columns
    with open(os.path.join(output_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
        
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)
        
    print(f"Saved artifacts to {output_dir}/ directory!")
    print(f"Feature count: {len(X.columns)}")
    print("Features list:", list(X.columns))

if __name__ == "__main__":
    main()
