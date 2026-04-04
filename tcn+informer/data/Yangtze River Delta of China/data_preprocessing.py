import pandas as pd
import numpy as np
import os

def process_yangtze_dataset(file_path):
    print(f"Loading {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    # Ensure date is parsed and sorted
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        # Set date as index for time-based interpolation
        df.set_index('date', inplace=True)
    
    print("Pre-cleaning missing counts:")
    print(df.isnull().sum())
    
    cols = df.columns
    # The Yangtze River Delta data contains zeroes in K↓ (radiation) and VPD, which are physically valid.
    # Target (NEE) and Tair have standard NaNs for missing data. Thus, we only interpolate NaNs.
    
    # Step 1: Linear Interpolation for short gaps (Limit = 6, i.e., 3 hours for 30-min data)
    df[cols] = df[cols].interpolate(method='linear', limit=6)
    
    # Step 2: Time-based Interpolation for medium gaps (Limit = 48, i.e., 24 hours)
    df[cols] = df[cols].interpolate(method='time', limit=48)
    
    # Step 3: Forward/Backward fill as ultimate fallback for long persistent gaps (like sensor downtime)
    df[cols] = df[cols].ffill().bfill()
    
    print("Post-cleaning missing counts:")
    print(df.isnull().sum())
    
    # Restore date column
    df.reset_index(inplace=True)
    
    # Save output 
    out_name = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(out_name, index=False)
    print(f"Saved cleaned data to: {os.path.basename(out_name)}\n")

if __name__ == "__main__":
    base_dir = r"f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\Yangtze River Delta of China"
    dt_path = os.path.join(base_dir, "DT_NEE_建模最终数据(2014年12月1日–2017年11月30日).csv")
    sx_path = os.path.join(base_dir, "SX_NEE_建模最终数据(2015年7月15日–2019年4月24日).csv")
    
    process_yangtze_dataset(dt_path)
    process_yangtze_dataset(sx_path)
