import pandas as pd
import os

def main():
    input_file = r"f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\raw_data\smear2008-18_allmonths.csv"
    output_dir = r"f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\change_data"
    output_file = os.path.join(output_dir, "smear_daily_processed.csv")

    print(f"Reading from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    df['Time'] = pd.to_datetime(df['Time'])
    df.rename(columns={
        'Time': 'date',
        'NEE': 'target'
    }, inplace=True)

    df.set_index('date', inplace=True)
    df_daily = df.resample('D').mean()
    df_daily.reset_index(inplace=True)

    df_daily['date'] = df_daily['date'].dt.strftime('%Y/%m/%d')
    df_daily.interpolate(method='linear', inplace=True)
    df_daily.dropna(inplace=True)
    df_daily = df_daily.round(5)

    cols = list(df_daily.columns)
    cols.remove('date')
    cols.remove('target')
    final_cols = ['date'] + cols + ['target']
    df_daily = df_daily[final_cols]

    os.makedirs(output_dir, exist_ok=True)
    df_daily.to_csv(output_file, index=False)
    print(f"Success! Output saved to: {output_file}")
    
if __name__ == "__main__":
    main()
