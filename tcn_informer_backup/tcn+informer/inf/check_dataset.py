import pandas as pd

try:
    file_path = r'f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\raw_data\SX_cropland.xlsx'
    df = pd.read_excel(file_path, nrows=5)
    print("----- COLUMNS -----")
    print(df.columns.tolist())
    print("\n----- HEAD -----")
    print(df.head().to_string())
except Exception as e:
    print("ERROR:")
    print(e)
