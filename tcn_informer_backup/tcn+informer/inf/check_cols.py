import pandas as pd
df = pd.read_excel(r'f:\Dachuang\Nee_Prediction\tcn_informer_backup\tcn+informer\inf\data\raw_data\SX_cropland.xlsx', nrows=3)
with open('cols.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(df.columns.tolist()))
