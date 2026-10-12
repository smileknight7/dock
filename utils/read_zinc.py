import pandas as pd
zinc_df = pd.read_csv('/data3/lpw/data/dock_data/zinc.csv')  
print(zinc_df.columns)
zinc_50 = zinc_df.head(50)
zinc_50.to_csv('/data3/lpw/data/dock_data/zinc_50.csv', index=False)