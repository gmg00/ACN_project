import pandas as pd
import os
'''
folder_path = 'data/dfs'

all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

#concatenazione csv
df_list = [pd.read_csv(file) for file in all_files]
df_combined = pd.concat(df_list, ignore_index=True)
'''

output_csv = 'decentraland.csv'
df_combined.to_csv(output_csv, index=False)

df = pd.read_csv('decentraland.csv')

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

unique_days_per_id = df.groupby('id')['date'].nunique().reset_index()
minutes_per_id = df.groupby('id')['rounded_time'].nunique().reset_index()
days_minutes_per_id = pd.merge(unique_days_per_id, minutes_per_id, on='id')

days_minutes_per_id.columns = ['id', 'days', 'minutes']

selected_ids = days_minutes_per_id[(days_minutes_per_id['days']>4) & (days_minutes_per_id['minutes']>30)]['id'].tolist()

new_df = df[df['id'].isin(selected_ids)]
output_csv = 'cleaned_decentraland.csv'
new_df.to_csv(output_csv, index=False)
