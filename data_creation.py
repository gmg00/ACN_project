import pandas as pd
import os
import datetime
import ast
import pytz

def resolve_conflicts_on_duplicates(df):
    # Identify rows where there are conflicts (more than one server at the same time for a user)
    conflicts = df.groupby(['id', 'date', 'rounded_time']).filter(lambda x: x['server'].nunique() > 1)
    
    # Get the most frequent server for each user (id) overall
    most_frequent_server = df.groupby('id')['server'].agg(lambda x: x.value_counts().idxmax())

    # Apply resolution only to conflicts by filtering to keep rows with the most frequent server for the user
    conflicts_resolved = conflicts[conflicts.apply(lambda row: row['server'] == most_frequent_server[row['id']], axis=1)]
    
    # Combine non-conflicting rows with resolved conflicts
    non_conflicts = df.drop(conflicts.index)
    resolved_df = pd.concat([non_conflicts, conflicts_resolved]).sort_index()
    
    return resolved_df

def create_df_by_date(directory, start_from=None, timezone='UTC'):
    # Set the initial date based on start_from if provided
    current_date = start_from if start_from else None
    df = pd.DataFrame()  # Initialize an empty DataFrame

    for filename in sorted(os.listdir(directory)):  # Sort files to process sequentially
        # Skip hidden files or non-numeric filenames
        if not filename.isdigit():
            continue
            
        file_path = os.path.join(directory, filename)
        file_date = datetime.datetime.fromtimestamp(int(filename)).astimezone(pytz.timezone(timezone)).date()

        # Skip files before the start date
        if current_date and file_date < current_date:
            continue

        # Save and reset the DataFrame if the date changes
        if current_date and file_date != current_date:
            save_df_for_date(df, directory, current_date, timezone)
            df = pd.DataFrame()  # Reset DataFrame for the new day

        # Update current_date for the first time or when the date changes
        current_date = file_date

        # Read and process file
        df = process_file(file_path, df)
    
    # Save the last day's data
    if not df.empty:
        save_df_for_date(df, directory, current_date, timezone)

def process_file(file_path, df):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("https"):
            server = line.strip()
        elif line.startswith("{'ok': True"):
            dict_obj = ast.literal_eval(line[:-1])
            tmp_df = pd.DataFrame(dict_obj['peers'])
            tmp_df['server'] = server
            df = pd.concat([df, tmp_df], ignore_index=True)
    return df

def save_df_for_date(df, directory, date, timezone):
    # Convert timestamps to datetime with the given timezone
    df['datetime'] = pd.to_datetime(df['lastPing'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone)
    df['date'] = df['datetime'].dt.date
    df['rounded_time'] = df['datetime'].dt.round('min').dt.strftime('%H:%M')
    df = resolve_conflicts_on_duplicates(df)

    # Save the DataFrame for the specified date
    output_path = os.path.join(directory, 'dfs', f'{str(date)}.csv')
    df.to_csv(output_path, index=False)
    print(f'Saved data for date: {date}')

create_df_by_date('data')
