import os
import ast
import pandas as pd
import datetime
import pytz
from tqdm import tqdm 


def resolve_conflicts_on_duplicates(df):
    tqdm.write("Risoluzione conflitti sui duplicati...")
    conflicts = df.groupby(['address', 'date', 'rounded_time']).filter(lambda x: x['server'].nunique() > 1)
    most_frequent_server = df.groupby('address')['server'].agg(lambda x: x.value_counts().idxmax())
    conflicts_resolved = conflicts[conflicts.apply(lambda row: row['server'] == most_frequent_server[row['address']], axis=1)]
    non_conflicts = df.drop(conflicts.index)
    resolved_df = pd.concat([non_conflicts, conflicts_resolved]).sort_index()
    return resolved_df


def are_coordinates_valid(pos):
    try:
        parsed = ast.literal_eval(pos) if isinstance(pos, str) else pos
        return all(isinstance(coord, (int, float)) for coord in parsed)
    except (ValueError, SyntaxError, TypeError):
        return False


def is_parcel_valid(parcel_str):
    try:
        # Convert the parcel string to a list
        parcel = ast.literal_eval(parcel_str)
        # Check if both elements are within the range [-150, 150]
        return all(-150 <= coord <= 150 for coord in parcel)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list or other errors
        return False


def process_files(directory, timezone='UTC'):
    tqdm.write(f"Processing files in directory: {directory}...")
    all_files = [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.isdigit()]
    data_frames = []

    # Read and process each file
    for file_path in tqdm(all_files, desc="Processing files", unit="file"):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        server = None
        temp_data = []
        for line in lines:
            if line.startswith("https"):
                server = line.strip()
            elif line.startswith("{'ok': True"):
                dict_obj = ast.literal_eval(line[:-1])
                tmp_df = pd.DataFrame(dict_obj['peers'])
                tmp_df['server'] = server
                temp_data.append(tmp_df)
        
        if temp_data:
            df = pd.concat(temp_data, ignore_index=True)
            data_frames.append(df)

    tqdm.write("Combining all DataFrames...")
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)

    tqdm.write("Adding useful columns for cleaning...")
    combined_df['datetime'] = pd.to_datetime(combined_df['lastPing'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone)
    combined_df['date'] = combined_df['datetime'].dt.date
    combined_df['rounded_time'] = combined_df['datetime'].dt.round('min')

    # Adjust for times close to midnight (23:59:50 -> 00:00:00 of next day)
    rollover_mask = (combined_df['rounded_time'].dt.time == datetime.time(0, 0)) & (
        combined_df['datetime'].dt.hour == 23) & (combined_df['datetime'].dt.minute == 59)

    combined_df.loc[rollover_mask, 'date'] += pd.Timedelta(days=1)
    combined_df['rounded_time'] = combined_df['rounded_time'].dt.strftime('%H:%M')

    # Remove data for dates before 2024-03-22
    combined_df = combined_df[combined_df['date'] >= datetime.date(2024, 3, 22)]
    combined_df.drop(columns=['id', 'lastPing', 'datetime'], inplace=True)

    return combined_df


def clean_data(df):
    tqdm.write("Inizio della pulizia dei dati...")

    # Risolvi conflitti sui duplicati
    tqdm.write("Risoluzione dei conflitti sui duplicati...")
    df = resolve_conflicts_on_duplicates(df)
    df.drop(columns=['server'], inplace=True)

    # Calcola giorni unici e minuti per ogni id
    tqdm.write("Calcolo giorni unici e minuti per ID...")
    unique_days_per_id = df.groupby('address')['date'].nunique().reset_index()
    minutes_per_id = df.groupby('address')['rounded_time'].nunique().reset_index()
    days_minutes_per_id = pd.merge(unique_days_per_id, minutes_per_id, on='address')
    days_minutes_per_id.columns = ['address', 'days', 'minutes']

    # Filtra per id con >5 giorni e >30 minuti
    tqdm.write("Filtraggio ID con più di 5 giorni e 30 minuti...")
    selected_ids = days_minutes_per_id[(days_minutes_per_id['days'] > 5) & (days_minutes_per_id['minutes'] > 30)]['address'].tolist()
    filtered_df = df[df['address'].isin(selected_ids)]

    # Rimuovi righe con coordinate non valide
    tqdm.write("Validazione delle coordinate...")
    invalid_coordinates = filtered_df[~filtered_df['position'].apply(are_coordinates_valid)]
    tqdm.write(f"Numero di liste con coordinate non valide: {len(invalid_coordinates)}")
    cleaned_df = filtered_df.drop(invalid_coordinates.index)
    cleaned_df = cleaned_df[cleaned_df['parcel'].apply(is_parcel_valid)]

    return cleaned_df

input_directory = 'data'
output_file = 'clean_df.csv'
timezone = 'UTC'

# Leggi e combina i dati
tqdm.write("Processamento dei file e combinazione dei dati...")
combined_df = process_files(input_directory, timezone)

combined_df.to_csv('combined_df.csv', index=False)

# Pulisci i dati
tqdm.write("Pulizia dei dati...")
cleaned_df = clean_data(combined_df)

# Salva il DataFrame pulito
tqdm.write(f"Salvataggio dei dati puliti in: {output_file}...")
cleaned_df.to_csv(output_file, index=False)
tqdm.write(f"Dati puliti salvati in: {output_file}")
