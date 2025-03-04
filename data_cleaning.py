import os
import ast
import pandas as pd
import datetime
import pytz
from tqdm import tqdm 


def resolve_conflicts_on_duplicates(df):
    tqdm.write("Risoluzione conflitti sui duplicati...")
    
    # Step 1: Identify the most frequent server for each address
    most_frequent_server = df.groupby('address')['server'].agg(lambda x: x.mode()[0])
    
    # Step 2: Filter rows to keep only those matching the most frequent server
    df['most_frequent_server'] = df['address'].map(most_frequent_server)
    filtered_df = df[df['server'] == df['most_frequent_server']].drop(columns=['most_frequent_server'])
    
    # Step 3: Deduplicate by selecting one random row per group
    deduplicated_df = filtered_df.groupby(['address', 'date', 'rounded_time'], group_keys=False).apply(
        lambda group: group.sample(1, random_state=42)
    )
    return deduplicated_df


def are_coordinates_valid(pos):
    try:
        parsed = ast.literal_eval(pos) if isinstance(pos, str) else pos
        return all(isinstance(coord, (int, float)) for coord in parsed)
    except (ValueError, SyntaxError, TypeError):
        return False


def is_parcel_valid(parcel_str):
    try:
        # Convert the parcel string to a list
        parcel = ast.literal_eval(parcel_str) if isinstance(parcel_str, str) else parcel_str
        # Check if both elements are within the range [-150, 150]
        return all(-150 <= coord <= 150 for coord in parcel)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list or other errors
        return False


def process_files(directory, timezone='UTC'):
    tqdm.write(f"Processing files in directory: {directory}...")
    all_files = [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.isdigit()]
    
    # Initialize an empty list for efficient concatenation
    data_frames = []

    # Process files in chunks
    for file_path in tqdm(all_files, desc="Processing files", unit="file"):
        temp_data = []
        server = None

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith("https"):
                    server = line.strip()
                elif line.startswith("{'ok': True"):
                    # Process only valid data lines
                    dict_obj = ast.literal_eval(line[:-1])
                    tmp_df = pd.DataFrame(dict_obj['peers'])
                    tmp_df['server'] = server
                    temp_data.append(tmp_df)

        if temp_data:
            df = pd.concat(temp_data, ignore_index=True)
            data_frames.append(df)

    # Combine all DataFrames efficiently
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Convert unhashable list-like columns to tuples for drop_duplicates
    for col in combined_df.columns:
        if combined_df[col].apply(lambda x: isinstance(x, list)).any():
            combined_df[col] = combined_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    
    # Drop duplicates and unnecessary columns early
    combined_df.drop_duplicates(inplace=True)
    combined_df['datetime'] = pd.to_datetime(combined_df['lastPing'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone)
    combined_df['date'] = combined_df['datetime'].dt.date
    combined_df['rounded_time'] = combined_df['datetime'].dt.floor('T')  # Floor to nearest minute

    # Rollover adjustment for midnight edge case
    rollover_mask = (combined_df['rounded_time'].dt.time == datetime.time(0, 0)) & (
        combined_df['datetime'].dt.hour == 23) & (combined_df['datetime'].dt.minute == 59)
    combined_df.loc[rollover_mask, 'date'] += pd.Timedelta(days=1)

    # Filter rows and clean up
    combined_df = combined_df[combined_df['date'] >= datetime.date(2024, 3, 22)]
    combined_df['rounded_time'] = combined_df['rounded_time'].dt.strftime('%H:%M')
    combined_df.drop(columns=['id', 'lastPing', 'datetime'], inplace=True)

    return combined_df

def clean_data(df):
    tqdm.write("Inizio della pulizia dei dati...")

    # Resolve duplicates efficiently
    tqdm.write("Risoluzione dei conflitti sui duplicati...")
    df = resolve_conflicts_on_duplicates(df)
    df.drop(columns=['server'], inplace=True)

    # Calculate unique days and minutes for filtering
    tqdm.write("Calcolo giorni unici e minuti per ID...")
    days_minutes = df.groupby('address').agg(days=('date', 'nunique'), minutes=('rounded_time', 'nunique')).reset_index()

    # Filter by conditions (>5 days, >30 minutes)
    tqdm.write("Filtraggio ID con più di 5 giorni e 30 minuti...")
    selected_ids = days_minutes.query("days > 5 and minutes > 30")['address']
    df = df[df['address'].isin(selected_ids)]

    # Validate coordinates and parcels using vectorized checks
    tqdm.write("Validazione delle coordinate e dei parcel...")
    df = df[df['position'].apply(are_coordinates_valid)]
    df = df[df['parcel'].apply(is_parcel_valid)]

    return df

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
