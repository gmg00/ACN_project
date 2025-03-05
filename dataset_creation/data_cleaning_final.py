import os
import ast
import pandas as pd
import datetime
import pytz
from tqdm import tqdm

    """ This script processes the raw data files for each day, resolves conflicts, and filters the data based on the given conditions.
        The input directory should contain timestamped files with data for each day.
        The output directory will contain cleaned data files for each day.
        The final output file will contain the combined and filtered data from all days.
        The script processes the data starting from 22nd March 2024 (to change that modify the function process_and_save_daily_batches).
        The filter conditions are: >5 days and >30 minutes of data for each address (to change that modify the function apply_final_filter_and_save).
    """

def resolve_conflicts_on_duplicates(df):
    """
    Resolves conflicts in a DataFrame such that each (rounded_time, address) pair is unique.
    """
    print("Resolving conflicts on duplicates...")

    resolved_rows = []

    # Group by 'address' and 'rounded_time' to process conflicts
    for (address, rounded_time), group in df.groupby(['address', 'rounded_time']):
        if len(group) > 1:  # Conflict exists if group size > 1
            # Step 1: Determine the most frequent server in the group
            server_counts = group['server'].value_counts()
            most_frequent_server = server_counts.idxmax()

            # Step 2: Filter rows to keep only the most frequent server
            filtered_group = group[group['server'] == most_frequent_server]

            # Step 3: Resolve ties by random selection if multiple rows remain
            if len(filtered_group) > 1:
                filtered_group = filtered_group.sample(1, random_state=42)

            resolved_rows.append(filtered_group)
        else:
            # No conflict, keep the single row
            resolved_rows.append(group)

    # Combine resolved rows into a single DataFrame
    resolved_df = pd.concat(resolved_rows, ignore_index=True)

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
        parcel = ast.literal_eval(parcel_str) if isinstance(parcel_str, str) else parcel_str
        # Check if both elements are within the range [-150, 150]
        return all(-150 <= coord <= 150 for coord in parcel)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list or other errors
        return False
    
def process_files_for_day(directory, date, timezone='UTC'):
    print(f"Processing files for {date}...")
    all_files = [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.isdigit()]
    
    # Filter files for the given date
    data_frames = []
    correct_date_found = False
    for file_path in all_files:
        temp_data = []
        server = None
        
        # Extract timestamp from the file name
        timestamp = int(os.path.basename(file_path))
        file_datetime = pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert(timezone)
        file_date = file_datetime.date()

        if file_date != date:
            if correct_date_found:
                break
            continue

        correct_date_found = True
        # Extract rounded time
        file_rounded_time = file_datetime.strftime('%H:%M')

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith("https"):
                    server = line.strip()
                elif line.startswith("{'ok': True"):
                    # Process only valid data lines
                    dict_obj = ast.literal_eval(line[:-1])
                    tmp_df = pd.DataFrame(dict_obj['peers'])
                    tmp_df['server'] = server
                    # Add date and rounded_time directly to the temporary DataFrame
                    tmp_df['date'] = file_date
                    tmp_df['rounded_time'] = file_rounded_time
                    temp_data.append(tmp_df)

        if temp_data:
            df = pd.concat(temp_data, ignore_index=True)
            data_frames.append(df)

    # Combine all DataFrames for the day
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df = combined_df.drop(columns=['id', 'lastPing'])
    else:
        combined_df = pd.DataFrame()  # Empty DataFrame if no data found for the day

    return combined_df

def clean_data(df):

    # Resolve duplicates
    df = resolve_conflicts_on_duplicates(df)
    df.drop(columns=['server'], inplace=True)

    # Validate coordinates and parcels
    df = df[df['position'].apply(are_coordinates_valid)]
    df = df[df['parcel'].apply(is_parcel_valid)]

    return df

def process_and_save_daily_batches(input_directory, timezone='UTC', output_directory='processed_data'):
    # Get the last processed date
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        last_processed_date = None
    else:
        processed_files = sorted([f for f in os.listdir(output_directory) if f.startswith('cleaned_data_')])
        if processed_files:
            last_processed_date = datetime.datetime.strptime(
                processed_files[-1].split('_')[-1].split('.')[0], "%Y-%m-%d"
            ).date()
        else:
            last_processed_date = None

    # Get the end date from the latest timestamp file in the input directory
    all_files = [file for file in os.listdir(input_directory) if file.isdigit()]
    if not all_files:
        print("No timestamp files found in the input directory.")
        return
    
    latest_timestamp = max(int(file) for file in all_files)
    end_date = pd.to_datetime(latest_timestamp, unit='s').tz_localize('UTC').date()

    # Process data from 2024-03-22 onward
    start_date = last_processed_date + datetime.timedelta(days=1) if last_processed_date else datetime.date(2024, 3, 22)

    for current_date in pd.date_range(start_date, end_date):
        current_date = current_date.date()
        day_df = process_files_for_day(input_directory, current_date, timezone)

        if not day_df.empty:
            # Clean data for the day
            cleaned_day_df = clean_data(day_df)

            # Save cleaned DataFrame
            output_path = os.path.join(output_directory, f"cleaned_data_{current_date}.csv")
            cleaned_day_df.to_csv(output_path, index=False)
            print(f"Saved cleaned data for {current_date} to {output_path}")

def load_and_combine_data(output_directory='processed_data'):
    # Load all daily cleaned data
    tqdm.write(f"Loading and combining data from {output_directory}...")
    all_files = [os.path.join(output_directory, file) for file in sorted(os.listdir(output_directory)) if file.endswith('.csv')]
    all_dfs = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    return combined_df

def apply_final_filter_and_save(combined_df, output_file):
    print("Applying final 5-day and 30-minute filter...")

    # Calculate unique days and minutes for filtering
    days_minutes = combined_df.groupby('address').agg(days=('date', 'nunique'), minutes=('rounded_time', 'nunique')).reset_index()

    # Filter by conditions (>5 days, >30 minutes)
    selected_ids = days_minutes.query("days > 5 and minutes > 30")['address']
    filtered_df = combined_df[combined_df['address'].isin(selected_ids)]

    # Save the final cleaned DataFrame
    filtered_df.to_csv(output_file, index=False)
    print(f"Final filtered data saved to {output_file}")

if __name__ == '__main__':
    # Main process
    input_directory = 'data'
    output_directory = 'processed_data'
    final_output_file = 'final_cleaned_df.csv'

    # Process and save daily batches
    process_and_save_daily_batches(input_directory, timezone='UTC', output_directory=output_directory)

    # Load and combine data
    combined_df = load_and_combine_data(output_directory)

    # Apply final filtering and save
    apply_final_filter_and_save(combined_df, final_output_file)