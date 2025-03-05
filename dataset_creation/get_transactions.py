import os
import time
import json
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from datetime import datetime

    """ This script fetches all events for a list of Ethereum addresses from the OpenSea API and saves them to individual JSON files. It includes rate-limiting, retries, and concurrency control to handle large datasets.
    """

# Constants
MAX_RETRIES = 5
DELAY = 0.1 # Base delay between requests
MAX_WORKERS = 2
MAX_CONCURRENT_REQUESTS = 2  # Number of requests allowed simultaneously
EVENT_TYPES = ['transfer', 'sale']  # List of event types to fetch

# Shared semaphore for rate-limiting
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)


def log_message(message):
    """Log standardized messages with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def fetch_all_events(address, headers):
    """Fetch all events for a single address across all event types."""
    all_events = []
    success = True  # Track if all event types were fetched successfully

    for event_type in EVENT_TYPES:
        #log_message(f"Fetching {event_type} events for {address}...")
        events = fetch_events(address, event_type, headers)

        if events is None:  # Check if an error occurred
            log_message(f"Failed to fetch {event_type} events for {address}. Skipping save.")
            success = False
            break  # Stop fetching further event types if one fails

        # Log when no events are found but there's no error
        if not events:
            log_message(f"No {event_type} events found for {address}.")

        all_events.extend(events)

    return all_events if success else None



def fetch_events(address, event_type, headers, retries=MAX_RETRIES):
    """Fetch events for a specific address and event type."""
    base_url = f"https://api.opensea.io/api/v2/events/accounts/{address}"
    events = []
    cursor = None
    retry_count = 0

    while True:
        params = {
            "limit": 50,
            "before": 1728950400,  # Replace with actual timestamp if needed
            "after": 1580515200,   # Replace with actual timestamp if needed
            "event_type": event_type,
        }
        if cursor:
            retry_count = 0
            #print(f'Next page ({address})')
            params["next"] = cursor

        try:
            with semaphore:  # Limit concurrent requests
                response = requests.get(base_url, headers=headers, params=params, timeout=30)
                time.sleep(DELAY)  # Throttle each request

            if response.status_code == 200:
                data = response.json()
                events.extend(data.get("asset_events", []))
                cursor = data.get("next")

                if not cursor:  # No more pages
                    break
            elif response.status_code == 429:
                log_message(f"Rate limit reached for {address} ({event_type}). Retrying after delay...")
                time.sleep(10)  # Backoff for rate-limiting
            else:
                log_message(f"Error {response.status_code} for {address} ({event_type}): {response.text}")
                return None  # Return None on error
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            #log_message(f'requests.exceptions.ConnectionError = {requests.exceptions.ConnectionError}, requests.exceptions.Timeout = {requests.exceptions.Timeout}, retry_count = {retry_count+1}')
            retry_count += 1
            if retry_count > retries:
                log_message(f"Max retries exceeded for {address} ({event_type}).")
                return None  # Return None if max retries are exceeded
            time.sleep(2 ** retry_count)  # Exponential backoff

    return events  # Return an empty list if no events found

def save_events_to_file(address, events, output_dir):
    """Save events to a JSON file."""
    output_path = os.path.join(output_dir, f"{address}.json")
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
    log_message(f"Saved events for {address} to {output_path}")

def check_account_existence(address, headers, retries=MAX_RETRIES):
    """
    Check if an account exists on OpenSea, with retry logic for connection errors or timeouts.

    Args:
        address (str): The address or username to check.
        headers (dict): API headers.
        retries (int): Maximum number of retries for connection issues.

    Returns:
        bool: True if the account exists, False otherwise.
    """
    url = f"https://api.opensea.io/api/v2/accounts/{address}"
    retry_count = 0

    while retry_count <= retries:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True
            elif response.status_code == 400:
                #log_message(f"Account {address} does not exist.")
                return False
            else:
                log_message(f"Unexpected status code {response.status_code} while checking account {address}.")
                return False
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retry_count += 1
            log_message(f"Error checking account {address} (attempt {retry_count}/{retries}): {e}")
            if retry_count > retries:
                log_message(f"Max retries exceeded for account check: {address}. Skipping existence check.")
                return True  # Assume account exists to skip the existence check
            time.sleep(2 ** retry_count)  # Exponential backoff

    return False



def save_non_existent_accounts(address, output_dir):
    """
    Save non-existent accounts to a file for later reference.

    Args:
        address (str): The address to save.
        output_dir (str): Directory to save the file.
    """
    file_path = os.path.join(output_dir, "non_existent_accounts.json")
    try:
        # Load existing data if file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                non_existent_accounts = json.load(f)
        else:
            non_existent_accounts = []

        # Add the new non-existent account
        if address not in non_existent_accounts:
            non_existent_accounts.append(address)

        # Save back to the file
        with open(file_path, "w") as f:
            json.dump(non_existent_accounts, f, indent=2)
        log_message(f"Saved non-existent account {address} to {file_path}")
    except Exception as e:
        log_message(f"Error saving non-existent account {address}: {e}")


def fetch_all_events_with_retry(addresses, api_key, output_dir, max_workers=MAX_WORKERS):
    """
    Fetch events for multiple addresses, checking account existence first,
    and save the events immediately after fetching.

    Args:
        addresses (list of str): List of addresses to process.
        api_key (str): OpenSea API key.
        output_dir (str): Directory to save the events.
        max_workers (int): Maximum concurrent workers.
    """
    headers = {"X-API-KEY": api_key, "accept": "application/json"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for address in addresses:
            # Check account existence before submitting the fetch task
            if check_account_existence(address, headers):
                # Submit the task to fetch events
                futures[executor.submit(fetch_and_save_events, address, headers, output_dir)] = address
            else:
                # Save non-existent accounts to a file
                save_non_existent_accounts(address, output_dir)

        # Wait for all tasks to complete
        for future in as_completed(futures):
            address = futures[future]
            try:
                # Check for any exceptions that occurred during fetching
                future.result()
                #log_message(f"Processing for {address} completed successfully.")
            except Exception as e:
                log_message(f"Error processing {address}: {e}")


def fetch_and_save_events(address, headers, output_dir):
    """
    Fetch all events for an address and save them immediately after fetching.

    Args:
        address (str): Address to fetch events for.
        headers (dict): API headers.
        output_dir (str): Directory to save events.
    """
    try:
        # Fetch all events for the address
        events = fetch_all_events(address, headers)
        
        if events is not None:  # Only save if all event types were fetched successfully
            save_events_to_file(address, events, output_dir)
            log_message(f"Successfully fetched and saved events for {address}: {len(events)} events")
        else:
            log_message(f"Skipping save for {address} due to incomplete data (errors occurred).")
    except Exception as e:
        log_message(f"Error fetching or saving events for {address}: {e}")

if __name__ == "__main__":
    # Configurations
    dir_name = ''  # Directory containing data
    addresses_file_name = ''  # CSV file with addresses
    output_dir_name = f'{dir_name}transactions/'  # Output directory
    api_key = ""  # API key (replace with your own)

    # Prepare output directory
    os.makedirs(output_dir_name, exist_ok=True)

    # Load already processed addresses

    done_addresses = {filename[:-5] for filename in sorted(os.listdir(output_dir_name))}

    addresses_list = np.load(f'{dir_name}unique_addresses.npy', allow_pickle=True)
    
    to_do = list(set(addresses_list) - set(done_addresses))
    log_message(f"Addresses to process: {len(to_do)}")

    # Fetch events for remaining addresses
    fetch_all_events_with_retry(to_do, api_key, output_dir_name)
