import pandas as pd
import json
import numpy as np
import os
import time
import datetime
from collections import defaultdict
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm  # Aggiungi tqdm per la barra di progresso


# Define a semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 4  # Adjust based on API rate limits
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

def fetch_events_for_address(address, headers, output_dir, max_retries=5):
    t0 = time.time()
    url = f"https://api.opensea.io/api/v2/events/accounts/{address}"
    events = []
    cursor = None
    retries = 0

    while True:
        params = {"limit": 50, "before": 1728950400, "after": 1580515200}
        if cursor:
            params["next"] = cursor

        try:
            with semaphore:  # Ensure only MAX_CONCURRENT_REQUESTS are active
                response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                events.extend(data.get("asset_events", []))
                cursor = data.get("next")

                if not cursor:  # No more pages, break out
                    break
            elif response.status_code == 429:
                time.sleep(10)  # Backoff for rate-limiting
            else:
                break

        except requests.exceptions.ConnectionError:
            retries += 1
            if retries > max_retries:
                break
            time.sleep(2 ** retries)  # Exponential backoff
        except requests.exceptions.Timeout:
            retries += 1
            if retries > max_retries:
                break
            time.sleep(2 ** retries)  # Exponential backoff

    # Save events to a JSON file for the current address
    output_path = os.path.join(output_dir, f"{address}.json")
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    return address, len(events)  # Return address and number of events


def fetch_all_events_with_workers(addresses, api_key, output_dir, max_workers=5, delay=1):
    headers = {"X-API-KEY": api_key, "accept": "application/json"}
    
    # Usa tqdm per la barra di progresso
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(addresses), desc="Fetching Events") as pbar:
        futures = [executor.submit(fetch_events_for_address, address, headers, output_dir) for address in addresses]
        
        for future in as_completed(futures):
            try:
                address, num_events = future.result()
                pbar.update(1)  # Incrementa la barra di progresso
            except Exception as e:
                print(f"Error processing address: {e}")  # Mantieni il log solo per errori
            
            time.sleep(delay)  # Optional: Delay tra i completamenti (puoi rimuoverlo)


### CAMBIA CSV, CAMBIA API KEY E CREA CARTELLA TRANSACTION !!!! ###
if __name__ == "__main__":

    dir_name = '' # Cartella dove tenete i dati
    addresses_file_name = 'tuofile.csv' # Nome del vostro file
    output_dir_name = f'{dir_name}transactions/' # Create una cartella che si chiama transactions
    api_key = "APIkey" # Mettete la vostra API key

    with open(f'{dir_name}{addresses_file_name}', 'r') as file:
        reader = csv.reader(file)
        addresses_list = [row[0] for row in reader]  # Aggiungi ogni indirizzo alla lista


### USO FUNZIONE ###
fetch_all_events_with_workers(addresses_list, api_key, output_dir_name, max_workers=10, delay=2)