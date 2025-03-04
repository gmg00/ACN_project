import csv
import requests
import json
import numpy as np
from tqdm import tqdm 

### PRENDO TUTTE RISPOSTE API OPENSEA E METTO IN JSON ###
API_KEY = "___"
BASE_URL = "https://api.opensea.io/api/v2/accounts/"

CSV_FILE = "addresses_with_errors.csv"
OUTPUT_FILE = "check_addresses_responses.json"

def read_addresses_from_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        addresses = [row[0] for row in csv_reader]  
    return addresses

def check_addresses(addresses):
    responses = {}
    # Usando tqdm per visualizzare una barra di progresso
    for address in tqdm(addresses, desc="Verificando indirizzi", unit="indirizzo"):
        url = f"{BASE_URL}{address}"
        headers = {
            "accept": "application/json",
            "x-api-key": API_KEY
        }
        try:
            response = requests.get(url, headers=headers)
            responses[address] = response.json()  # Salva la risposta JSON
        except Exception as e:
            responses[address] = {"error": str(e)}  # Gestisce errori
    return responses

def save_responses_to_json(responses, file_path):
    with open(file_path, mode='w') as file:
        json.dump(responses, file, indent=4)

def main():
    print("Leggo gli indirizzi dal file CSV...")
    addresses = read_addresses_from_csv(CSV_FILE)
    print(f"Trovati {len(addresses)} indirizzi.")

    print("Interrogo l'API di OpenSea...")
    responses = check_addresses(addresses)
    print("Richieste completate.")

    print(f"Salvo le risposte in {OUTPUT_FILE}...")
    save_responses_to_json(responses, OUTPUT_FILE)
    print("Operazione completata.")

if __name__ == "__main__":
    main()

### CHECK JSON PER RISPOSTA API DI ADDRESS NON ESISTENTE ###
INPUT_FILE = "addresses_responses.json"
OUTPUT_FILE = "final_nonexistent.csv"

def extract_addresses_with_errors_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    addresses_with_errors = [
        address for address, details in data.items()
        if "errors" in details and "not found" in details["errors"][0].lower()
    ]

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for address in addresses_with_errors:
            writer.writerow([address])  

    print(f"Trovati {len(addresses_with_errors)} indirizzi con errori.")
    print(f"Lista salvata in: {output_file}")

extract_addresses_with_errors_to_csv(INPUT_FILE, OUTPUT_FILE)

### EVENTUALE CONFRONTO CON ALTRI RISULTATI ###
def load_addresses_from_responses_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return set(data.keys())  # Le chiavi del dizionario sono gli indirizzi

# Funzione per caricare indirizzi dal file "non_existent_accounts.npy"
def load_addresses_from_npy(filename):
    data = np.load(filename, allow_pickle=True)  # Carica il file .npy
    return set(data)  # Converte l'array NumPy in un set

# Carica gli indirizzi dai due file
inesistenti_addresses = load_addresses_from_responses_json('check_addresses_responses.json')
non_existing_accounts_addresses = load_addresses_from_npy('non_existent_addresses.npy')

# Calcola i confronti
common_addresses = inesistenti_addresses & non_existing_accounts_addresses  # Intersezione
unique_to_inesistenti = inesistenti_addresses - non_existing_accounts_addresses  # Solo in "check_addresses_responses.json"
unique_to_non_existing = non_existing_accounts_addresses - inesistenti_addresses  # Solo in "non_existent_accounts.npy"
different_addresses = inesistenti_addresses ^ non_existing_accounts_addresses  # Simmetrica (diversi)

# Stampa i risultati
print(f"\n--- RISULTATI ---")
print(f"Numero di indirizzi comuni: {len(common_addresses)}")
print(f"Numero di indirizzi unici in 'check_addresses_responses.json': {len(unique_to_inesistenti)}")
print(f"Numero di indirizzi unici in 'non_existent_accounts.npy': {len(unique_to_non_existing)}")
print(f"Numero di indirizzi diversi in totale: {len(different_addresses)}\n")

# Stampa alcune liste
print("--- Indirizzi Comuni ---")
for address in list(common_addresses)[:10]:  # Mostra i primi 10 indirizzi comuni
    print(address)
if len(common_addresses) > 10:
    print("...")

print("\n--- Indirizzi Unici a 'check_addresses_responses.json' ---")
for address in list(unique_to_inesistenti)[:10]:  # Mostra i primi 10 indirizzi unici
    print(address)
if len(unique_to_inesistenti) > 10:
    print("...")

print("\n--- Indirizzi Unici a 'non_existent_accounts.npy' ---")
for address in list(unique_to_non_existing)[:10]:  # Mostra i primi 10 indirizzi unici
    print(address)
if len(unique_to_non_existing) > 10:
    print("...")

print("\n--- Fine Risultati ---")
