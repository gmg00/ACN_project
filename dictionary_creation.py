import csv
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm  
import json


### CREAZIONE DIZIONARIO ###
data_dict = defaultdict(lambda: defaultdict(lambda: {"users": [], "user_count": 0}))

def extract_datetime(datetime_str):
    return datetime_str[:16]  # Primo 16 caratteri (YYYY-MM-DD HH:MM)

with open('carlo_cleaned_df.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in tqdm(reader, desc="Elaborando righe", unit="riga"):
        # Unisci la data e l'ora arrotondata
        datetime = row['date'] + " " + row['rounded_time']
        
        # Estrai solo la parte utile della data e ora
        datetime = extract_datetime(datetime)
        
        # Estrai gli altri valori necessari
        parcel = ','.join(map(str, eval(row['parcel'])))  # Trasforma la lista in stringa
        position = eval(row['position'])  # Converte la stringa in una lista
        user_id = row['address']  # Uso address come nome utente
        
        # Aggiunta utente al dizionario 
        if not any(user["id"] == user_id for user in data_dict[datetime][parcel]["users"]):
            data_dict[datetime][parcel]["users"].append({
                "id": user_id,
                "position": position
            })
            data_dict[datetime][parcel]["user_count"] += 1


### CREAZIONE FILE JSON ###
output_file_path = 'data_dict.json'

# Calcola il numero totale di timestamp da scrivere
total_items = len(data_dict)

# Iniziamo scrivendo l'apertura del file JSON
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write("{\n")  

    first_item = True  # Flag per gestire la scrittura progressiva

    for idx, (datetime_key, parcels) in enumerate(tqdm(data_dict.items(), total=total_items, desc="Scrittura JSON", unit="timestamp")):
        if not first_item:
            json_file.write(",\n")  # Aggiungi una virgola prima di ogni nuovo blocco
        first_item = False

        json.dump({datetime_key: parcels}, json_file, ensure_ascii=False, indent=4)
    
    json_file.write("\n}")  

print(f'File ultimato {output_file_path}')
