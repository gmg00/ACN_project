from itertools import combinations
from math import sqrt
from tqdm import tqdm
from collections import defaultdict
import json

file_path = 'data_dict.json'

with open(file_path, 'r', encoding='utf-8') as json_file:
    data_dict = json.load(json_file)  

### FUNZIONE PER CALCOLO DISTANZA ###
def calculate_distance(pos1, pos2):
    return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)

# FUNZIONE PER OTTENERE EDGE ###
def get_edges(data_dict):
    edges_dict = defaultdict(list)

    # Raggruppa i dati per giorno
    daily_data = defaultdict(dict)
    for time, parcels in data_dict.items():
        date = time[:10]  # La data è la parte 'YYYY-MM-DD' del timestamp
        daily_data[date][time] = parcels

    # Itera per ogni giorno
    for date, timestamps in tqdm(daily_data.items(), desc="Elaborando giorni", unit="giorno"):  # Barra di avanzamento per giorno
        edges_tmp_per_day = []

        # Itera per ogni timestamp all'interno di quel giorno
        for time, parcels in timestamps.items():
            edges_tmp = []

            # Raccogli tutti gli utenti di tutti i parcel in un'unica lista per ogni timestamp
            all_users = []
            for parcel, data in parcels.items():
                all_users.extend(data["users"])  # Aggiungi tutti gli utenti di quel parcel

            # Se ci sono almeno due utenti, calcola le combinazioni
            if len(all_users) > 1:
                for user1, user2 in combinations(all_users, 2):
                    pos1 = user1["position"]
                    pos2 = user2["position"]

                    # Calcola la distanza tra i due utenti
                    if calculate_distance(pos1, pos2) <= 10:  # Soglia di distanza, puoi modificarla
                        edges_tmp.append([user1["id"], user2["id"]])

            # Aggiungi gli edge per questo timestamp al dizionario per il giorno
            if edges_tmp:  # Solo se ci sono edge
                edges_tmp_per_day.append((time, edges_tmp))

        # Aggiungi la data e gli edge per tutti i timestamp di quel giorno
        if edges_tmp_per_day:
            edges_dict[date] = edges_tmp_per_day

    return edges_dict


edges_dict = get_edges(data_dict)


### CREAZIONE JSON ###
output_file_path = 'edges.json'

with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(edges_dict, json_file, ensure_ascii=False, indent=4)
