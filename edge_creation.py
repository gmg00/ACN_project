import json
from math import sqrt
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

file_path = 'data_dict.json'

with open(file_path, 'r', encoding='utf-8') as json_file:
    data_dict = json.load(json_file)  

### FUNZIONE PER CALCOLO DISTANZA ###

def calculate_distance(pos1, pos2):
    return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)

# FUNZIONE PER OTTENERE EDGE ###

def get_edges(data_dict):
    edges_dict = {}

    # Raggruppa i timestamp per giorno
    daily_data = defaultdict(dict)
    for time, parcels in data_dict.items():
        date = time[:10]  # Prendi solo la parte della data (YYYY-MM-DD)
        daily_data[date][time] = parcels

    # Itera per giorno
    for date, timestamps in tqdm(daily_data.items(), desc="Elaborando giorni", unit="giorno"):  # Barra di avanzamento per giorno
        # Itera su ogni timestamp all'interno di quel giorno
        for time, parcels in timestamps.items():
            edges_tmp = []

            # Per ogni parcel (dove sono presenti gli utenti)
            for parcel, data in parcels.items():
                users = data["users"]
                user_count = data["user_count"]

                # Se ci sono almeno due utenti, possiamo creare combinazioni di edges
                if user_count > 1:
                    # Crea tutte le combinazioni di coppie di utenti nello stesso parcel
                    for user1, user2 in combinations(users, 2):
                        pos1 = user1["position"]
                        pos2 = user2["position"]

                        # Calcola la distanza tra i due utenti
                        if calculate_distance(pos1, pos2) <= 10:
                            edges_tmp.append([user1["id"], user2["id"]])

            edges_dict[time] = edges_tmp

    return edges_dict

edges_dict = get_edges(data_dict)

### CREAZIONE JSON ###
output_file_path = 'edges.json'

with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(edges_dict, json_file, ensure_ascii=False, indent=4)
