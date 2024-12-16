import json
from collections import defaultdict
from tqdm import tqdm

### CALCOLO DEI PESI AGGREGATI CON METADATI DEGLI UTENTI ###
def calculate_global_weights_with_user_info(edges_file_path, output_file_path, window_minutes=20):
    # Leggi il file JSON degli edge
    with open(edges_file_path, 'r', encoding='utf-8') as json_file:
        edges_data = json.load(json_file)

    # Dati aggregati
    time_shared = defaultdict(lambda: defaultdict(int))  # {user1: {user2: T_ij}}
    user_totals = defaultdict(int)  # {user: T_i}
    shared_windows = defaultdict(lambda: defaultdict(set))  # {user1: {user2: {windows}}}
    user_windows = defaultdict(set)  # {user: {windows}}

    # Itera sui dati per calcolare i totali
    for date, timestamps in tqdm(edges_data.items(), desc="Elaborando giorni", unit="giorno"):
        for time, edges in timestamps:
            try:
                # Calcolo della finestra temporale
                date_time = time[:16]  # Prende solo "YYYY-MM-DD HH:MM"
                hour = date_time[:13]  # Prende "YYYY-MM-DD HH"
                minute = int(date_time[14:16])
                rounded_minute = (minute // window_minutes) * window_minutes
                window_start = f"{hour}:{rounded_minute:02d}"

                # Aggiorna i dati per ogni coppia
                for edge in edges:
                    user1, user2 = tuple(sorted(edge))  # Ordina gli utenti per evitare duplicati
                    time_shared[user1][user2] += 1  # Incrementa il tempo condiviso T_ij
                    time_shared[user2][user1] += 1  # Simmetria per T_ij
                    user_totals[user1] += 1  # Incrementa il tempo totale T_i
                    user_totals[user2] += 1  # Incrementa il tempo totale T_j
                    shared_windows[user1][user2].add(window_start)  # Aggiungi la finestra condivisa
                    shared_windows[user2][user1].add(window_start)  # Simmetria per le finestre condivise
                    user_windows[user1].add(window_start)  # Aggiorna le finestre totali di user1
                    user_windows[user2].add(window_start)  # Aggiorna le finestre totali di user2
            except (ValueError, IndexError):
                print(f"Timestamp non valido: {time}")
                continue

    # Calcolo dei pesi
    weights = []
    for user1 in time_shared:
        for user2 in time_shared[user1]:
            if user1 < user2:  # Calcola una sola volta per ogni coppia
                T_ij = time_shared[user1][user2]  # Tempo totale condiviso
                T_i = user_totals[user1]  # Tempo totale giocato da user1
                T_j = user_totals[user2]  # Tempo totale giocato da user2
                F_ij = len(shared_windows[user1][user2])  # Numero di finestre condivise
                F_i = len(user_windows[user1])  # Numero totale di finestre di user1
                F_j = len(user_windows[user2])  # Numero totale di finestre di user2

                # Calcola il peso per user1 rispetto a user2
                weight_user1 = (T_ij / T_i) * (F_ij / F_i) if T_i > 0 and F_i > 0 else 0
                weight_user2 = (T_ij / T_j) * (F_ij / F_j) if T_j > 0 and F_j > 0 else 0

                # Aggiungi al risultato
                weights.append({
                    "users": [user1, user2],
                    "weight_user1": weight_user1,
                    "weight_user2": weight_user2,
                    "shared_time": T_ij,
                    "shared_windows": F_ij
                })

    # Aggiungi informazioni sui singoli utenti
    user_info = []
    for user, total_time in user_totals.items():
        total_windows = len(user_windows[user])  # Numero totale di finestre
        user_info.append({
            "user": user,
            "total_time_user": total_time,  # Minuti totali giocati
            "total_windows_user": total_windows  # Finestre totali
        })

    # Combina i risultati
    output_data = {
        "edges": weights,
        "users": user_info
    }

    # Scrittura del file JSON finale
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

# File di input e output
edges_file_path = 'edges.json'
output_file_path = 'global_weights_with_user_info.json'

# Chiamata alla funzione
calculate_global_weights_with_user_info(edges_file_path, output_file_path)
