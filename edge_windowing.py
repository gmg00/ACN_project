import json
from collections import defaultdict
from tqdm import tqdm

### RAGRUPPAMENTO EDGE IN WINDOW DI 20 MINUTI E CALCOLO DEI PESI NORMALIZZATI ###
def aggregate_edges_with_normalized_weights(edges_file_path, output_file_path, window_minutes=20):
    # Leggi il file JSON degli edge
    with open(edges_file_path, 'r', encoding='utf-8') as json_file:
        edges_data = json.load(json_file)

    # Dati raggruppati per finestra temporale
    aggregated_edges = defaultdict(lambda: defaultdict(int))  # {window_start: {(user1, user2): count}}

    # Itera sui dati raggruppati per giorno
    for date, timestamps in tqdm(edges_data.items(), desc="Elaborando giorni", unit="giorno"):
        for time, edges in timestamps:
            try:
                # Estrarre l'ora e i minuti dal timestamp
                date_time = time[:16]  # Prende solo "YYYY-MM-DD HH:MM"
                hour = date_time[:13]  # Prende "YYYY-MM-DD HH"
                minute = int(date_time[14:16])

                # Calcola l'inizio della finestra di 20 minuti
                rounded_minute = (minute // window_minutes) * window_minutes
                window_start = f"{hour}:{rounded_minute:02d}"

                # Accumula gli edge nella finestra corrispondente
                for edge in edges:
                    user_pair = tuple(sorted(edge))  # Ordina per evitare duplicati tipo (A, B) e (B, A)
                    aggregated_edges[window_start][user_pair] += 1  # Incrementa il peso per questa coppia
            except (ValueError, IndexError):
                print(f"Timestamp non valido: {time}")
                continue

    # DIZIONARIO CON PESI NORMALIZZATI
    final_edges = {}
    for window, edge_counts in aggregated_edges.items():
        n = sum(edge_counts.values())  # Numero totale di contatti nella finestra
        weighted_edges = [
            {"users": list(edge), "weight": count / n} for edge, count in edge_counts.items()
        ]
        final_edges[window] = weighted_edges

    # DIZIONARIO IN JSON 
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_edges, json_file, ensure_ascii=False, indent=4)

edges_file_path = 'edges.json'
output_file_path = 'edges_with_weights.json'

aggregate_edges_with_normalized_weights(edges_file_path, output_file_path)
