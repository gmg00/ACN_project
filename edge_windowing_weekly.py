import json
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime, timedelta

### RAGGRUPPAMENTO EDGE IN FINESTRE SETTIMANALI E CALCOLO DEI PESI NORMALIZZATI ###
def aggregate_edges_with_normalized_weights(edges_file_path, output_file_path, window_days=7):
    # Leggi il file JSON degli edge
    with open(edges_file_path, 'r', encoding='utf-8') as json_file:
        edges_data = json.load(json_file)

    # Dati raggruppati per finestra temporale (settimana)
    aggregated_edges = defaultdict(lambda: defaultdict(int))  # {week_start: {(user1, user2): count}}

    # Funzione per ottenere l'inizio della settimana da una data
    def get_week_start(date_str):
        date = datetime.strptime(date_str, "%Y-%m-%d")
        # Trova il primo giorno della settimana (lunedì)
        start_of_week = date - timedelta(days=date.weekday())  # settimana inizia di lunedì
        return start_of_week.strftime("%Y-%m-%d")  # formato 'YYYY-MM-DD'

    # Itera sui dati raggruppati per giorno
    for date, timestamps in tqdm(edges_data.items(), desc="Elaborando giorni", unit="giorno"):
        week_start = get_week_start(date)  # Ottieni la data di inizio della settimana per il giorno
        for time, edges in timestamps:
            try:
                # Estrarre l'ora e i minuti dal timestamp
                date_time = time[:16]  # Prende solo "YYYY-MM-DD HH:MM"
                # Accumula gli edge nella settimana corrispondente
                for edge in edges:
                    user_pair = tuple(sorted(edge))  # Ordina per evitare duplicati tipo (A, B) e (B, A)
                    aggregated_edges[week_start][user_pair] += 1  # Incrementa il peso per questa coppia
            except (ValueError, IndexError):
                print(f"Timestamp non valido: {time}")
                continue

    # DIZIONARIO CON PESI NORMALIZZATI
    final_edges = {}
    for week, edge_counts in aggregated_edges.items():
        n = sum(edge_counts.values())  # Numero totale di contatti nella settimana
        weighted_edges = [
            {"users": list(edge), "weight": count / n} for edge, count in edge_counts.items()
        ]
        final_edges[week] = weighted_edges

    # DIZIONARIO IN JSON 
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_edges, json_file, ensure_ascii=False, indent=4)

edges_file_path = 'edges.json'
output_file_path = 'edges_with_weights_weekly.json'

aggregate_edges_with_normalized_weights(edges_file_path, output_file_path)
