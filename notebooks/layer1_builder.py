# layer1_builder.py
# MIT License
import networkx as nx
import numpy as np
import ast
import json
from pathlib import Path

# --- Fix compatibilità NumPy >= 2.0 ---
if not hasattr(np, "float_"):
    np.float_ = np.float64
# --------------------------------------

def create_layer1_threshold(edges_dict: dict, k_value: float, undirected: bool = False) -> nx.Graph:
    """
    Crea il Layer 1 filtrando gli archi in base a una soglia percentilica sui pesi
    e normalizzando i pesi con min-max normalization.

    Args:
        edges_dict (dict): Dizionario con chiavi come stringhe "(u, v)" e valori con pesi.
        k_value (float): Percentile (0–100) per la soglia dei pesi.
        undirected (bool): Se True, usa 'simmetric_weight' e restituisce un grafo non diretto.

    Returns:
        nx.Graph: Grafo filtrato con pesi normalizzati.
    """
    # 1️⃣ Raccogli tutti i pesi
    if undirected:
        weights = [edge_data['simmetric_weight'] for edge_data in edges_dict.values()]
    else:
        weights = [
            w for edge_data in edges_dict.values()
            for w in (edge_data['weight_user1'], edge_data['weight_user2'])
        ]

    # 2️⃣ Normalizzazione min-max
    w_min, w_max = min(weights), max(weights)
    normalize = (lambda w: 1.0) if w_max == w_min else (lambda w: (w - w_min) / (w_max - w_min))

    # 3️⃣ Calcola la soglia (percentile, poi normalizzato)
    raw_threshold = np.percentile(weights, k_value)
    threshold = 1.0 if w_max == w_min else (raw_threshold - w_min) / (w_max - w_min)

    # 4️⃣ Costruisci il grafo
    if undirected:
        G = nx.Graph()
        for edge_str, edge_data in edges_dict.items():
            u, v = ast.literal_eval(edge_str)
            w = normalize(edge_data['simmetric_weight'])
            if w > threshold:
                G.add_edge(u, v, weight=w)
    else:
        G = nx.DiGraph()
        for edge_str, edge_data in edges_dict.items():
            u, v = ast.literal_eval(edge_str)
            w1 = normalize(edge_data['weight_user1'])
            w2 = normalize(edge_data['weight_user2'])
            if w1 > threshold:
                G.add_edge(u, v, weight=w1)
            if w2 > threshold:
                G.add_edge(v, u, weight=w2)

    return G


def get_percentile(edges_dict: dict, k_value: float) -> float:
    """Restituisce il valore di peso (non normalizzato) corrispondente al percentile k_value."""
    weights = [edge_data['simmetric_weight'] for edge_data in edges_dict.values()]
    return float(np.percentile(weights, k_value))


if __name__ == "__main__":
    # >>> CONFIGURAZIONE <<<
    EDGES_PATH = '/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/edges_dict_weight.json'
    OUTPUT_GEXF = "/Users/HP/Desktop/layer1.gexf"
    K_PERCENTILE = 90.0
    UNDIRECTED = True
    # >>>>>>>>>>>>>>>>>>>>>>>

    # Carica i dati
    print(f"Reading edges from: {EDGES_PATH}")
    with open(EDGES_PATH, "r") as f:
        edges_dict = json.load(f)

    # Crea Layer 1
    layer1 = create_layer1_threshold(edges_dict, K_PERCENTILE, undirected=UNDIRECTED)

    # Salva in GEXF
    Path(OUTPUT_GEXF).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(layer1, OUTPUT_GEXF)
    print(f"✅ Layer 1 built with threshold={K_PERCENTILE}th percentile")
    print(f"   |V|={layer1.number_of_nodes()}, |E|={layer1.number_of_edges()}")
    print(f"   Saved to: {OUTPUT_GEXF}")