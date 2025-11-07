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
def contains_keyword(string, keywords):
    return any(keyword in string for keyword in keywords)

def create_edges(all_events, keywords):
    edges = []
    addresses = list(all_events.keys())
    for address, events in all_events.items():
        for event in events:
            nft_description = event.get('nft_description')

            if nft_description and "DCL Wearable" in nft_description:
                continue

            if event.get('nft_name') and contains_keyword(event.get('nft_name'), keywords):
                continue

            if event.get('from') == address and event.get('to') in addresses and event.get('to') != address:

                edges.append([address, event.get('to'), 1])
            if event.get('seller') == address and event.get('buyer') in addresses and event.get('buyer') != address:

                edges.append([address, event.get('buyer'), 1+event.get('price', 0)])

    return edges


if __name__ == "__main__":
    # >>> CONFIGURAZIONE <<<
    EDGES_PATH = '/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/address_to_events_final.json'
    OUTPUT_GEXF = "/Users/HP/Desktop/layer3.gexf"
    UNDIRECTED = True
    # >>>>>>>>>>>>>>>>>>>>>>>

    # Carica i dati
    print(f"Reading edges from: {EDGES_PATH}")
    with open(EDGES_PATH, "r") as f:
        address_to_events = json.load(f)

    # Crea Layer 3
    # Creating wearables windows for each address
    keywords = ["decentraland", "dcl", "decentral", "wearable", "decentral-games", "parcel", "MANA", 'Decentraland']
    edges = create_edges(address_to_events, keywords=keywords)
    
    edges_dict = {}
    for edge in edges:
        e = tuple(set(edge[0:2]))
        if e not in edges_dict.keys():
            edges_dict[e] = edge[2]
        else:
            edges_dict[e] += edge[2]

    layer3 = nx.Graph()
    for edge, weight in edges_dict.items():
        layer3.add_edge(edge[0], edge[1], weight=weight)

    # Salva in GEXF
    Path(OUTPUT_GEXF).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(layer3, OUTPUT_GEXF)

    print(f"   |V|={layer3.number_of_nodes()}, |E|={layer3.number_of_edges()}")
    print(f"   Saved to: {OUTPUT_GEXF}")