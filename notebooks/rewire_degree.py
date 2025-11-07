# rewire_degree_weights.py
# MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Hashable, Set, Tuple, List
import networkx as nx
import random
import time
import numpy as np
import os
import csv

# --- Fix per NumPy >= 2.0 e NetworkX ---
if not hasattr(np, "float_"):
    np.float_ = np.float64
# ---------------------------------------

Edge = Tuple[Hashable, Hashable]


# =========================
# Config e risultati
# =========================

@dataclass(frozen=True)
class RewireConfig:
    """Configurazione dell'algoritmo di rewiring."""
    nswap: int = 10_000         # numero di double-edge-swap da tentare
    max_tries: int = 100_000    # limite massimo di tentativi
    log_every: Optional[int] = 2_000  # log periodico


@dataclass
class RewireResult:
    graph: nx.Graph
    swaps: int
    steps: int


# =========================
# Funzioni di utilità
# =========================

def _edge_set_undirected(G: nx.Graph) -> Set[Edge]:
    out = set()
    for u, v in G.edges():
        if u == v:
            continue
        try:
            a, b = (u, v) if u < v else (v, u)
        except TypeError:
            a, b = (u, v) if str(u) < str(v) else (v, u)
        out.add((a, b))
    return out


# =========================
# Core: rewiring + shuffle pesi
# =========================

def rewire_preserving_degree_and_randomizing_weights(
    G: nx.Graph,
    cfg: RewireConfig = RewireConfig(),
    seed: Optional[int] = None
) -> RewireResult:
    """
    Esegue double-edge swaps preservando i gradi dei nodi e
    randomizzando i pesi sugli archi.
    """
    rng = random.Random(seed)
    H = G.copy()

    # --- Estrai i pesi originali ---
    original_weights = [data.get("weight", 1.0) for _, _, data in H.edges(data=True)]

    # --- Rewiring topologico ---
    swaps_done = nx.double_edge_swap(H, nswap=cfg.nswap, max_tries=cfg.max_tries, seed=seed)

    # --- Randomizza i pesi ---
    rng.shuffle(original_weights)
    for i, (u, v) in enumerate(H.edges()):
        H[u][v]["weight"] = original_weights[i]

    if cfg.log_every:
        print(f"✅ Completed {swaps_done} swaps (target {cfg.nswap})")

    return RewireResult(H, swaps_done, cfg.max_tries)


# =========================
# MAIN: genera N null models
# =========================

if __name__ == "__main__":
    # >>>>>>> CONFIGURAZIONE <<<<<<<
    LAYER1_GEXF = "/Users/HP/Desktop/layer1.gexf"
    LAYER3_GEXF = "/Users/HP/Desktop/layer3.gexf"
    OUT_DIR = '/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/'
    OUTPUT_NPY = "edge_overlap_L1_Null3.npy"

    N_MODELS = 5000

    cfg = RewireConfig(
        nswap=10_000,
        max_tries=100_000,
        log_every=None,
    )
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    print(f"Loading Layer 1 from: {LAYER1_GEXF}")
    layer1 = nx.read_gexf(LAYER1_GEXF)
    print(f"Loaded Layer 1: |V|={layer1.number_of_nodes()}, |E|={layer1.number_of_edges()}")

    print(f"Loading Layer 3 from: {LAYER3_GEXF}")
    layer3 = nx.read_gexf(LAYER3_GEXF)
    print(f"Loaded Layer 3: |V|={layer3.number_of_nodes()}, |E|={layer3.number_of_edges()}")

    E1 = _edge_set_undirected(layer1)
    lenE1 = len(E1)

    overlaps = np.zeros(N_MODELS)
    rows = []

    t0 = time.time()
    for i in range(N_MODELS):
        res = rewire_preserving_degree_and_randomizing_weights(layer3.copy(), cfg, seed=i)
        EN = _edge_set_undirected(res.graph)
        overlap = len(E1 & EN) / min(lenE1, len(EN)) if min(lenE1, len(EN)) > 0 else 0.0
        overlaps[i] = overlap
        rows.append([i, overlap])

        frac = (i + 1) / N_MODELS
        bar = "█" * int(30 * frac) + "·" * (30 - int(30 * frac))
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (N_MODELS - (i + 1))
        print(f"\r[{bar}] {i+1}/{N_MODELS} ({frac:>6.2%}) | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", end="", flush=True)

    print("\n✅ Completed null model generation.")

    npy_path = os.path.join(OUT_DIR, OUTPUT_NPY)
    np.save(npy_path, overlaps)
    print(f"Saved NPY -> {npy_path}")
