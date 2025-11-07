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
from scipy.stats import spearmanr  # >>> NEW

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
    nswap: int = 10_000
    max_tries: int = 100_000
    log_every: Optional[int] = 2_000


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


def _edge_weight_dict(G: nx.Graph) -> dict[Edge, float]:
    """Ritorna un dizionario (edge -> peso), con edge come (min,max)."""
    weights = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        weights[(a, b)] = data.get("weight", 1.0)
    return weights


# =========================
# Core: rewiring + shuffle pesi
# =========================

def rewire_preserving_degree_and_randomizing_weights(
    G: nx.Graph,
    cfg: RewireConfig = RewireConfig(),
    seed: Optional[int] = None
) -> RewireResult:
    """Esegue double-edge swaps preservando i gradi e randomizzando i pesi."""
    rng = random.Random(seed)
    H = G.copy()

    original_weights = [data.get("weight", 1.0) for _, _, data in H.edges(data=True)]

    swaps_done = nx.double_edge_swap(H, nswap=cfg.nswap, max_tries=cfg.max_tries, seed=seed)

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
    OUTPUT_CORR_NPY = "spearman_corr_L1_Null3.npy"

    N_MODELS = 5000

    cfg = RewireConfig(
        nswap=50_000,
        max_tries=500_000,
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
    weights1 = _edge_weight_dict(layer1)

    overlaps = np.zeros(N_MODELS)
    correlations = np.zeros(N_MODELS)  # >>> NEW
    rows = []

    t0 = time.time()
    for i in range(N_MODELS):
        res = rewire_preserving_degree_and_randomizing_weights(layer3.copy(), cfg, seed=i)
        EN = _edge_set_undirected(res.graph)
        weightsN = _edge_weight_dict(res.graph)

        # --- Overlap ---
        overlap = len(E1 & EN) / min(lenE1, len(EN)) if min(lenE1, len(EN)) > 0 else 0.0
        overlaps[i] = overlap

        # --- Correlazione di Spearman ---
        common_edges = E1 & EN
        if len(common_edges) > 1:
            w1 = [weights1[e] for e in common_edges]
            w2 = [weightsN[e] for e in common_edges]
            corr, _ = spearmanr(w1, w2)
            correlation = corr if not np.isnan(corr) else 0.0
        else:
            correlation = 0.0
        correlations[i] = correlation

        rows.append([i, overlap, correlation])

        frac = (i + 1) / N_MODELS
        bar = "█" * int(30 * frac) + "·" * (30 - int(30 * frac))
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (N_MODELS - (i + 1))
        print(f"\r[{bar}] {i+1}/{N_MODELS} ({frac:>6.2%}) | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", end="", flush=True)

    print("\n✅ Completed null model generation.")

    npy_path = os.path.join(OUT_DIR, OUTPUT_NPY)
    np.save(npy_path, overlaps)
    print(f"Saved overlaps -> {npy_path}")

    corr_path = os.path.join(OUT_DIR, OUTPUT_CORR_NPY)
    np.save(corr_path, correlations)
    print(f"Saved correlations -> {corr_path}")

