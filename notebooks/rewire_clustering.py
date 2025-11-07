# rewire_clustering.py
# MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Iterable, Optional, Hashable
import random
import networkx as nx
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
    # Obiettivo: numero di rewiring ACCETTATI da raggiungere per ogni null model
    target_accepted: int = 10_000
    # Limite di sicurezza sul n. massimo di step tentati
    max_steps: Optional[int] = None

    # Tolleranze
    tol_global_rel: float = 0.05
    tol_global_abs: float = 1e-4
    tol_local_rel: float = 0.10
    tol_local_abs: float = 1e-3

    # Logging e policy
    log_every: Optional[int] = 2_000
    accept_only_if_toward_target: bool = False
    fast_integer_local_check: bool = True


@dataclass
class RewireResult:
    graph: nx.Graph
    accepted: int          # rewiring accettati
    steps: int             # step tentati
    final_Cg: float
    target_Cg: float


# =========================
# Stato e utilità
# =========================

@dataclass
class _State:
    nodes: List[Hashable]
    neighbors: Dict[Hashable, Set[Hashable]]
    edges: List[Edge]
    edge_set: Set[Edge]
    deg: Dict[Hashable, int]
    den: Dict[Hashable, int]
    t: Dict[Hashable, int]
    triangles_total: int
    Wc: int
    target_Cg: float
    current_Cg: float
    C_base: Dict[Hashable, float]
    max_dt: Dict[Hashable, int]


def _normalize_edge(u: Hashable, v: Hashable) -> Edge:
    return (u, v) if u < v else (v, u)


def _build_state(G: nx.Graph, cfg: RewireConfig) -> _State:
    nodes = list(G.nodes())
    neighbors: Dict[Hashable, Set[Hashable]] = {u: set(G[u]) for u in nodes}
    edges = sorted([(min(u, v), max(u, v)) for u, v in G.edges() if u != v])
    edge_set = set(edges)
    deg = {u: len(neighbors[u]) for u in nodes}
    den = {u: d * (d - 1) for u, d in deg.items()}
    Wc = sum(d * (d - 1) // 2 for d in deg.values())
    t = nx.triangles(G)
    triangles_total = sum(t.values()) // 3
    target_Cg = 0.0 if Wc == 0 else (3.0 * triangles_total) / Wc
    current_Cg = target_Cg
    C_base = {u: (0.0 if den[u] == 0 else 2.0 * t[u] / den[u]) for u in nodes}
    from math import inf
    b_i = {u: max(cfg.tol_local_rel * C_base[u], cfg.tol_local_abs) for u in nodes}
    max_dt = {u: (inf if den[u] == 0 else int((b_i[u] * den[u]) / 2 + 1e-9)) for u in nodes}
    return _State(nodes, neighbors, edges, edge_set, deg, den, t, triangles_total, Wc,
                  target_Cg, current_Cg, C_base, max_dt)


def _cn_list(state: _State, a: Hashable, b: Hashable) -> List[Hashable]:
    A, B = state.neighbors[a], state.neighbors[b]
    if len(A) > len(B):
        A, B = B, A
    return [z for z in A if z in B]


def _global_update(state: _State, len_old1: int, len_old2: int, len_new1: int, len_new2: int):
    triangles_new = state.triangles_total - len_old1 - len_old2 + len_new1 + len_new2
    Cg_new = 0.0 if state.Wc == 0 else (3.0 * triangles_new) / state.Wc
    return triangles_new, Cg_new


def _global_ok(state: _State, Cg_new: float, cfg: RewireConfig) -> bool:
    eps = 1e-12
    return abs(Cg_new - state.target_Cg) <= cfg.tol_global_rel * max(state.target_Cg, eps)


def _local_dt_from_cn(old1, old2, new1, new2, cn_old1, cn_old2, cn_new1, cn_new2):
    dt: Dict[Hashable, int] = {}
    u, v = old1
    for z in cn_old1:
        dt[u] = dt.get(u, 0) - 1; dt[v] = dt.get(v, 0) - 1; dt[z] = dt.get(z, 0) - 1
    x, y = old2
    for z in cn_old2:
        dt[x] = dt.get(x, 0) - 1; dt[y] = dt.get(y, 0) - 1; dt[z] = dt.get(z, 0) - 1
    a1, b1 = new1
    for z in cn_new1:
        dt[a1] = dt.get(a1, 0) + 1; dt[b1] = dt.get(b1, 0) + 1; dt[z] = dt.get(z, 0) + 1
    a2, b2 = new2
    for z in cn_new2:
        dt[a2] = dt.get(a2, 0) + 1; dt[b2] = dt.get(b2, 0) + 1; dt[z] = dt.get(z, 0) + 1
    return dt


def _local_ok_fast(state: _State, dt: Dict[Hashable, int], cfg: RewireConfig) -> bool:
    return all(abs(delta) <= state.max_dt[n] for n, delta in dt.items())


def _local_ok_float(state: _State, dt: Dict[Hashable, int], cfg: RewireConfig) -> bool:
    eps = 1e-12
    for n, dtn in dt.items():
        if state.den[n] == 0:
            continue
        Ci_base = state.C_base[n]
        Ci_new = 2.0 * (state.t[n] + dtn) / state.den[n]
        if abs(Ci_new - Ci_base) > cfg.tol_local_rel * max(Ci_base, eps):
            return False
    return True


def _commit(state: _State, i, j, old1, old2, new1, new2, dt, triangles_new, Cg_new):
    (u, v), (x, y) = old1, old2
    a1, b1 = new1; a2, b2 = new2
    state.neighbors[u].remove(v); state.neighbors[v].remove(u); state.edge_set.remove((u, v))
    state.neighbors[x].remove(y); state.neighbors[y].remove(x); state.edge_set.remove((x, y))
    state.neighbors[a1].add(b1); state.neighbors[b1].add(a1); state.edge_set.add((a1, b1))
    state.neighbors[a2].add(b2); state.neighbors[b2].add(a2); state.edge_set.add((a2, b2))
    state.edges[i] = (a1, b1); state.edges[j] = (a2, b2)
    for n, dtn in dt.items():
        state.t[n] = state.t.get(n, 0) + dtn
    state.triangles_total = triangles_new
    state.current_Cg = Cg_new


# =========================
# API principale
# =========================

def rewire_preserving_clustering(G: nx.Graph, cfg: RewireConfig = RewireConfig(), seed: Optional[int] = None) -> RewireResult:
    rng = random.Random(seed)
    state = _build_state(G, cfg)
    m = len(state.edges)
    if m < 2:
        return RewireResult(G.copy(), 0, 0, state.current_Cg, state.target_Cg)

    # default max_steps se non passato: 20x il target accettati
    max_steps = cfg.max_steps if cfg.max_steps is not None else 20 * max(1, cfg.target_accepted)

    accepted = 0
    steps = 0

    while accepted < cfg.target_accepted and steps < max_steps:
        steps += 1

        i, j = rng.sample(range(m), 2)
        old1, old2 = state.edges[i], state.edges[j]
        u, v = old1; x, y = old2
        if len({u, v, x, y}) < 4:
            continue

        cand_new1 = _normalize_edge(u, x)
        cand_new2 = _normalize_edge(v, y)
        alt_new1 = _normalize_edge(u, y)
        alt_new2 = _normalize_edge(v, x)

        def valid(n1, n2): 
            return n1[0] != n1[1] and n2[0] != n2[1] and (n1 not in state.edge_set) and (n2 not in state.edge_set)

        proposals = []
        if valid(cand_new1, cand_new2): proposals.append((cand_new1, cand_new2))
        if valid(alt_new1, alt_new2):   proposals.append((alt_new1, alt_new2))
        if not proposals: 
            continue

        cn_old1 = _cn_list(state, u, v)
        cn_old2 = _cn_list(state, x, y)
        best = None; best_dist = float("inf")
        choice = rng.randrange(len(proposals))

        for idx, (n1, n2) in enumerate(proposals):
            cn_new1 = _cn_list(state, *n1)
            cn_new2 = _cn_list(state, *n2)
            triangles_new, Cg_new = _global_update(state, len(cn_old1), len(cn_old2), len(cn_new1), len(cn_new2))
            if not _global_ok(state, Cg_new, cfg):
                continue
            dist = abs(Cg_new - state.target_Cg)
            if cfg.accept_only_if_toward_target and dist > abs(state.current_Cg - state.target_Cg):
                continue
            dt = _local_dt_from_cn(old1, old2, n1, n2, cn_old1, cn_old2, cn_new1, cn_new2)
            if cfg.fast_integer_local_check:
                if not _local_ok_fast(state, dt, cfg): 
                    continue
            else:
                if not _local_ok_float(state, dt, cfg): 
                    continue
            if cfg.accept_only_if_toward_target:
                if dist < best_dist:
                    best = (n1, n2, dt, triangles_new, Cg_new)
                    best_dist = dist
            else:
                if idx == choice:
                    best = (n1, n2, dt, triangles_new, Cg_new)
                    break

        if best is None:
            continue

        new1, new2, dt, triangles_new, Cg_new = best
        _commit(state, i, j, old1, old2, new1, new2, dt, triangles_new, Cg_new)
        accepted += 1

        if cfg.log_every and (steps % cfg.log_every == 0):
            print(f"Step {steps}, accepted={accepted}, C={state.current_Cg:.6f}")

    if cfg.log_every:
        print(f"✅ Successful rewirings: {accepted}/{cfg.target_accepted} (steps tried: {steps})")
        print(f"Final C={state.current_Cg:.6f}, Target={state.target_Cg:.6f}")

    H = nx.Graph()
    H.add_nodes_from(state.nodes)
    H.add_edges_from(state.edges)
    return RewireResult(H, accepted, steps, state.current_Cg, state.target_Cg)


__all__ = ["RewireConfig", "RewireResult", "rewire_preserving_clustering"]

# ============================================================
# MAIN: carica L1/L2 + genera N null models con barra + overlap L1–Null
# ============================================================

def _edge_set_undirected(G: nx.Graph) -> Set[Tuple[Hashable, Hashable]]:
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


if __name__ == "__main__":
    # >>>>>>> CONFIGURAZIONE <<<<<<<
    LAYER1_GEXF = "/Users/HP/Desktop/layer1.gexf"
    LAYER3_GEXF = "/Users/HP/Desktop/layer3.gexf"
    OUT_DIR = '/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/'
    OUTPUT_NPY = "edge_overlap_L1_Null3.npy"

    N_MODELS = 5000

    # Obiettivo: 10k accettati per modello, con limite di sicurezza sugli step
    cfg = RewireConfig(
        target_accepted=10_000,
        max_steps=5_000_000,        
        tol_global_rel=0.05,
        tol_local_rel=0.05,
        accept_only_if_toward_target=False,
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
        res = rewire_preserving_clustering(layer3.copy(), cfg, seed=i)
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