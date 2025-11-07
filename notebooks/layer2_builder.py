

from __future__ import annotations

import json
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import networkx as nx
import numpy as np

# --- NumPy 2.x compatibility patch for NetworkX ---
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
# --------------------------------------------------

# -------------------------
# Types
# -------------------------
User = str
Item = str
Collection = str
Edge = Tuple[User, User, float]  # (u, v, weight)


# -------------------------
# Core similarity routines
# -------------------------
def calculate_similarity_index2(
    users_items: Mapping[User, Sequence[Item]],
    item_rarity: Mapping[Item, int],
    item_collections: Mapping[Item, Collection],
    alpha: float = 0.5,
    collection_rarity: Mapping[Collection, int] | None = None,
) -> List[Edge]:
    """
    Calculate normalized similarity weights for all user pairs.

    Args:
        users_items: dict user -> list of item ids owned by the user.
        item_rarity: dict item -> frequency across all users (lower => rarer).
        item_collections: dict item -> collection id.
        alpha: weight given to collection-level similarity.
        collection_rarity: optional dict collection -> frequency across all users.

    Returns:
        List of tuples (user1, user2, normalized_weight) with normalized_weight in [0, 1].
        Only pairs with non-zero signal (common items OR common collections) are returned.
    """
    edges: List[Edge] = []

    # Iterate all unordered pairs of users
    for u, v in combinations(users_items.keys(), 2):
        items_u = set(users_items[u])
        items_v = set(users_items[v])

        if not items_u or not items_v:
            continue

        # Shared signals
        common_items = items_u & items_v
        if not common_items:
            # We still consider common collections, so do not early-continue here
            pass

        cols_u = {item_collections[item] for item in items_u}
        cols_v = {item_collections[item] for item in items_v}
        common_cols = cols_u & cols_v

        if not common_items and not common_cols:
            continue

        # Item-based component: rarer shared items contribute more
        weight_items = sum(1.0 / item_rarity[item] for item in common_items)

        # Collection-based component
        if collection_rarity:
            weight_cols = sum(1.0 / collection_rarity[c] for c in common_cols)
        else:
            weight_cols = float(len(common_cols))

        raw_weight = weight_items + alpha * weight_cols

        # Normalization by a per-pair upper bound
        max_items = sum(1.0 / item_rarity[item] for item in (items_u | items_v))
        if collection_rarity:
            max_cols = sum(1.0 / collection_rarity[c] for c in (cols_u | cols_v))
        else:
            max_cols = float(len(cols_u | cols_v))

        denom = max_items + alpha * max_cols
        if denom <= 0.0:
            continue

        normalized = raw_weight / denom  # in [0, 1]

        # Keep every positive signal (you can post-filter by percentile later)
        if normalized > 0.0:
            edges.append((u, v, float(normalized)))

    return edges


def compute_edges2(
    all_wearables: Mapping[User, Sequence[Mapping[str, str]]],
    alpha: float = 0.5,
    wearable_to_collection: Mapping[Item, Collection] | None = None,
) -> List[Edge]:
    """
    Convert raw wearable records into pairwise edges with normalized weights.

    Args:
        all_wearables: dict user -> list of wearables,
            each wearable record contains 'nft_name' and 'nft_collection' keys.
        alpha: weight for the collection-based component.
        wearable_to_collection: optional override mapping item -> collection.

    Returns:
        List of edges (u, v, weight) with weight in [0, 1].
    """
    users_items: Dict[User, List[Item]] = {}
    item_rarity: Dict[Item, int] = {}
    item_collections: Dict[Item, Collection] = {}
    collection_rarity: Dict[Collection, int] = {}

    # Populate per-user items, item rarity and collections
    for user, wearables in all_wearables.items():
        users_items[user] = []
        for w in wearables:
            nft_name: Item = w["nft_name"]
            # Allow an override mapping for collections if you maintain one elsewhere
            nft_collection: Collection = (
                wearable_to_collection.get(nft_name)  # type: ignore[union-attr]
                if wearable_to_collection is not None
                else w["nft_collection"]
            )

            item_rarity[nft_name] = item_rarity.get(nft_name, 0) + 1
            collection_rarity[nft_collection] = collection_rarity.get(nft_collection, 0) + 1
            users_items[user].append(nft_name)
            item_collections[nft_name] = nft_collection

    # Build edges with the similarity function above
    edges = calculate_similarity_index2(
        users_items=users_items,
        item_rarity=item_rarity,
        item_collections=item_collections,
        alpha=alpha,
        collection_rarity=collection_rarity,
    )
    return edges


def create_layer2(edges: Iterable[Edge], threshold: float) -> nx.Graph:
    """
    Build an undirected weighted graph from edges whose weight > threshold.

    Args:
        edges: iterable of (u, v, weight).
        threshold: keep edges with weight strictly > threshold.

    Returns:
        nx.Graph with `weight` edge attribute.
    """
    G = nx.Graph()
    for u, v, w in edges:
        if w > threshold:
            G.add_edge(u, v, weight=float(w))
    return G


# -------------------------
# IO helpers
# -------------------------
def load_address_to_wearables(json_path: str) -> Dict[User, List[Dict[str, str]]]:
    """Load the address_to_wearables JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def filter_users_by_k(address_to_wearables: Mapping[User, Sequence[Mapping[str, str]]], k: int) -> Dict[User, List[Dict[str, str]]]:
    """Keep only users with at least k wearables."""
    return {u: list(ws) for u, ws in address_to_wearables.items() if len(ws) >= k}


def save_graph_graphml(G: nx.Graph, path: str) -> None:
    """Save graph to GraphML (.graphml)."""
    nx.write_graphml(G, path)


def save_graph_gexf(G: nx.Graph, path: str) -> None:
    """Save graph to GEXF (.gexf)."""
    nx.write_gexf(G, path)


# -------------------------
# High-level builder
# -------------------------
def build_layer2_from_json(
    json_path: str,
    k: int = 15,
    alpha: float = 0.5,
    percentile: float = 90.0,
    *,
    wearable_to_collection: Mapping[Item, Collection] | None = None,
) -> Tuple[nx.Graph, float]:
    """
    Full pipeline to build Layer 2 directly from a JSON file.

    Steps:
      (1) load JSON,
      (2) filter users with at least k wearables,
      (3) compute pairwise edges with normalized weights,
      (4) threshold by the given percentile,
      (5) build and return nx.Graph.

    Args:
        json_path: path to address_to_wearables.json
        k: minimum number of wearables per user
        alpha: collection similarity weight
        percentile: percentile for the weight threshold (e.g., 90.0)
        wearable_to_collection: optional mapping item->collection override.

    Returns:
        (G, thr) where:
          - G is the resulting nx.Graph
          - thr is the numeric threshold used (float)
    """
    data = load_address_to_wearables(json_path)
    filtered = filter_users_by_k(data, k)
    edges = compute_edges2(filtered, alpha=alpha, wearable_to_collection=wearable_to_collection)

    weights = np.asarray([w for _, _, w in edges], dtype=float)
    if weights.size == 0:
        # Empty graph edge-case: return an empty graph and threshold 0.0
        return nx.Graph(), 0.0

    thr = float(np.percentile(weights, percentile))
    G = create_layer2(edges, threshold=thr)
    return G, thr


# -------------------------
# Example usage 
# -------------------------
if __name__ == "__main__":
     JSON_PATH = "/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/address_to_wearables.json"
     OUTPUT_GEXF = "/Users/HP/Desktop/layer2.gexf"

     G, thr = build_layer2_from_json(
         json_path=JSON_PATH,
         k=15,
         alpha=0.5,
         percentile=90.0,
     )
     print(f"Layer 2 built with threshold={thr:.6f}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
     nx.write_gexf(G, OUTPUT_GEXF)