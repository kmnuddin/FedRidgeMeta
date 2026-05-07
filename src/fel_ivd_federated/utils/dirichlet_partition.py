"""
dirichlet_partition.py — Partition client pools into C simulated sub-clients
using Dirichlet allocation over class labels.

Given a pool dict (emb, label, originals, augs_by_axial, ...),
split its originals into `n_sub` sub-clients where each sub-client
receives a non-IID class distribution drawn from Dir(alpha).

Small alpha → extreme non-IID (near single-class clients)
Large alpha → near-IID (uniform class distribution)
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def dirichlet_partition(
    pool: dict,
    n_sub: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_sub: int = 2,
) -> List[dict]:
    """
    Partition a single client pool into n_sub simulated sub-clients
    using Dirichlet allocation.

    Parameters
    ----------
    pool : dict
        One entry from load_clients_from_csvs. Must have keys:
        'emb', 'label', 'originals', 'axial', 'augs_by_axial'.
    n_sub : int
        Number of sub-clients to create from this pool.
    alpha : float
        Dirichlet concentration parameter. Smaller = more heterogeneous.
    seed : int
        Random seed.
    min_samples_per_sub : int
        Minimum samples any sub-client must receive (redistribute if below).

    Returns
    -------
    sub_pools : list[dict]
        Each dict has the same schema as the input pool but with a subset
        of originals. The emb/label/axial arrays are shared (not copied)
        since indices reference the same underlying arrays.
    """
    rng = np.random.RandomState(seed)
    originals = sorted(pool["originals"])
    labels = pool["label"]

    # Group originals by class
    cls_to_idx: Dict[str, List[int]] = defaultdict(list)
    for idx in originals:
        cls_to_idx[labels[idx]].append(idx)
    classes = sorted(cls_to_idx.keys())

    # Initialize sub-client bins
    sub_originals: List[List[int]] = [[] for _ in range(n_sub)]

    # For each class, draw a proportion vector from Dir(alpha) and allocate
    for cls in classes:
        idxs = cls_to_idx[cls].copy()
        rng.shuffle(idxs)

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * n_sub)

        # Convert proportions to counts (at least 0 per sub-client)
        counts = (proportions * len(idxs)).astype(int)
        # Distribute remainder to largest-proportion sub-clients
        remainder = len(idxs) - counts.sum()
        top_subs = np.argsort(-proportions)
        for i in range(remainder):
            counts[top_subs[i % n_sub]] += 1

        # Assign indices to sub-clients
        offset = 0
        for s in range(n_sub):
            sub_originals[s].extend(idxs[offset:offset + counts[s]])
            offset += counts[s]

    # Handle sub-clients with too few samples: steal from largest
    for s in range(n_sub):
        while len(sub_originals[s]) < min_samples_per_sub:
            # Find the largest sub-client
            largest = max(range(n_sub), key=lambda i: len(sub_originals[i]))
            if len(sub_originals[largest]) <= min_samples_per_sub:
                break  # Can't steal anymore
            moved = sub_originals[largest].pop()
            sub_originals[s].append(moved)

    # Build sub-pool dicts (share underlying arrays, just different originals)
    sub_pools = []
    for s in range(n_sub):
        orig_set = set(sub_originals[s])
        # Filter augs_by_axial to only include axial paths whose parent is in this sub-client
        sub_augs = {}
        sub_idx_by_axial = {}
        axial_arr = pool["axial"]
        for idx in sub_originals[s]:
            ax = axial_arr[idx] if hasattr(axial_arr, '__getitem__') else str(axial_arr[idx])
            ax = str(ax)
            if ax in pool.get("augs_by_axial", {}):
                sub_augs[ax] = pool["augs_by_axial"][ax]
            if ax in pool.get("idx_by_axial", {}):
                sub_idx_by_axial[ax] = pool["idx_by_axial"][ax]

        sub_pools.append({
            "emb": pool["emb"],           # shared reference
            "label": pool["label"],        # shared reference
            "axial": pool["axial"],        # shared reference
            "originals": sorted(sub_originals[s]),
            "augs_by_axial": sub_augs,
            "idx_by_axial": sub_idx_by_axial,
        })

    return sub_pools


def partition_all_clients(
    pools: Dict[str, dict],
    n_clients_total: int,
    alpha: float,
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Partition all real client pools into n_clients_total simulated clients.

    Distributes the target client count proportionally across real clients
    based on their pool sizes, with a minimum of 1 sub-client per real client.

    Parameters
    ----------
    pools : dict[str, dict]
        Real client pools from load_clients_from_csvs.
    n_clients_total : int
        Total number of simulated clients desired.
    alpha : float
        Dirichlet concentration parameter.
    seed : int
        Random seed.

    Returns
    -------
    sim_pools : dict[str, dict]
        Simulated client pools. Keys are "{real_client_id}_{sub_idx}".
    """
    real_ids = sorted(pools.keys())
    n_real = len(real_ids)

    if n_clients_total <= n_real:
        # No partitioning needed, just return original pools
        return dict(pools)

    # Distribute sub-clients proportionally by pool size
    sizes = {cid: len(pools[cid]["originals"]) for cid in real_ids}
    total_size = sum(sizes.values())

    # Allocate sub-clients proportionally, minimum 1 each
    sub_counts = {}
    remaining = n_clients_total
    for cid in real_ids:
        sub_counts[cid] = max(1, int(round(n_clients_total * sizes[cid] / total_size)))
        remaining -= sub_counts[cid]

    # Adjust to hit exact total
    while remaining > 0:
        # Add to the largest real client
        cid = max(real_ids, key=lambda c: sizes[c])
        sub_counts[cid] += 1
        remaining -= 1
    while remaining < 0:
        # Remove from the real client with most sub-clients
        cid = max(real_ids, key=lambda c: sub_counts[c])
        if sub_counts[cid] > 1:
            sub_counts[cid] -= 1
            remaining += 1
        else:
            break

    # Partition each real client
    sim_pools = {}
    for cid in real_ids:
        n_sub = sub_counts[cid]
        if n_sub == 1:
            sim_pools[f"{cid}_0"] = pools[cid]
        else:
            subs = dirichlet_partition(pools[cid], n_sub, alpha, seed=seed)
            for i, sp in enumerate(subs):
                sim_pools[f"{cid}_{i}"] = sp

    return sim_pools
