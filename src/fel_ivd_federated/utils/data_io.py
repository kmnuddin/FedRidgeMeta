# at top
import os
import numpy as np
import pandas as pd
from typing import Dict
from collections import defaultdict

def _load_vec_from_path(p):
    p = str(p)
    ext = os.path.splitext(p)[1].lower()

    if ext == ".npy":
        # DO NOT use mmap_mode here — load into RAM then free the FD
        arr = np.load(p, allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)

    elif ext == ".npz":
        # Use a context manager so the file handle is closed immediately
        with np.load(p, allow_pickle=False) as npz:
            # choose a key deterministically
            for k in ("emb", "embedding", "arr_0"):
                if k in npz:
                    arr = npz[k]
                    break
            else:
                # fallback to first key
                k0 = list(npz.keys())[0]
                arr = npz[k0]
            arr = np.asarray(arr, dtype=np.float32)

    elif ext in (".pt", ".pth"):
        import torch
        # If your .pt is big, open via file handle to ensure closure
        with open(p, "rb") as f:
            t = torch.load(f, map_location="cpu")
        if hasattr(t, "detach"):
            arr = t.detach().cpu().numpy().astype(np.float32)
        elif isinstance(t, dict):
            for k in ("emb", "embedding"):
                if k in t:
                    v = t[k]
                    arr = (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)).astype(np.float32)
                    break
            else:
                arr = np.asarray(t, dtype=np.float32)
        else:
            arr = np.asarray(t, dtype=np.float32)

    else:
        # last-resort: parse string of numbers
        s = p
        if "," in s and " " not in s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.fromstring(s, sep=" ", dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Loaded zero-length embedding from {p}")
    return arr

def load_clients_from_csvs(csv_map: Dict[str,str],
                           label_col: str,
                           id_col: str,
                           emb_col: str,
                           axial_col: str = "axial_path",
                           is_aug_col: str = "is_aug",
                           aug_idx_col: str = "aug_idx"):
    out = {}
    for cname, path in csv_map.items():
        df = pd.read_csv(path)
        if emb_col not in df.columns:
            raise ValueError(f"{cname}: expected embedding column '{emb_col}' in {path}")
        if axial_col not in df.columns:
            raise ValueError(f"{cname}: expected axial column '{axial_col}' in {path}")
        if label_col not in df.columns:
            raise ValueError(f"{cname}: expected label column '{label_col}' in {path}")

        # --- LOAD EMBEDDINGS FROM FILE PATHS ---
        emb_list = []
        dim = None
        for i, pth in enumerate(df[emb_col].tolist()):
            vec = _load_vec_from_path(pth)
            if dim is None:
                dim = int(vec.shape[0])
            elif vec.shape[0] != dim:
                raise ValueError(
                    f"{cname}: inconsistent embedding dim at row {i}: "
                    f"expected {dim}, got {vec.shape[0]} from {pth}"
                )
            emb_list.append(vec)
        emb = np.stack(emb_list, axis=0).astype(np.float32)
        # --------------------------------------

        y = df[label_col].astype(str).to_numpy()
        axial = df[axial_col].astype(str).tolist()

        by_ax = defaultdict(list)
        for i, ax in enumerate(axial):
            by_ax[ax].append(i)

        originals = []
        augs_by_axial = defaultdict(list)
        idx_by_axial = {}

        has_is_aug = is_aug_col in df.columns
        has_aug_idx = aug_idx_col in df.columns

        for ax, idxs in by_ax.items():
            if has_is_aug:
                origs = [i for i in idxs if int(df.loc[i, is_aug_col]) == 0]
                orig_idx = origs[0] if len(origs) else idxs[0]
            else:
                orig_idx = idxs[0]
            originals.append(orig_idx)
            idx_by_axial[ax] = orig_idx

            aug_candidates = [i for i in idxs if i != orig_idx]
            if has_is_aug:
                aug_candidates = [i for i in aug_candidates if int(df.loc[i, is_aug_col]) == 1]
            if has_aug_idx:
                aug_candidates = sorted(aug_candidates, key=lambda i: int(df.loc[i, aug_idx_col]))
            augs_by_axial[ax] = aug_candidates

        out[cname] = dict(
            df=df, emb=emb, label=y, axial=axial,
            originals=originals, augs_by_axial=augs_by_axial,
            idx_by_axial=idx_by_axial
        )
    return out
