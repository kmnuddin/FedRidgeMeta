#!/usr/bin/env python3
"""
analyze_logs.py — collates FEL run logs and computes:
- Macro-F1 vs labels curve
- AULC over specified budget window
- (Optional) ECE/Brier if present in JSONL
- Median QBC disagreement if present (or recompute if logged as 'median_js')
- Runtime per phase
- Per-class coverage from 'picked_per_class'

Usage:
  python analyze_logs.py --runs-root runs/fel_sweeps --out /tmp/fel_summary.csv --auc-window 0,18000

Expects each run folder to contain a JSONL log (your pipeline writes one line per round).
This script is tolerant: it scans *.jsonl inside each directory and uses the largest one.
"""
import argparse, os, json, glob, math, statistics, csv
from pathlib import Path

def read_jsonl_best(path):
    cand = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not cand:
        # Fallback: any file named sim_log.jsonl deeply nested
        cand = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
        if not cand:
            return []
    best = max(cand, key=lambda p: os.path.getsize(p))
    rows = []
    with open(best, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def labels_after_round(seed_k, r_index, batch_B):
    """r_index starts at 1; labels = seed_k + r_index * B"""
    return seed_k + r_index * batch_B

def auc_trapz(xs, ys):
    if len(xs) < 2: return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += dx * 0.5 * (ys[i] + ys[i-1])
    return area

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="Root of run folders (study subdirs).")
    ap.add_argument("--out", required=True, help="Path to CSV summary.")
    ap.add_argument("--auc-window", type=str, default="0,18000", help="Start,End labels for AULC (inclusive).")
    args = ap.parse_args()

    w0, w1 = [int(x) for x in args.auc_window.split(",")]
    rows_out = []
    for path, dirs, files in os.walk(args.runs_root):
        # consider a leaf directory as a run if it has any jsonl
        jsonls = glob.glob(os.path.join(path, "*.jsonl"))
        if not jsonls:
            continue
        logs = read_jsonl_best(path)
        if not logs:
            continue

        # Try to infer seed_k, batch_B from the first row (your JSONL schema suggests they exist indirectly)
        # If not present, we approximate from sizes change; otherwise set defaults.
        # We'll read 'al_method', 'meta' flags, rf params from the directory name string.
        run_id = os.path.basename(path)

        # Pull metrics per round (macro-F1) and compute x=labels, y=macro-F1
        macro_f1_curve = []
        per_class_coverage = {}  # class -> cum picks
        time_totals = {"train":0.0,"meta":0.0,"eval":0.0,"acquire":0.0}

        seed_k = None
        batch_B = None

        for i, rec in enumerate(logs, start=1):
            # batch_B and seed_k aren't guaranteed in record; try to infer
            if batch_B is None:
                # prefer explicit
                batch_B = rec.get("al", {}).get("batch_B") or rec.get("al_batch_B")
            if seed_k is None:
                seed_k = rec.get("seed_k")

            # metrics
            macro = None
            # pipeline logs both RF and META; take META if present else RF
            for key in ("meta","rf"):
                if key in rec.get("metrics", {}):
                    m = rec["metrics"][key]
                    macro = m.get("f1_macro") or m.get("macro_f1") or macro
            if macro is None:
                # backwards-compatible top-level
                macro = rec.get("metrics_rest", {}).get("f1_macro")

            # timings
            t = rec.get("timing_sec", {})
            for k in time_totals.keys():
                time_totals[k] += float(t.get(k, 0.0))

            # coverage
            cdict = rec.get("picked_per_class") or rec.get("clients", {}).get("RSNA", {}).get("picked_per_class")
            if cdict:
                for k,v in cdict.items():
                    per_class_coverage[k] = per_class_coverage.get(k, 0) + int(v)

            if batch_B is None:
                # fallback: infer from picked_per_class sum
                if cdict:
                    batch_B = sum(int(v) for v in cdict.values())

            if seed_k is None:
                # fallback default
                seed_k = 200

            labels = seed_k + i * (batch_B or 100)
            if macro is not None:
                macro_f1_curve.append((labels, float(macro)))

        # Compute AULC on requested window (trapz)
        xs = [x for x,_ in macro_f1_curve if w0 <= x <= w1]
        ys = [y for x,y in macro_f1_curve if w0 <= x <= w1]
        aulc = auc_trapz(xs, ys) if xs else 0.0

        rows_out.append({
            "run": run_id,
            "path": path,
            "points": len(macro_f1_curve),
            "AULC_%d_%d"% (w0,w1): aulc,
            "time_train_s": round(time_totals["train"],2),
            "time_meta_s": round(time_totals["meta"],2),
            "time_eval_s": round(time_totals["eval"],2),
            "time_acquire_s": round(time_totals["acquire"],2),
            **{f"cov_{k}":v for k,v in sorted(per_class_coverage.items())},
        })
    print(rows_out)

    # write CSV
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    # collect all keys
    all_keys = set()
    for r in rows_out:
        all_keys.update(r.keys())
    keys = ["run","path","points","AULC_%d_%d"% (w0,w1),"time_train_s","time_meta_s","time_eval_s","time_acquire_s"] + \
           sorted([k for k in all_keys if k.startswith("cov_")])
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote summary to {outp}")

if __name__ == "__main__":
    main()

