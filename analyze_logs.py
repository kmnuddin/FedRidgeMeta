#!/usr/bin/env python3
"""
analyze_logs.py — collates FEL run logs and computes:
- Macro-F1 vs labels curve (on both shrinking-U and fixed test set)
- AULC over specified budget window (for both eval modes)
- Runtime per phase
- Per-class coverage from 'picked_per_class'
- Mode detection (federated vs centralized)

Usage:
  python analyze_logs.py --runs-root runs/fel_sweeps --out /tmp/fel_summary.csv --auc-window 0,18000

Expects each run folder to contain a JSONL log.
"""
import argparse, os, json, glob, math, statistics, csv
from pathlib import Path

def read_jsonl_best(path):
    cand = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not cand:
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
    ap.add_argument("--runs-root", required=True, help="Root of run folders.")
    ap.add_argument("--out", required=True, help="Path to CSV summary.")
    ap.add_argument("--auc-window", type=str, default="0,18000", help="Start,End labels for AULC.")
    args = ap.parse_args()

    w0, w1 = [int(x) for x in args.auc_window.split(",")]
    rows_out = []
    for path, dirs, files in os.walk(args.runs_root):
        jsonls = glob.glob(os.path.join(path, "*.jsonl"))
        if not jsonls:
            continue
        logs = read_jsonl_best(path)
        if not logs:
            continue

        run_id = os.path.basename(path)

        # curves for both eval modes
        macro_f1_curve_U = []       # legacy: evaluated on shrinking U
        macro_f1_curve_test = []    # NEW: evaluated on fixed holdout
        per_class_coverage = {}
        time_totals = {"train":0.0,"meta":0.0,"eval":0.0,"acquire":0.0}

        seed_k = None
        batch_B = None
        mode = "federated"  # default; overridden if log says "centralized"

        for i, rec in enumerate(logs, start=1):
            # skip summary row
            if "summary" in rec:
                if rec["summary"].get("mode") == "centralized":
                    mode = "centralized"
                continue

            if rec.get("mode") == "centralized":
                mode = "centralized"

            if batch_B is None:
                batch_B = rec.get("al", {}).get("batch_B") or rec.get("al_batch_B")
            if seed_k is None:
                seed_k = rec.get("seed_k")

            # --- U-based metrics (legacy) ---
            macro_U = None
            avg = rec.get("avg", {})
            for key in ("meta", "rf"):
                if key in avg:
                    m = avg[key]
                    macro_U = m.get("f1_macro") or m.get("macro_f1") or macro_U

            # --- Fixed-test metrics (NEW) ---
            macro_test = None
            avg_test = rec.get("avg_test", {})
            for key in ("meta", "rf"):
                if key in avg_test:
                    m = avg_test[key]
                    macro_test = m.get("f1_macro") or m.get("macro_f1") or macro_test

            # timings
            t = rec.get("timing_sec", {})
            time_totals["train"] += float(t.get("train", 0.0))
            time_totals["meta"] += float(t.get("meta_refresh", t.get("meta", 0.0)))
            time_totals["eval"] += float(t.get("eval", 0.0))
            time_totals["acquire"] += float(t.get("acquire", 0.0))

            # coverage — handle both federated and centralized formats
            cdict = rec.get("picked_per_class")
            if cdict is None:
                # federated: look inside clients
                for cid_data in rec.get("clients", {}).values():
                    cd = cid_data.get("picked_per_class")
                    if cd:
                        if cdict is None:
                            cdict = {}
                        for k, v in cd.items():
                            cdict[k] = cdict.get(k, 0) + int(v)

            if cdict:
                for k,v in cdict.items():
                    per_class_coverage[k] = per_class_coverage.get(k, 0) + int(v)

            if batch_B is None and cdict:
                batch_B = sum(int(v) for v in cdict.values())

            if seed_k is None:
                seed_k = 200

            labels = seed_k + i * (batch_B or 100)
            if macro_U is not None:
                macro_f1_curve_U.append((labels, float(macro_U)))
            if macro_test is not None:
                macro_f1_curve_test.append((labels, float(macro_test)))

        # AULC on U (legacy)
        xs_U = [x for x,_ in macro_f1_curve_U if w0 <= x <= w1]
        ys_U = [y for x,y in macro_f1_curve_U if w0 <= x <= w1]
        aulc_U = auc_trapz(xs_U, ys_U) if xs_U else 0.0

        # AULC on fixed test (NEW)
        xs_T = [x for x,_ in macro_f1_curve_test if w0 <= x <= w1]
        ys_T = [y for x,y in macro_f1_curve_test if w0 <= x <= w1]
        aulc_test = auc_trapz(xs_T, ys_T) if xs_T else 0.0

        # final test F1 (last recorded value)
        final_test_f1 = macro_f1_curve_test[-1][1] if macro_f1_curve_test else None

        # final calibration metrics (from last record's avg_test)
        final_rf_ece = final_meta_ece = None
        final_rf_brier = final_meta_brier = None
        final_rf_f1 = final_meta_f1 = None
        if logs:
            last_data_recs = [r for r in logs if "summary" not in r]
            last_rec = last_data_recs[-1] if last_data_recs else {}
            last_avg_test = last_rec.get("avg_test", {}) or {}
            last_rf = last_avg_test.get("rf", {}) or {}
            last_me = last_avg_test.get("meta", {}) or {}
            final_rf_ece = last_rf.get("ece")
            final_meta_ece = last_me.get("ece")
            final_rf_brier = last_rf.get("brier")
            final_meta_brier = last_me.get("brier")
            final_rf_f1 = last_rf.get("macro_f1") or last_rf.get("f1_macro")
            final_meta_f1 = last_me.get("macro_f1") or last_me.get("f1_macro")

        row = {
            "run": run_id,
            "path": path,
            "mode": mode,
            "points": len(macro_f1_curve_U),
            "points_test": len(macro_f1_curve_test),
            "AULC_U_%d_%d" % (w0, w1): aulc_U,
            "AULC_test_%d_%d" % (w0, w1): aulc_test,
            "final_test_f1": final_test_f1,
            "final_rf_f1": final_rf_f1,
            "final_meta_f1": final_meta_f1,
            "final_rf_ece": final_rf_ece,
            "final_meta_ece": final_meta_ece,
            "final_rf_brier": final_rf_brier,
            "final_meta_brier": final_meta_brier,
            "time_train_s": round(time_totals["train"], 2),
            "time_meta_s": round(time_totals["meta"], 2),
            "time_eval_s": round(time_totals["eval"], 2),
            "time_acquire_s": round(time_totals["acquire"], 2),
        }
        row.update({f"cov_{k}": v for k, v in sorted(per_class_coverage.items())})
        rows_out.append(row)

    print(f"Found {len(rows_out)} runs.")

    # write CSV
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for r in rows_out:
        all_keys.update(r.keys())

    fixed_keys = [
        "run", "path", "mode", "points", "points_test",
        "AULC_U_%d_%d" % (w0, w1),
        "AULC_test_%d_%d" % (w0, w1),
        "final_test_f1",
        "final_rf_f1", "final_meta_f1",
        "final_rf_ece", "final_meta_ece",
        "final_rf_brier", "final_meta_brier",
        "time_train_s", "time_meta_s", "time_eval_s", "time_acquire_s",
    ]
    cov_keys = sorted([k for k in all_keys if k.startswith("cov_")])
    keys = fixed_keys + cov_keys

    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote summary to {outp}")

if __name__ == "__main__":
    main()
