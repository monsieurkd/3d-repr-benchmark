"""Aggregate results/*.json into a comparison table.

Reads every run's JSON and prints:
  - all runs (raw), and
  - a focused dev-scope comparison (one row per representation).

The headline metrics for the research question are steady_epoch_sec (train) and
infer_us_per_sample (inference); accuracy is reported but secondary.
"""
import glob
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"

COLS = [
    ("rep",                 "{}",  6),
    ("classes",             "{:>3}", 3),
    ("params",              "{:>9,}", 9),
    ("preprocess_train_s",  "{:>7.1f}", 7),
    ("steady_epoch_s",      "{:>8.2f}", 8),
    ("total_train_s",       "{:>8.1f}", 8),
    ("test_acc",            "{:>7.3f}", 7),
    ("infer_us/sample",     "{:>9.1f}", 9),
]
KEYS = [c[0] for c in COLS]


def fmt(row):
    out = {}
    for k, spec, _ in COLS:
        v = row.get(k)
        if k == "params":
            out[k] = spec.format(int(v)) if v is not None else "-"
        elif k in ("preprocess_train_s",):
            out[k] = spec.format(row.get("preprocess_train_sec", 0)) if row.get("preprocess_train_sec") is not None else "-"
        elif k == "steady_epoch_s":
            out[k] = spec.format(row.get("steady_epoch_sec", 0))
        elif k == "total_train_s":
            out[k] = spec.format(row.get("total_train_sec", 0))
        elif k == "infer_us/sample":
            out[k] = spec.format(row.get("infer_us_per_sample", 0))
        elif k == "test_acc":
            out[k] = spec.format(v) if v is not None else "-"
        else:
            out[k] = spec.format(v) if v is not None else "-"
    return out


def print_table(rows, title):
    print(f"\n=== {title} ===")
    header = "  ".join(f"{c[0]:>{c[2]}}" for c in COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        f = fmt(r)
        print("  ".join(f[k] for k in KEYS))


def main():
    runs = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        d["_file"] = Path(f).name
        runs.append(d)

    if not runs:
        print("no results yet (run scripts/run_baseline.py first)")
        return

    # all runs, most recent last
    print_table(runs, f"all runs ({len(runs)})")

    # focused: best (most epochs) dev-scope run per rep
    dev = [r for r in runs if "_dev_" in r["_file"] or "dev" in r["_file"]]
    if not dev:
        # fall back to highest class count per rep
        dev = runs
    best = {}
    for r in dev:
        rep = r.get("rep")
        if rep not in best or r.get("epochs", 0) > best[rep].get("epochs", 0):
            best[rep] = r
    if len(best) > 1:
        print_table(sorted(best.values(), key=lambda r: r["rep"]),
                    "dev comparison (best run per representation)")


if __name__ == "__main__":
    main()
