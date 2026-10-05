"""
log_results.py
==============

Lightweight experiment logger: collects RecBole evaluation results into one CSV
(``results/results.csv``, one row per training run) and summarises them across
models, datasets, noise conditions and seeds.

Noise conditions are read from the dataset name, so corrupted datasets must be
named ``<base>-<type><percent>``:

    ml-1m          -> clean (noise_type "none", noise_level 0)
    ml-1m-ins20    -> 20% random item insertions
    ml-1m-del10    -> 10% random deletions
    ml-1m-mix30    -> 30% mixed insertions + deletions

Usage
-----
    # Import every finished run found in RecBole's ./log/ folder (safe to repeat;
    # runs already in the CSV are skipped).
    python log_results.py import-logs

    # Mean ± std per model / dataset / noise condition, plus the relative drop
    # from the clean result.
    python log_results.py summary
    python log_results.py summary --metric hit@10

    # From Python (train_baseline.py does this automatically after each run):
    from log_results import parse_recbole_log, log_result
    log_result("results/results.csv", parse_recbole_log(path_to_log))
"""

import argparse
import ast
import csv
import glob
import os
import re
from datetime import datetime
from typing import Optional

DEFAULT_CSV = os.path.join("results", "results.csv")
METRICS = ["ndcg@10", "recall@10", "hit@10", "ndcg@20", "recall@20", "hit@20"]
COLUMNS = [
    "run_id", "timestamp", "model", "dataset", "base_dataset", "noise_type",
    "noise_level", "seed", "max_seq_len", "hidden_size", "learning_rate",
    "epochs_run", "best_epoch", "best_valid_score", *METRICS,
]
NOISE_TYPES = {"ins": "insertion", "del": "deletion", "mix": "mixed"}

_NOISE_SUFFIX = re.compile(r"^(?P<base>.+)-(?P<type>ins|del|mix)(?P<pct>\d{1,3})$")
_CONFIG_LINE = re.compile(r"^(?P<key>\w+) = (?P<value>.*)$")
_EPOCH_LINE = re.compile(r"epoch (?P<epoch>\d+) training \[")
_VALID_LINE = re.compile(r"epoch (?P<epoch>\d+) evaluating \[.*valid_score: (?P<score>[\d.]+)\]")
# Matches both "{'ndcg@10': ...}" (train_baseline.py) and
# "OrderedDict([('ndcg@10', ...)])" (RecBole's run_recbole).
_TEST_LINE = re.compile(r"test result\S*\s*: (?:OrderedDict\((?P<pairs>\[.*\])\)|(?P<dict>\{.*\}))")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_dataset_name(dataset: str) -> tuple:
    """Split a dataset name into (base_dataset, noise_type, noise_level)."""
    match = _NOISE_SUFFIX.match(dataset)
    if not match:
        return dataset, "none", 0
    return match["base"], NOISE_TYPES[match["type"]], int(match["pct"])


def parse_recbole_log(log_path: str) -> Optional[dict]:
    """Extract one result row from a RecBole log file.

    Returns None for runs that never reached test evaluation (crashed or still
    running), so they are never mistaken for real results.
    """
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    config, train_epochs, valid_scores, test_result = {}, set(), {}, None
    for line in lines:
        if (m := _CONFIG_LINE.match(line)) and m["key"] not in config:
            config[m["key"]] = m["value"].strip()
        if m := _EPOCH_LINE.search(line):
            train_epochs.add(int(m["epoch"]))
        if m := _VALID_LINE.search(line):
            valid_scores[int(m["epoch"])] = float(m["score"])
        if m := _TEST_LINE.search(line):
            test_result = (dict(ast.literal_eval(m["pairs"])) if m["pairs"]
                           else ast.literal_eval(m["dict"]))

    if test_result is None:
        return None

    # RecBole names logs "<model>-<dataset>-<date>-<hash>.log".
    model = os.path.basename(log_path).split("-", 1)[0]
    dataset = os.path.basename(config.get("data_path", "").rstrip("/")) or "unknown"
    base, noise_type, noise_level = parse_dataset_name(dataset)
    best_epoch = max(valid_scores, key=valid_scores.get) if valid_scores else None

    # The first line starts with "Mon 05 Oct 2026 17:35:54 INFO".
    try:
        timestamp = datetime.strptime(lines[0][:24], "%a %d %b %Y %H:%M:%S").isoformat()
    except (ValueError, IndexError):
        timestamp = datetime.fromtimestamp(os.path.getmtime(log_path)).isoformat(timespec="seconds")

    return {
        "run_id": os.path.basename(log_path),
        "timestamp": timestamp,
        "model": model,
        "dataset": dataset,
        "base_dataset": base,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "seed": config.get("seed"),
        "max_seq_len": config.get("MAX_ITEM_LIST_LENGTH"),
        "hidden_size": config.get("hidden_size"),
        "learning_rate": config.get("learning_rate"),
        "epochs_run": len(train_epochs),
        # RecBole counts epochs from 0; report them from 1.
        "best_epoch": best_epoch + 1 if best_epoch is not None else None,
        "best_valid_score": valid_scores.get(best_epoch),
        **{metric: test_result.get(metric) for metric in METRICS},
    }


# --------------------------------------------------------------------------- #
# CSV storage
# --------------------------------------------------------------------------- #
def _existing_run_ids(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def log_result(csv_path: str, record: Optional[dict]) -> bool:
    """Append one result row to the CSV. Returns False if skipped (None or duplicate)."""
    if record is None or record["run_id"] in _existing_run_ids(csv_path):
        return False
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)
    return True


def import_logs(log_dir: str, csv_path: str) -> None:
    """Scan a RecBole log folder and add every finished, not-yet-logged run."""
    paths = sorted(glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True))
    added = skipped = incomplete = 0
    for path in paths:
        record = parse_recbole_log(path)
        if record is None:
            incomplete += 1
        elif log_result(csv_path, record):
            added += 1
            print(f"  + {record['model']:<10} {record['dataset']:<14} seed={record['seed']}  "
                  f"NDCG@10={record['ndcg@10']:.4f}  ({os.path.basename(path)})")
        else:
            skipped += 1
    print(f"\n{added} added, {skipped} already logged, {incomplete} incomplete/crashed "
          f"(of {len(paths)} logs) -> {csv_path}")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(csv_path: str, metric: str) -> None:
    """Print mean ± std over seeds and the relative drop versus the clean dataset."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    keys = ["model", "base_dataset", "noise_type", "noise_level"]
    grouped = df.groupby(keys)[metric].agg(["mean", "std", "count"]).reset_index()

    clean = (grouped[grouped["noise_type"] == "none"]
             .set_index(["model", "base_dataset"])["mean"])
    grouped["drop_vs_clean_%"] = [
        100 * (clean[(m, d)] - mean) / clean[(m, d)] if (m, d) in clean.index else float("nan")
        for m, d, mean in zip(grouped["model"], grouped["base_dataset"], grouped["mean"])
    ]
    grouped[f"{metric} (mean ± std)"] = [
        f"{mean:.4f} ± {std:.4f}" if count > 1 else f"{mean:.4f}"
        for mean, std, count in zip(grouped["mean"], grouped["std"], grouped["count"])
    ]
    table = grouped[keys + [f"{metric} (mean ± std)", "count", "drop_vs_clean_%"]]
    table = (table.rename(columns={"count": "n_runs"})
             .sort_values(["model", "base_dataset", "noise_level", "noise_type"]))

    print(f"\nSummary of {metric} from {csv_path}\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and summarise RecBole results.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-logs", help="Add finished runs from RecBole log files.")
    p_import.add_argument("--log-dir", default="log")
    p_import.add_argument("--csv", default=DEFAULT_CSV)

    p_summary = sub.add_parser("summary", help="Mean ± std per model and noise condition.")
    p_summary.add_argument("--csv", default=DEFAULT_CSV)
    p_summary.add_argument("--metric", default="ndcg@10", choices=METRICS)

    args = parser.parse_args()
    if args.command == "import-logs":
        import_logs(args.log_dir, args.csv)
    else:
        summarize(args.csv, args.metric)


if __name__ == "__main__":
    main()
