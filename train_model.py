"""
train_model.py
==============

Train and evaluate any built-in RecBole sequential model on a clean or noisy
dataset with RecBole's ``run_recbole`` quick-start API, then append the result
to ``results/results.csv`` (see ``log_results.py``).

All models share the same data processing, leave-one-out split, full-ranking
evaluation and embedding size, so differences between runs come from the model
and the noise condition only.

Usage
-----
    python train_model.py --model SASRec --dataset ml-1m
    python train_model.py --model SASRec --dataset ml-1m-ins20 --seed 2027
    python train_model.py --model GRU4Rec --dataset ml-1m-del10
    python train_model.py --model BERT4Rec --dataset ml-1m --epochs 300

Noisy datasets (e.g. ``ml-1m-ins20``) must first be created with
``python inject_noise.py``. Clean ``ml-1m`` is downloaded automatically.
"""

import argparse
import os
from logging import FileHandler, getLogger

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot unpickle the
# config objects RecBole stores in its checkpoints, breaking best-model reloading.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from recbole.quick_start import run_recbole

from log_results import DEFAULT_CSV, log_result, parse_recbole_log

METRICS = ["NDCG", "Recall", "Hit"]
TOPK = [10, 20]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RecBole sequential model and log its test metrics."
    )
    parser.add_argument("--model", default="SASRec",
                        help="RecBole model name, e.g. SASRec, BERT4Rec, GRU4Rec.")
    parser.add_argument("--dataset", default="ml-1m",
                        help="Dataset folder under --data_path, e.g. ml-1m or ml-1m-ins20.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--MAX_ITEM_LIST_LENGTH", type=int, default=200,
                        help="Maximum history length fed to the model.")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=64,
                        help="Item embedding / hidden dimension, applied to every model.")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--stopping_step", type=int, default=10,
                        help="Early-stopping patience in epochs (validation NDCG@10).")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--data_path", default="./dataset/")
    parser.add_argument("--results_csv", default=DEFAULT_CSV)
    return parser.parse_args()


def model_params(model: str, dim: int) -> dict:
    """Map the shared embedding size onto each model's own parameter names.

    Settings not listed here fall back to RecBole's per-model defaults
    (recbole/properties/model/<Model>.yaml).
    """
    if model in ("SASRec", "BERT4Rec"):
        params = {"hidden_size": dim, "inner_size": dim * 4}
        if model == "SASRec":
            # RecBole's default dropout (0.5) is tuned for sparse data; 0.2 is
            # the value the SASRec paper used for MovieLens-1M.
            params.update({"hidden_dropout_prob": 0.2, "attn_dropout_prob": 0.2})
        return params
    # GRU4Rec and most other models: item embedding plus a hidden state.
    return {"embedding_size": dim, "hidden_size": dim}


def build_config_dict(args: argparse.Namespace) -> dict:
    """Shared RecBole settings for every model and dataset."""
    use_gpu = torch.cuda.is_available()
    return {
        # ---- Environment ------------------------------------------------- #
        "data_path": args.data_path,
        "checkpoint_dir": "./saved/",
        "use_gpu": use_gpu,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "reproducibility": True,
        "show_progress": False,

        # ---- Data (identical for clean and noisy datasets) -------------- #
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "MAX_ITEM_LIST_LENGTH": args.MAX_ITEM_LIST_LENGTH,

        # ---- Training ---------------------------------------------------- #
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "learner": "adam",
        "learning_rate": args.learning_rate,
        "stopping_step": args.stopping_step,
        "eval_step": 1,
        # Cross-entropy over all items; RecBole requires negative sampling off for CE.
        "loss_type": "CE",
        "train_neg_sample_args": None,

        # ---- Evaluation -------------------------------------------------- #
        "eval_args": {
            "split": {"LS": "valid_and_test"},  # leave-one-out
            "order": "TO",                       # chronological
            "group_by": "user",
            "mode": "full",                      # rank against all items
        },
        "metrics": METRICS,
        "topk": TOPK,
        "valid_metric": "NDCG@10",
        "eval_batch_size": 2048,

        # ---- Model ------------------------------------------------------- #
        **model_params(args.model, args.embedding_size),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dataset(args: argparse.Namespace) -> None:
    """Make sure the .inter file exists before RecBole tries its (broken) download."""
    path = os.path.join(args.data_path, args.dataset, f"{args.dataset}.inter")
    if os.path.exists(path):
        return
    if args.dataset == "ml-1m":
        from train_baseline import prepare_ml1m
        prepare_ml1m(args.data_path)
        return
    raise SystemExit(f"{path} not found. Create it first, e.g.\n"
                     f"  python inject_noise.py --type insertion --level 0.20")


def print_test_results(model: str, dataset: str, test_result: dict) -> None:
    """Print test metrics as a table: one row per metric, one column per K."""
    print("\n" + "=" * 46)
    print(f" Test results: {model} on {dataset}")
    print("=" * 46)
    print(f"  {'Metric':<10}" + "".join(f"{'@' + str(k):>12}" for k in TOPK))
    print("  " + "-" * (10 + 12 * len(TOPK)))
    for metric in METRICS:
        row = "".join(f"{test_result[f'{metric.lower()}@{k}']:>12.4f}" for k in TOPK)
        print(f"  {metric.upper():<10}{row}")
    print("=" * 46, flush=True)


def append_to_csv(args: argparse.Namespace, result: dict) -> None:
    """Build the CSV row from RecBole's log file (config, epochs) and returned metrics."""
    log_files = [h.baseFilename for h in getLogger().handlers if isinstance(h, FileHandler)]
    record = parse_recbole_log(log_files[0]) if log_files else None
    if record is None:
        print("Warning: could not read the RecBole log; result not added to the CSV.")
        return
    record.update({k: float(v) for k, v in result["test_result"].items()})
    record["best_valid_score"] = float(result["best_valid_score"])
    if log_result(args.results_csv, record):
        print(f"Run appended to {args.results_csv}  (run_id: {record['run_id']})", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    device = torch.cuda.get_device_name(int(args.gpu_id)) if torch.cuda.is_available() else "CPU"
    print(f"[device] CUDA available: {torch.cuda.is_available()} -> running on {device}", flush=True)

    ensure_dataset(args)
    result = run_recbole(model=args.model, dataset=args.dataset,
                         config_dict=build_config_dict(args), saved=True)

    print_test_results(args.model, args.dataset, result["test_result"])
    print(f"  Best validation NDCG@10: {result['best_valid_score']:.4f}")
    append_to_csv(args, result)


if __name__ == "__main__":
    main()
