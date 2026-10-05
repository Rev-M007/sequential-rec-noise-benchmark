"""
train_baseline.py
=================

Train and evaluate a baseline SASRec model on MovieLens-1M (``ml-1m``) with RecBole.

This is the clean-data reference point for the project
"Comparative Analysis and Benchmarking of Deep Sequential Recommender Systems
Under Interaction Noise and Signal-Processing Noise Reduction Techniques".

Pipeline
--------
1. Build a RecBole ``Config`` (model, dataset, training and evaluation settings).
2. Load ``ml-1m``. On first run the raw data is downloaded from GroupLens and
   converted to RecBole's atomic ``.inter`` format (RecBole's own download
   mirror on Amazon S3 currently returns HTTP 403).
3. Split each user's chronologically ordered sequence with leave-one-out:
   last item -> test, second-to-last -> validation, the rest -> training.
4. Train SASRec with early stopping on validation NDCG@10.
5. Evaluate the best checkpoint on the test set with full ranking
   (every item is a candidate) and report NDCG / Recall / Hit @10 and @20.

Usage
-----
    pip install recbole torch
    python train_baseline.py                      # default settings
    python train_baseline.py --epochs 5 --seed 7  # quick run / different seed
    python train_baseline.py --help               # all options

Note: with leave-one-out there is exactly one ground-truth item per user, so
Recall@K and Hit@K are mathematically identical. Both are reported because they
are standard RecBole metrics.
"""

import argparse
import io
import json
import os
import urllib.request
import zipfile
from datetime import datetime
from logging import FileHandler, getLogger

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot unpickle the
# config objects RecBole stores in its checkpoints, breaking load_best_model=True.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color

from log_results import log_result, parse_recbole_log


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def prepare_ml1m(data_path: str) -> None:
    """Download ml-1m from GroupLens and write it as a RecBole atomic file.

    Produces ``<data_path>/ml-1m/ml-1m.inter`` with the same columns as RecBole's
    processed version: user_id, item_id, rating, timestamp. Skipped if the file
    already exists.
    """
    out_dir = os.path.join(data_path, "ml-1m")
    out_file = os.path.join(out_dir, "ml-1m.inter")
    if os.path.exists(out_file):
        return

    print(f"[data] Downloading ml-1m from {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    # ratings.dat lines look like "UserID::MovieID::Rating::Timestamp".
    raw = archive.read("ml-1m/ratings.dat").decode("latin-1")
    os.makedirs(out_dir, exist_ok=True)
    n_rows = 0
    with open(out_file, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for line in raw.splitlines():
            if line.strip():
                f.write("\t".join(line.split("::")) + "\n")
                n_rows += 1
    print(f"[data] Wrote {n_rows:,} interactions to {out_file}")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Command-line options for the most commonly tuned settings."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a SASRec baseline on ml-1m with RecBole."
    )
    parser.add_argument("--dataset", default="ml-1m", help="RecBole dataset name.")
    parser.add_argument("--data-path", default="./dataset/", help="Where datasets are stored/downloaded.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--stopping-step", type=int, default=10,
                        help="Early stopping patience (epochs without validation improvement).")
    parser.add_argument("--max-seq-len", type=int, default=200,
                        help="Maximum history length fed to the model (200 is standard for ml-1m).")
    parser.add_argument("--hidden-size", type=int, default=64, help="Item embedding / hidden dimension.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility.")
    parser.add_argument("--gpu-id", default="0", help="CUDA device id used when a GPU is available.")
    parser.add_argument("--results-dir", default="./results/", help="Where the final metrics JSON is written.")
    return parser.parse_args()


def build_config_dict(args: argparse.Namespace, use_gpu: bool) -> dict:
    """Assemble the RecBole parameter dictionary for SASRec."""
    return {
        # ---- Environment ------------------------------------------------- #
        "data_path": args.data_path,
        "checkpoint_dir": "./saved/",
        "use_gpu": use_gpu,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "reproducibility": True,
        "show_progress": False,  # per-epoch summary lines only, no tqdm bars

        # ---- Data -------------------------------------------------------- #
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        # Ratings are ignored: every interaction is treated as implicit feedback.
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        # 5-core filtering keeps preprocessing identical to the Amazon datasets.
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "MAX_ITEM_LIST_LENGTH": args.max_seq_len,

        # ---- SASRec architecture ---------------------------------------- #
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": args.hidden_size,
        "inner_size": args.hidden_size * 4,  # feed-forward width
        "hidden_dropout_prob": 0.2,
        "attn_dropout_prob": 0.2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        # Cross-entropy over all items; RecBole requires negative sampling off for CE.
        "loss_type": "CE",
        "train_neg_sample_args": None,

        # ---- Training ---------------------------------------------------- #
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "learner": "adam",
        "learning_rate": args.lr,
        "stopping_step": args.stopping_step,
        "eval_step": 1,  # validate after every epoch

        # ---- Evaluation -------------------------------------------------- #
        "eval_args": {
            "split": {"LS": "valid_and_test"},  # leave-one-out
            "order": "TO",                       # chronological order
            "group_by": "user",
            "mode": "full",                      # rank against all items, no sampling
        },
        "metrics": ["NDCG", "Recall", "Hit"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        "eval_batch_size": 2048,
    }


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def epoch_loss(loss) -> float:
    """RecBole stores the epoch loss summed over batches; multi-loss models store a tuple."""
    return float(sum(loss) if isinstance(loss, tuple) else loss)


def print_loss_history(trainer: Trainer, n_batches: int) -> None:
    """Print one line per epoch with the summed and per-batch mean training loss."""
    print("\n" + "=" * 44)
    print(" Training loss per epoch")
    print("=" * 44)
    print(f"  {'epoch':>5} | {'total loss':>12} | {'mean / batch':>12}")
    for epoch, loss in sorted(trainer.train_loss_dict.items()):
        total = epoch_loss(loss)
        print(f"  {epoch + 1:>5d} | {total:>12.4f} | {total / n_batches:>12.4f}")


def print_test_results(test_result: dict, best_valid_score: float) -> None:
    """Print test metrics as a small table: one row per metric, one column per K."""
    metrics = ["ndcg", "recall", "hit"]
    cutoffs = [10, 20]

    print("\n" + "=" * 44)
    print(" Final test results (best validation checkpoint)")
    print("=" * 44)
    print(f"  {'Metric':<10}" + "".join(f"{'@' + str(k):>12}" for k in cutoffs))
    print("  " + "-" * (10 + 12 * len(cutoffs)))
    for metric in metrics:
        row = "".join(f"{test_result[f'{metric}@{k}']:>12.4f}" for k in cutoffs)
        print(f"  {metric.upper():<10}{row}")
    print(f"\n  Best validation NDCG@10: {best_valid_score:.4f}")
    print("=" * 44)


def save_results(args: argparse.Namespace, test_result: dict, best_valid_score: float,
                 loss_history: dict) -> str:
    """Write metrics to JSON so later scripts can aggregate runs across seeds/noise levels."""
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"SASRec_{args.dataset}_seed{args.seed}.json")
    payload = {
        "model": "SASRec",
        "dataset": args.dataset,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "best_valid_ndcg@10": float(best_valid_score),
        "test": {k: float(v) for k, v in test_result.items()},
        "train_loss_total": {int(e) + 1: epoch_loss(l) for e, l in loss_history.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(int(args.gpu_id)) if use_gpu else "CPU"
    print(f"[device] CUDA available: {use_gpu} -> running on {device_name}", flush=True)

    # 1. Configuration + logging (RecBole logs to the console and to ./log/).
    config = Config(model="SASRec", dataset=args.dataset,
                    config_dict=build_config_dict(args, use_gpu))
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # 2. Dataset: fetched into data_path on first run if missing.
    if args.dataset == "ml-1m":
        prepare_ml1m(args.data_path)
    dataset = create_dataset(config)
    logger.info(dataset)

    # 3. Leave-one-out split into train / validation / test loaders.
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 4. Model. Re-seed so weight initialisation does not depend on data loading.
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = SASRec(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # 5. Train with early stopping on validation NDCG@10; best weights are saved.
    trainer = Trainer(config, model)
    best_valid_score, _ = trainer.fit(train_data, valid_data, saved=True,
                                      show_progress=config["show_progress"])

    # 6. Reload the best checkpoint and evaluate on the held-out test items.
    test_result = trainer.evaluate(test_data, load_best_model=True,
                                   show_progress=config["show_progress"])

    logger.info(set_color("test result", "yellow") + f": {dict(test_result)}")
    print_loss_history(trainer, n_batches=len(train_data))
    print_test_results(test_result, best_valid_score)
    path = save_results(args, test_result, best_valid_score, trainer.train_loss_dict)
    print(f"\nMetrics saved to {path}", flush=True)

    # 7. Append this run to the shared experiment CSV (see log_results.py).
    log_files = [h.baseFilename for h in logger.handlers if isinstance(h, FileHandler)]
    if log_files:
        csv_path = os.path.join(args.results_dir, "results.csv")
        if log_result(csv_path, parse_recbole_log(log_files[0])):
            print(f"Run appended to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
