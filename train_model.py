"""
train_model.py
==============

Train and evaluate a RecBole sequential model on a clean or noisy dataset, then
append the result to ``results/results.csv`` (see ``log_results.py``).
Built-in models run through RecBole's ``run_recbole`` quick-start API; models
implemented in this project (``CUSTOM_MODELS``: FMLPRec, Sig4Rec) go through
``run_custom``, which performs the same steps.

All models share the same data processing, leave-one-out split, full-ranking
evaluation and embedding size, so differences between runs come from the model
and the noise condition only.

Usage
-----
    python train_model.py --model SASRec --dataset ml-1m
    python train_model.py --model SASRec --dataset ml-1m-ins20 --seed 2027
    python train_model.py --model GRU4Rec --dataset ml-1m-del10
    python train_model.py --model BERT4Rec --dataset ml-1m --epochs 300
    python train_model.py --model FMLPRec --dataset ml-1m-ins30
    python train_model.py --model Sig4Rec --dataset ml-1m-del20   # BPR loss by default

Noisy datasets (e.g. ``ml-1m-ins20``) are generated automatically via
``inject_noise.ensure_noisy_dataset`` when missing. Clean ``ml-1m`` is
downloaded automatically.
"""

# =============================================================================
# Compatibility patches — MUST run before RecBole / PyTorch / scientific imports
# (Google Colab: Python 3.13 + NumPy 2.x + no Ray + TensorBoard/TF quirks)
# =============================================================================
import sys
import types
import ssl
import os

# A. Bypass SSL verification for online dataset downloads (GroupLens / Colab)
ssl._create_default_https_context = ssl._create_unverified_context

# B. Restore NumPy aliases removed in 2.x (RecBole 1.2.1 still references them).
#    Only set names that are missing. Never overwrite np.bool_ / other live dtypes
#    with Python builtins — that turns bool masks into object arrays and breaks
#    pandas row filtering on Colab (KeyError on Index([True, True, ...])).
import numpy as np
for _name, _alias in (
    ("float_", np.float64),
    ("float", np.float64),
    ("int_", np.int64),
    ("int", np.int64),
    ("complex_", np.complex128),
    ("complex", np.complex128),
    ("unicode_", str),
    ("unicode", str),
    ("bool", bool),  # np.bool was removed; np.bool_ must remain the dtype
):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

# C. Mock Ray stubs to prevent ModuleNotFoundError when RecBole imports ray.tune
#    (we install recbole with --no-deps to avoid Ray resolution crashes)
ray = types.ModuleType("ray")
tune = types.ModuleType("tune")
ray.tune = tune
sys.modules["ray"] = ray
sys.modules["ray.tune"] = tune

# D. Redirect PyTorch TensorBoard calls to tensorboardX to prevent TensorFlow
#    C++ crashes on Python 3.13 / Colab
try:
    import tensorboardX
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = tensorboardX.SummaryWriter
    sys.modules["torch.utils.tensorboard"] = tb
except ImportError:
    pass

# E. Patch pandas dtype_to_subclass for boolean indexers in RecBole filtering.
#    Keep the unbound original (__func__) so non-bool dtypes (DatetimeIndex,
#    CategoricalIndex, …) still resolve correctly on pandas 2.x / Colab.
import pandas as pd
_orig_dtype_subclass = pd.core.indexes.base.Index._dtype_to_subclass
_orig_dtype_subclass_func = getattr(_orig_dtype_subclass, "__func__", _orig_dtype_subclass)


@classmethod
def _patched_dtype_subclass(cls, dtype):
    if str(dtype) in ["bool", "bool8", "boolean"]:
        return pd.core.indexes.base.Index
    try:
        return _orig_dtype_subclass_func(cls, dtype)
    except Exception:
        return pd.core.indexes.base.Index


pd.core.indexes.base.Index._dtype_to_subclass = _patched_dtype_subclass

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot unpickle the
# config objects RecBole stores in its checkpoints, breaking best-model reloading.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# =============================================================================
# Normal imports (safe after the patches above)
# =============================================================================
import argparse
import importlib
from logging import FileHandler, getLogger

import torch

# Full CUDA hardware acceleration on Colab T4 / local GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import run_recbole
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color

from log_results import DEFAULT_CSV, log_result, parse_dataset_name, parse_recbole_log

METRICS = ["NDCG", "Recall", "Hit"]
TOPK = [10, 20]

# Custom models: RecBole-style split between code (models/) and YAML (props/model/).
# run_recbole only knows RecBole's built-in models, so these use run_custom().
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROPS_MODEL_DIR = os.path.join(_REPO_ROOT, "props", "model")
CUSTOM_MODELS = {
    "FMLPRec": ("models.fmlp_rec", "FMLPRec"),
    "Sig4Rec": ("models.sig4rec", "Sig4Rec"),
}
# One negative item per positive, as required by the BPR loss.
BPR_NEG_SAMPLING = {"distribution": "uniform", "sample_num": 1, "alpha": 1.0,
                    "dynamic": False, "candidate_num": 0}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RecBole sequential model and log its test metrics."
    )
    parser.add_argument("--model", default="SASRec",
                        help="RecBole model name (SASRec, BERT4Rec, GRU4Rec, ...), FMLPRec or Sig4Rec.")
    parser.add_argument("--dataset", default="ml-1m",
                        help="Dataset folder under --data_path, e.g. ml-1m or ml-1m-ins20.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--MAX_ITEM_LIST_LENGTH", type=int, default=200,
                        help="Maximum history length fed to the model.")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--embedding_size", type=int, default=64,
                        help="Item embedding / hidden dimension, applied to every model.")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--loss_type", choices=["CE", "BPR"], default=None,
                        help="Override the model's default loss (CE for most models, BPR for Sig4Rec).")
    parser.add_argument("--stopping_step", type=int, default=10,
                        help="Early-stopping patience in epochs (validation NDCG@10).")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--data_path", default="./dataset/")
    parser.add_argument("--results_csv", default=DEFAULT_CSV)
    return parser.parse_args()


def model_yaml_path(name: str) -> str:
    """Path to this project's RecBole-style model YAML (``props/model/<Name>.yaml``)."""
    return os.path.join(PROPS_MODEL_DIR, f"{name}.yaml")


def load_model_yaml(name: str) -> dict:
    """Read ``props/model/<Name>.yaml`` (hyperparameters only; code stays in ``models/``)."""
    import yaml
    path = model_yaml_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model YAML: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_custom_model(name: str):
    """Import a project model class from ``models/`` (lazy import)."""
    module_name, class_name = CUSTOM_MODELS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def model_params(model: str, dim: int) -> dict:
    """Map the shared embedding size onto each model's own parameter names.

    Custom-model architecture defaults come from ``props/model/<Model>.yaml``.
    Built-in models use RecBole's ``properties/model/<Model>.yaml``.
    """
    if model in CUSTOM_MODELS:
        # Size only — other hyperparams stay in the YAML file.
        return {"embedding_size": dim, "hidden_size": dim, "inner_size": dim * 4}
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
    """Shared RecBole settings for every model and dataset.

    Model-specific hyperparams for FMLPRec / Sig4Rec are NOT duplicated here;
    they are loaded from ``props/model/*.yaml`` via ``config_file_list``.
    """
    use_gpu = torch.cuda.is_available()
    if args.model in CUSTOM_MODELS:
        loss_type = args.loss_type or load_model_yaml(args.model).get("loss_type", "CE")
    else:
        loss_type = args.loss_type or "CE"

    config = {
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
        "loss_type": loss_type,

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
        # 4096 keeps full-ranking eval saturated on a T4 without OOM.
        "eval_batch_size": 4096,

        # ---- Model size (YAML supplies the rest for custom models) ------ #
        **model_params(args.model, args.embedding_size),
    }
    # CE scores all items, so RecBole requires negative sampling off; BPR needs one negative.
    config["train_neg_sample_args"] = BPR_NEG_SAMPLING if config["loss_type"] == "BPR" else None
    if args.model == "Sig4Rec":
        config["relation_file"] = relation_file(args)
    return config


def relation_file(args: argparse.Namespace) -> str:
    """Item-attribute file shared by a dataset and all its noisy copies (same items)."""
    base = parse_dataset_name(args.dataset)[0]
    return os.path.join(args.data_path, base, f"{base}.genre")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dataset(args: argparse.Namespace) -> None:
    """Make sure the .inter (and Sig4Rec's relation) file exists before RecBole runs.

    Clean ``ml-1m`` is downloaded on demand. Noisy names like ``ml-1m-ins20``
    are generated automatically via ``inject_noise.ensure_noisy_dataset``.
    """
    if args.model == "Sig4Rec" and not os.path.exists(relation_file(args)):
        if parse_dataset_name(args.dataset)[0] != "ml-1m":
            raise SystemExit(f"{relation_file(args)} not found; Sig4Rec needs item relations.")
        importlib.import_module("models.sig4rec").prepare_ml1m_genres(args.data_path)

    path = os.path.join(args.data_path, args.dataset, f"{args.dataset}.inter")
    if os.path.exists(path):
        return
    if args.dataset == "ml-1m":
        from train_baseline import prepare_ml1m
        prepare_ml1m(args.data_path)
        return

    # Auto-build noisy variants (ml-1m-ins10, ml-1m-del20, ...) for Colab one-shot runs.
    try:
        from inject_noise import ensure_noisy_dataset
        ensure_noisy_dataset(args.dataset, data_path=args.data_path, seed=args.seed)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def run_custom(args: argparse.Namespace) -> dict:
    """Same steps as RecBole's run_recbole, but for a model class defined in this project.

    Hyperparameters come from ``props/model/<Model>.yaml`` (like RecBole's
    ``properties/model/``), then shared benchmark settings in ``config_dict``.
    """
    model_class = load_custom_model(args.model)
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=[model_yaml_path(args.model)],
        config_dict=build_config_dict(args),
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"])
    test_result = trainer.evaluate(test_data, load_best_model=True,
                                   show_progress=config["show_progress"])

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return {"best_valid_score": best_valid_score, "best_valid_result": best_valid_result,
            "test_result": test_result}


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
    print(f"[config] train_batch_size={args.train_batch_size}  "
          f"eval_batch_size=4096  MAX_ITEM_LIST_LENGTH={args.MAX_ITEM_LIST_LENGTH}",
          flush=True)

    ensure_dataset(args)
    if args.model in CUSTOM_MODELS:
        result = run_custom(args)
    else:
        result = run_recbole(model=args.model, dataset=args.dataset,
                             config_dict=build_config_dict(args), saved=True)

    print_test_results(args.model, args.dataset, result["test_result"])
    print(f"  Best validation NDCG@10: {result['best_valid_score']:.4f}")
    append_to_csv(args, result)


if __name__ == "__main__":
    main()
