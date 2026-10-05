"""
inject_noise.py
===============

Create corrupted copies of a RecBole sequential dataset (default: ``ml-1m``)
with controlled synthetic interaction noise:

* **insertion** ("accidental clicks"): random items the user never interacted
  with are inserted at random positions of the training history.
* **deletion** ("missing history"): random training interactions are removed.

The last ``--protect-last`` interactions of every user (default 2: the
validation and test targets under RecBole's leave-one-out split) are never
modified, and no item is ever inserted that duplicates one of the user's
existing interactions.

Noise level
-----------
``--level p`` is relative to each user's *training* history (all interactions
except the protected ones). With n training interactions, p * n items are
inserted or deleted (stochastic rounding keeps the expected count exact even
for short sequences). For insertion this means p=0.20 adds 20 noisy items per
100 genuine ones, so noisy items make up p / (1 + p) ~ 16.7% of the result.

Consistency with RecBole
------------------------
* The same k-core filter used for training (``user/item_inter_num_interval =
  [5,inf)``) is applied to the clean data *before* corruption, so the protected
  items are exactly the ones RecBole will later hold out.
* Deletions never push a user below ``--min-train-len`` training items or an
  item below the k-core threshold, so RecBole's filter removes nothing extra.
* Inserted rows copy the timestamp of the interaction just before them and are
  written in sequence order; RecBole's stable timestamp sort keeps that order.
  Sequences are ordered with RecBole's float32 timestamp precision (see
  ``sort_sequences``) so "last interaction" means the same thing in both.
* Inserted rows get ``rating = 0`` (real ml-1m ratings are 1-5), which marks
  them for later analysis. Training ignores ratings.

Output
------
``dataset/<source>-ins20/<source>-ins20.inter`` (or ``-del10`` etc.) plus a
``noise_meta.json`` with the settings and statistics. These names are what
``log_results.py`` parses to recover the noise type and level.

Usage
-----
    python inject_noise.py --type insertion --level 0.20
    python inject_noise.py --type deletion  --level 0.10 --seed 7
    python inject_noise.py --all            # insertion + deletion at 10/20/30%
    python train_baseline.py --dataset ml-1m-ins20
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

NOISE_TAGS = {"insertion": "ins", "deletion": "del"}
STANDARD_LEVELS = [0.10, 0.20, 0.30]


# --------------------------------------------------------------------------- #
# Loading & preprocessing
# --------------------------------------------------------------------------- #
def find_column(df: pd.DataFrame, name: str) -> str:
    """Return the full RecBole header for a field, e.g. 'item_id' -> 'item_id:token'."""
    for col in df.columns:
        if col.split(":")[0] == name:
            return col
    raise KeyError(f"Column '{name}' not found in {list(df.columns)}")


def load_inter(data_path: str, source: str) -> pd.DataFrame:
    """Load ``<data_path>/<source>/<source>.inter``, downloading ml-1m if needed."""
    path = os.path.join(data_path, source, f"{source}.inter")
    if not os.path.exists(path):
        if source != "ml-1m":
            raise FileNotFoundError(f"{path} not found.")
        from train_baseline import prepare_ml1m  # imported lazily: pulls in torch/RecBole
        prepare_ml1m(data_path)

    df = pd.read_csv(path, sep="\t", dtype=str)
    ts_col = find_column(df, "timestamp")
    df[ts_col] = pd.to_numeric(df[ts_col])
    return df


def k_core(df: pd.DataFrame, uid: str, iid: str, k: int) -> pd.DataFrame:
    """Iteratively keep users and items with >= k interactions (mirrors RecBole)."""
    while True:
        user_ok = df[uid].map(df[uid].value_counts()) >= k
        item_ok = df[iid].map(df[iid].value_counts()) >= k
        keep = user_ok & item_ok
        if keep.all():
            return df
        df = df[keep]


def sort_sequences(df: pd.DataFrame, uid: str, ts: str) -> pd.DataFrame:
    """Order each user's interactions exactly as RecBole will.

    RecBole loads ``:float`` fields as float32, which only resolves Unix
    timestamps to ~64 s, then sorts stably. Interactions closer together than
    that keep their file order, so the sort key must be float32 too.
    """
    key = df[ts].astype(np.float32)
    return (df.assign(_ts32=key)
              .sort_values([uid, "_ts32"], kind="stable")
              .drop(columns="_ts32")
              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Noise operations (one user at a time)
# --------------------------------------------------------------------------- #
def stochastic_round(x: float, rng: np.random.Generator) -> int:
    """Round x up with probability equal to its fractional part (unbiased)."""
    base = int(np.floor(x))
    return base + int(rng.random() < x - base)


def sample_new_items(k: int, exclude: set, pool: np.ndarray, probs, rng) -> list:
    """Draw k distinct items from pool that are not in exclude."""
    chosen, chosen_set = [], set()
    while len(chosen) < k:
        for item in rng.choice(pool, size=2 * k, p=probs):
            if item not in exclude and item not in chosen_set:
                chosen.append(item)
                chosen_set.add(item)
                if len(chosen) == k:
                    break
    return chosen


def insert_noise(seq: pd.DataFrame, n_train: int, level: float, cols: dict,
                 pool: np.ndarray, probs, rng) -> pd.DataFrame:
    """Insert random unseen items into the first n_train positions of one user's sequence."""
    k = stochastic_round(level * n_train, rng)
    if k == 0:
        return seq

    # Slot j means "just before original position j"; slot n_train is right
    # before the first protected item, which is still training history.
    slots = rng.integers(0, n_train + 1, size=k)
    items = sample_new_items(k, set(seq[cols["iid"]]), pool, probs, rng)
    ts_values = seq[cols["ts"]].to_numpy()

    noise = pd.DataFrame({col: [None] * k for col in seq.columns})
    noise[cols["uid"]] = seq[cols["uid"]].iloc[0]
    noise[cols["iid"]] = items
    noise[cols["ts"]] = ts_values[np.maximum(slots - 1, 0)]
    if cols["rating"]:
        noise[cols["rating"]] = "0"

    # Order key: originals at j, inserted rows at slot - 0.5 (i.e. just before j).
    original = seq.assign(_order=np.arange(len(seq), dtype=float))
    noise = noise.assign(_order=slots - 0.5)
    merged = pd.concat([original, noise], ignore_index=True)
    return merged.sort_values("_order", kind="stable").drop(columns="_order")


def delete_noise(seq: pd.DataFrame, n_train: int, level: float, cols: dict,
                 item_counts: dict, k_min: int, min_train_len: int, rng) -> pd.DataFrame:
    """Delete random training interactions while respecting the k-core constraints."""
    k = min(stochastic_round(level * n_train, rng), max(n_train - min_train_len, 0))
    if k == 0:
        return seq

    items = seq[cols["iid"]].to_numpy()
    dropped = []
    for pos in rng.permutation(n_train):
        if len(dropped) == k:
            break
        if item_counts[items[pos]] > k_min:
            item_counts[items[pos]] -= 1
            dropped.append(pos)
    return seq.drop(index=seq.index[dropped])


# --------------------------------------------------------------------------- #
# Dataset-level pipeline
# --------------------------------------------------------------------------- #
def corrupt(clean: pd.DataFrame, cols: dict, noise_type: str, level: float, args,
            rng: np.random.Generator) -> pd.DataFrame:
    """Apply one noise type to every user's training history."""
    item_counts = clean[cols["iid"]].value_counts()
    pool = item_counts.index.to_numpy()
    probs = (item_counts / item_counts.sum()).to_numpy() if args.sampling == "popularity" else None
    counts = item_counts.to_dict()

    groups = dict(tuple(clean.groupby(cols["uid"], sort=False)))
    # Random user order so the item-count limit in deletion does not favour any user.
    users = rng.permutation(list(groups))
    corrupted = {}
    for user in users:
        seq = groups[user]
        n_train = len(seq) - args.protect_last
        if n_train <= 0:
            corrupted[user] = seq
        elif noise_type == "insertion":
            corrupted[user] = insert_noise(seq, n_train, level, cols, pool, probs, rng)
        else:
            corrupted[user] = delete_noise(seq, n_train, level, cols, counts,
                                           args.k_core, args.min_train_len, rng)

    # Write users in their original order so the file stays sorted by user.
    return pd.concat([corrupted[u] for u in groups], ignore_index=True)


def verify(clean: pd.DataFrame, noisy: pd.DataFrame, cols: dict, protect_last: int,
           k: int) -> None:
    """Fail loudly if corruption touched protected items or would be altered by RecBole."""
    uid, iid, ts = cols["uid"], cols["iid"], cols["ts"]
    resorted = sort_sequences(noisy, uid, ts)
    if not resorted[[uid, iid, ts]].equals(noisy[[uid, iid, ts]].reset_index(drop=True)):
        raise AssertionError("RecBole's timestamp sort would reorder the noisy sequences.")

    tail_clean = clean.groupby(uid, sort=False).tail(protect_last)[[uid, iid, ts]]
    tail_noisy = resorted.groupby(uid, sort=False).tail(protect_last)[[uid, iid, ts]]
    if not tail_clean.reset_index(drop=True).equals(tail_noisy.reset_index(drop=True)):
        raise AssertionError("Protected (validation/test) interactions were modified.")

    if len(k_core(noisy, uid, iid, k)) != len(noisy):
        raise AssertionError(f"RecBole's {k}-core filter would drop rows from the noisy data.")


def describe(df: pd.DataFrame, cols: dict) -> dict:
    lengths = df.groupby(cols["uid"]).size()
    return {"users": int(lengths.size), "items": int(df[cols["iid"]].nunique()),
            "interactions": int(len(df)), "avg_seq_len": round(float(lengths.mean()), 2),
            "min_seq_len": int(lengths.min())}


def run(noise_type: str, level: float, clean: pd.DataFrame, cols: dict, args) -> None:
    """Create, verify and save one corrupted dataset."""
    name = f"{args.source}-{NOISE_TAGS[noise_type]}{round(level * 100)}"
    rng = np.random.default_rng(args.seed)

    noisy = corrupt(clean, cols, noise_type, level, args, rng)
    verify(clean, noisy, cols, args.protect_last, args.k_core)

    n_train = len(clean) - clean[cols["uid"]].nunique() * args.protect_last
    n_changed = abs(len(noisy) - len(clean))
    before, after = describe(clean, cols), describe(noisy, cols)

    out_dir = os.path.join(args.data_path, name)
    os.makedirs(out_dir, exist_ok=True)
    noisy.to_csv(os.path.join(out_dir, f"{name}.inter"), sep="\t", index=False)

    meta = {
        "name": name, "source": args.source, "noise_type": noise_type, "level": level,
        "seed": args.seed, "sampling": args.sampling, "protect_last": args.protect_last,
        "min_train_len": args.min_train_len, "k_core": args.k_core,
        "interactions_changed": n_changed,
        "actual_rate_of_training_history": round(n_changed / n_train, 4),
        "clean": before, "noisy": after,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "noise_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    verb = "inserted" if noise_type == "insertion" else "deleted"
    print(f"[{name}] {n_changed:,} interactions {verb} "
          f"({meta['actual_rate_of_training_history']:.1%} of training history) | "
          f"avg seq len {before['avg_seq_len']} -> {after['avg_seq_len']} | "
          f"min seq len {after['min_seq_len']} | checks passed")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject synthetic interaction noise into a RecBole dataset.")
    parser.add_argument("--type", choices=list(NOISE_TAGS), help="Noise type.")
    parser.add_argument("--level", type=float, help="Noise level as a fraction, e.g. 0.20.")
    parser.add_argument("--all", action="store_true",
                        help="Generate insertion and deletion at 10%%, 20%% and 30%%.")
    parser.add_argument("--source", default="ml-1m", help="Clean dataset name.")
    parser.add_argument("--data-path", default="./dataset/", help="RecBole dataset folder.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--sampling", choices=["uniform", "popularity"], default="uniform",
                        help="How inserted items are drawn from the item catalogue.")
    parser.add_argument("--protect-last", type=int, default=2,
                        help="Trailing interactions per user that are never modified.")
    parser.add_argument("--min-train-len", type=int, default=3,
                        help="Minimum training items kept per user under deletion.")
    parser.add_argument("--k-core", type=int, default=5,
                        help="Must match user/item_inter_num_interval used for training.")
    args = parser.parse_args()

    if not args.all and (args.type is None or args.level is None):
        parser.error("give --type and --level, or --all")
    if args.level is not None and not 0 < args.level < 1:
        parser.error("--level must be a fraction between 0 and 1, e.g. 0.20")
    if args.protect_last + args.min_train_len < args.k_core:
        parser.error("--protect-last + --min-train-len must be >= --k-core")
    return args


def main() -> None:
    args = parse_args()

    raw = load_inter(args.data_path, args.source)
    cols = {
        "uid": find_column(raw, "user_id"),
        "iid": find_column(raw, "item_id"),
        "ts": find_column(raw, "timestamp"),
        "rating": next((c for c in raw.columns if c.split(":")[0] == "rating"), None),
    }
    clean = sort_sequences(k_core(raw, cols["uid"], cols["iid"], args.k_core), cols["uid"], cols["ts"])
    print(f"[{args.source}] after {args.k_core}-core filtering: {describe(clean, cols)}")

    jobs = ([(t, lvl) for t in NOISE_TAGS for lvl in STANDARD_LEVELS] if args.all
            else [(args.type, args.level)])
    for noise_type, level in jobs:
        run(noise_type, level, clean, cols, args)


if __name__ == "__main__":
    main()
