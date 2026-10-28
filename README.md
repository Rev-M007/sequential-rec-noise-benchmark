# Sequential Rec Noise Benchmark

Comparative analysis of deep sequential recommenders under interaction noise
(MovieLens-1M). Built with PyTorch and RecBole.

## Main files

| File | Role |
|------|------|
| `train_model.py` | **Use this** — train any model on clean or noisy data |
| `inject_noise.py` | Build noisy datasets (also called automatically by `train_model.py`) |
| `fmlp_rec.py` | FMLPRec model |
| `sig4rec.py` | Sig4Rec model |
| `log_results.py` | Append / summarise runs in `results/results.csv` |
| `setup_environment.sh` | Install dependencies (local or Colab) |
| `requirements.txt` | Python packages |
| `train_baseline.py` | Older SASRec-only script (kept for ml-1m download helper). Prefer `train_model.py`. |

## Setup

```bash
chmod +x setup_environment.sh
./setup_environment.sh
source .venv/bin/activate   # local machine only; skip on Colab
```

Or: `pip install -r requirements.txt`  
(RecBole is installed with `--no-deps` in the setup script to avoid Ray issues.)

## Run on a local GPU

```bash
# Clean MovieLens-1M (downloads on first run)
python train_model.py --model SASRec --dataset ml-1m --gpu_id 0

# Noisy data (created automatically if missing)
python train_model.py --model SASRec --dataset ml-1m-ins20 --gpu_id 0
```

**Models:** `SASRec`, `GRU4Rec`, `BERT4Rec`, `FMLPRec`, `Sig4Rec`

**Noise names:** `ml-1m-ins10` / `ins20` / `ins30`, `ml-1m-del10` / `del20` / `del30`  
(insertion = random extra clicks; deletion = missing history)

Optional: generate all noisy sets at once:

```bash
python inject_noise.py --all
```

## Default settings

- Sequence length: **200**
- Embedding size: **64**
- Train batch size: **2048**
- Metrics: **NDCG@10**, **Recall@10**, **Hit@10** (also @20)
- Max epochs: 200 (early stopping on validation NDCG@10)

## Results

Each finished run is appended to `results/results.csv`.

```bash
python log_results.py summary --csv results/results.csv --metric ndcg@10
```

## Notes for reviewers

- Datasets, logs, and checkpoints are **not** in git (see `.gitignore`). They are created when you run training.
- Use **`train_model.py`** for all experiments. `train_baseline.py` is legacy.
