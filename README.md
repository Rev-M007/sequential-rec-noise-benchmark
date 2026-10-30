# Sequential Rec Noise Benchmark

Comparative analysis of deep sequential recommenders under interaction noise
(MovieLens-1M). Built with PyTorch and RecBole.

## Main files

| File / folder | Role |
|------|------|
| `train_model.py` | **Use this** — train any model on clean or noisy data |
| `inject_noise.py` | Build noisy datasets (also called automatically by `train_model.py`) |
| `models/` | Custom model **code** (`fmlp_rec.py`, `sig4rec.py`) — like RecBole’s `model/` |
| `props/model/` | Custom model **YAML** (`FMLPRec.yaml`, `Sig4Rec.yaml`) — like RecBole’s `properties/model/` |
| `log_results.py` | Append / summarise runs in `results/results.csv` |
| `setup_environment.sh` | Install dependencies (local or Colab) |
| `requirements.txt` | Python packages |
| `train_baseline.py` | Older SASRec-only script (kept for ml-1m download helper). Prefer `train_model.py`. |

Custom models follow RecBole’s split: network in `.py`, hyperparameters in `.yaml`.

## Setup (local GPU machine)

```bash
git clone https://github.com/Rev-M007/sequential-rec-noise-benchmark.git
cd sequential-rec-noise-benchmark

chmod +x setup_environment.sh
./setup_environment.sh
source .venv/bin/activate
```

Or: `pip install -r requirements.txt`  
(RecBole is installed with `--no-deps` in the setup script to avoid Ray issues.)

Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run on a local GPU

Default batch size is **1024**. Use `--gpu_id 0` (or `1`, `2`, … if you have more GPUs).

```bash
# Clean MovieLens-1M (downloads on first run)
python train_model.py --model SASRec --dataset ml-1m --gpu_id 0

# Same with an explicit batch size
python train_model.py --model SASRec --dataset ml-1m --gpu_id 0 --train_batch_size 1024

# Larger batch (if GPU memory allows)
python train_model.py --model SASRec --dataset ml-1m --gpu_id 0 --train_batch_size 2048

# Noisy data (created automatically if missing)
python train_model.py --model SASRec --dataset ml-1m-ins20 --gpu_id 0
python train_model.py --model SASRec --dataset ml-1m-del10 --gpu_id 0
```

### All models (clean ml-1m)

```bash
python train_model.py --model SASRec   --dataset ml-1m --gpu_id 0 --train_batch_size 1024
python train_model.py --model GRU4Rec  --dataset ml-1m --gpu_id 0 --train_batch_size 1024
python train_model.py --model BERT4Rec --dataset ml-1m --gpu_id 0 --train_batch_size 1024
python train_model.py --model FMLPRec  --dataset ml-1m --gpu_id 0 --train_batch_size 1024
python train_model.py --model Sig4Rec  --dataset ml-1m --gpu_id 0 --train_batch_size 1024
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
- Train batch size: **1024**
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
