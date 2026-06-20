# CENG463-Crosslingual-Clustering

Clean project skeleton for the CENG463 Machine Learning term project:
Supervised Contrastive Cross-lingual Clustering.

## Installation

Run these commands from the project root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```text
configs/
data/
  raw/
  processed/
  splits/
notebooks/
src/
  data/
  embeddings/
  baselines/
  models/
  training/
  clustering/
  evaluation/
  experiments/
outputs/
  embeddings/
  checkpoints/
  metrics/
  figures/
  reports/
scripts/
```

## Notes

- Main implementation code should live under `src/`.
- Reproducible command-line entry points should live under `scripts/`.
- Notebooks should be used only for exploration.
- Data files and generated outputs should not be committed by default.
- Model logic has not been implemented yet.

## Recommended Commands

Run all eight baseline configurations:

```powershell
python -m scripts.run_baselines
```

Use local model cache only for SentenceTransformer baselines:

```powershell
python -m scripts.run_baselines --local-files-only
```

Run mSimCSE when you have selected a compatible model:

```powershell
python -m scripts.run_baselines --msimcse_model_name MODEL_NAME
```

Run individual new baselines:

```powershell
python -m src.baselines.run_tfidf_svd_kmeans
python -m src.baselines.run_minilm_agglomerative
python -m src.baselines.run_labse_kmeans
```

CPU-safe quick pipeline:

```powershell
python -m scripts.run_final_experiment --quick
```

CPU-safe full pipeline:

```powershell
python -m scripts.run_final_experiment --epochs 1 --batch_size 2 --num_workers 0 --torch_num_threads 1 --gradient_accumulation_steps 2
```

Optional longer SupCon training:

```powershell
python -m src.training.train_supcon --train_file data/processed/xnli_train.csv --epochs 3 --batch_size 2 --num_workers 0 --torch_num_threads 1 --gradient_accumulation_steps 4
```
