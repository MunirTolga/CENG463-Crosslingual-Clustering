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
