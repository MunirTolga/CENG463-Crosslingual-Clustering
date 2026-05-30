from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.kmeans import run_kmeans
from data.preprocess import load_dataframe, save_dataframe
from embeddings.mbert_embedder import encode_texts_with_mbert
from embeddings.transformer_embedder import save_embeddings
from evaluation.metrics import evaluate_clustering, save_metrics
from utils.io import ensure_dir
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline 3: raw mBERT + K-Means.")
    parser.add_argument("--input-file", default="data/processed/xnli_validation_en_tr.csv")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--embeddings-path", default="outputs/embeddings/mbert_embeddings.npy")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    input_path = ROOT / args.input_file
    output_dir = ensure_dir(ROOT / args.output_dir)
    embeddings_path = ROOT / args.embeddings_path
    metrics_path = output_dir / "mbert_kmeans_metrics.json"
    predictions_path = output_dir / "mbert_kmeans_predictions.csv"

    LOGGER.info("Loading processed data from: %s", input_path)
    dataframe = load_dataframe(input_path)
    _validate_columns(dataframe, args.text_column, args.label_column)

    texts = dataframe[args.text_column].fillna("").astype(str).tolist()
    true_labels = dataframe[args.label_column].to_numpy()

    LOGGER.info("Encoding %d texts with raw mBERT: %s", len(texts), args.model_name)
    embeddings = encode_texts_with_mbert(
        texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        pooling=args.pooling,
    )
    save_embeddings(embeddings, embeddings_path)

    LOGGER.info("Running K-Means with n_clusters=%d", args.n_clusters)
    cluster_labels, _ = run_kmeans(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )

    LOGGER.info("Evaluating clustering results")
    metrics = evaluate_clustering(embeddings, true_labels, cluster_labels)

    predictions = dataframe.copy()
    predictions["cluster_label"] = cluster_labels

    save_metrics(metrics, metrics_path)
    save_dataframe(predictions, predictions_path)

    LOGGER.info("Saved embeddings to: %s", embeddings_path)
    LOGGER.info("Saved metrics to: %s", metrics_path)
    LOGGER.info("Saved predictions to: %s", predictions_path)


def _validate_columns(dataframe, text_column: str, label_column: str) -> None:
    missing_columns = [
        column for column in (text_column, label_column) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")


if __name__ == "__main__":
    main()
