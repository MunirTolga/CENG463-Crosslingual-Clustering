from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.kmeans import run_kmeans
from data.preprocess import load_dataframe, save_dataframe
from embeddings.transformer_embedder import (
    encode_texts_with_sentence_transformer,
    load_embeddings,
    save_embeddings,
)
from evaluation.metrics import evaluate_clustering, save_metrics
from utils.io import ensure_dir
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LaBSE + K-Means.")
    parser.add_argument("--input-file", default="data/processed/xnli_validation.csv")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--model-name", default="sentence-transformers/LaBSE")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-files-only", "--local_files_only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    input_path = ROOT / args.input_file
    output_dir = ensure_dir(ROOT / args.output_dir)
    embeddings_dir = ensure_dir(ROOT / args.embeddings_dir)
    embeddings_path = embeddings_dir / "labse_embeddings.npy"
    metrics_path = output_dir / "labse_kmeans_metrics.json"
    predictions_path = output_dir / "labse_kmeans_predictions.csv"

    LOGGER.info("Loading processed data from: %s", input_path)
    dataframe = load_dataframe(input_path)
    _validate_columns(dataframe, args.text_column, args.label_column)

    texts = dataframe[args.text_column].fillna("").astype(str).tolist()
    true_labels = dataframe[args.label_column].to_numpy()

    embeddings = _load_or_build_embeddings(
        args=args,
        texts=texts,
        embeddings_path=embeddings_path,
        expected_count=len(dataframe),
    )

    LOGGER.info("Running K-Means with n_clusters=%d", args.n_clusters)
    cluster_labels, _ = run_kmeans(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )

    LOGGER.info("Evaluating clustering results")
    metrics = evaluate_clustering(embeddings, true_labels, cluster_labels)

    predictions = dataframe.copy()
    predictions["true_label"] = true_labels
    predictions["cluster_label"] = cluster_labels

    save_metrics(metrics, metrics_path)
    save_dataframe(predictions, predictions_path)

    LOGGER.info("Saved embeddings to: %s", embeddings_path)
    LOGGER.info("Saved metrics to: %s", metrics_path)
    LOGGER.info("Saved predictions to: %s", predictions_path)


def _load_or_build_embeddings(
    args: argparse.Namespace,
    texts: list[str],
    embeddings_path: Path,
    expected_count: int,
):
    if embeddings_path.exists() and not args.force:
        LOGGER.info("Reusing cached LaBSE embeddings from: %s", embeddings_path)
        embeddings = load_embeddings(embeddings_path)
        if len(embeddings) == expected_count:
            return embeddings
        LOGGER.warning(
            "Cached LaBSE embeddings have %d rows, but input data has %d rows. "
            "Recomputing embeddings.",
            len(embeddings),
            expected_count,
        )

    LOGGER.info("Encoding %d texts with model: %s", len(texts), args.model_name)
    embeddings = encode_texts_with_sentence_transformer(
        texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=True,
        local_files_only=args.local_files_only,
        offline=args.offline,
    )
    save_embeddings(embeddings, embeddings_path)
    return embeddings


def _validate_columns(dataframe, text_column: str, label_column: str) -> None:
    missing_columns = [
        column for column in (text_column, label_column) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")


if __name__ == "__main__":
    main()
