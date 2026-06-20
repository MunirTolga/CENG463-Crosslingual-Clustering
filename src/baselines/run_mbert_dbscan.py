from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.dbscan import run_dbscan
from clustering.preprocess import preprocess_embeddings_for_clustering
from data.preprocess import load_dataframe, save_dataframe
from embeddings.mbert_embedder import encode_texts_with_mbert
from embeddings.transformer_embedder import load_embeddings, save_embeddings
from evaluation.metrics import evaluate_clustering, save_metrics
from utils.io import ensure_dir
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline 4: raw mBERT + DBSCAN.")
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
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--reducer", choices=["none", "pca"], default="none")
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)

    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize", dest="normalize_embeddings", action="store_true")
    normalize_group.add_argument("--no-normalize", dest="normalize_embeddings", action="store_false")
    parser.set_defaults(normalize_embeddings=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    input_path = ROOT / args.input_file
    output_dir = ensure_dir(ROOT / args.output_dir)
    embeddings_path = ROOT / args.embeddings_path
    metrics_path = output_dir / "mbert_dbscan_metrics.json"
    predictions_path = output_dir / "mbert_dbscan_predictions.csv"

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

    processed_embeddings, preprocessing_info = preprocess_embeddings_for_clustering(
        embeddings,
        normalize_embeddings=args.normalize_embeddings,
        reducer=args.reducer,
        pca_components=args.pca_components,
        random_state=args.random_state,
    )

    LOGGER.info("Original embedding shape: %s", tuple(preprocessing_info["original_shape"]))
    LOGGER.info("Processed embedding shape: %s", tuple(preprocessing_info["processed_shape"]))
    LOGGER.info("Running DBSCAN with eps=%s and min_samples=%s", args.eps, args.min_samples)
    LOGGER.info("DBSCAN metric: %s", args.metric)
    LOGGER.info("Normalize embeddings: %s", args.normalize_embeddings)
    LOGGER.info("Reducer: %s", args.reducer)
    cluster_labels, _ = run_dbscan(
        processed_embeddings,
        eps=args.eps,
        min_samples=args.min_samples,
        metric=args.metric,
    )

    LOGGER.info("Evaluating clustering results")
    metrics = evaluate_clustering(processed_embeddings, true_labels, cluster_labels)
    metrics.update(
        {
            "dbscan_eps": args.eps,
            "dbscan_min_samples": args.min_samples,
            "dbscan_metric": args.metric,
            "normalize_embeddings": args.normalize_embeddings,
            "reducer": args.reducer,
            "pca_components": args.pca_components,
            "pca_components_used": preprocessing_info["pca_components_used"],
            "original_embedding_shape": preprocessing_info["original_shape"],
            "processed_embedding_shape": preprocessing_info["processed_shape"],
            "valid_clustering": (
                metrics["number_of_clusters"] >= 2 and metrics["noise_ratio"] < 1.0
            ),
        }
    )
    if metrics["number_of_clusters"] == 0:
        LOGGER.warning(
            "DBSCAN produced no valid clusters. All points may have been assigned as noise. "
            "Try larger eps, lower min_samples, cosine metric, normalization, or PCA."
        )

    predictions = dataframe.copy()
    predictions["cluster_label"] = cluster_labels

    save_metrics(metrics, metrics_path)
    save_dataframe(predictions, predictions_path)

    LOGGER.info("Saved metrics to: %s", metrics_path)
    LOGGER.info("Saved predictions to: %s", predictions_path)
    LOGGER.info("Number of clusters: %s", metrics["number_of_clusters"])
    LOGGER.info("Noise ratio: %.4f", metrics["noise_ratio"])


def _load_or_build_embeddings(
    args: argparse.Namespace,
    texts: list[str],
    embeddings_path: Path,
    expected_count: int,
):
    if embeddings_path.exists():
        LOGGER.info("Loading cached mBERT embeddings from: %s", embeddings_path)
        embeddings = load_embeddings(embeddings_path)
        if len(embeddings) == expected_count:
            return embeddings
        LOGGER.warning(
            "Cached mBERT embeddings have %d rows, but input data has %d rows. "
            "Recomputing embeddings.",
            len(embeddings),
            expected_count,
        )

    LOGGER.info("Encoding %d texts with mBERT.", len(texts))
    embeddings = encode_texts_with_mbert(
        texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        pooling=args.pooling,
    )
    save_embeddings(embeddings, embeddings_path)
    LOGGER.info("Saved new mBERT embeddings to: %s", embeddings_path)
    return embeddings


def _validate_columns(dataframe, text_column: str, label_column: str) -> None:
    missing_columns = [
        column for column in (text_column, label_column) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")


if __name__ == "__main__":
    main()
