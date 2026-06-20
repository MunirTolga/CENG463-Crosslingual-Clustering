from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.kmeans import run_kmeans
from data.preprocess import load_dataframe, save_dataframe
from embeddings.transformer_embedder import load_embeddings, save_embeddings
from evaluation.metrics import evaluate_clustering, save_metrics
from utils.io import ensure_dir
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF + SVD/LSA + K-Means.")
    parser.add_argument("--input-file", default="data/processed/xnli_validation.csv")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--svd-components", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    input_path = ROOT / args.input_file
    output_dir = ensure_dir(ROOT / args.output_dir)
    embeddings_dir = ensure_dir(ROOT / args.embeddings_dir)
    embeddings_path = embeddings_dir / "tfidf_svd_embeddings.npy"
    metrics_path = output_dir / "tfidf_svd_kmeans_metrics.json"
    predictions_path = output_dir / "tfidf_svd_kmeans_predictions.csv"

    LOGGER.info("Loading processed data from: %s", input_path)
    dataframe = load_dataframe(input_path)
    _validate_columns(dataframe, args.text_column, args.label_column)

    texts = dataframe[args.text_column].fillna("").astype(str).tolist()
    true_labels = dataframe[args.label_column].to_numpy()

    embeddings = _load_or_build_embeddings(
        texts=texts,
        embeddings_path=embeddings_path,
        max_features=args.max_features,
        svd_components=args.svd_components,
        random_state=args.random_state,
        force=args.force,
    )

    LOGGER.info("Running K-Means with n_clusters=%d", args.n_clusters)
    cluster_labels, _ = run_kmeans(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )

    LOGGER.info("Evaluating clustering results")
    metrics = evaluate_clustering(embeddings, true_labels, cluster_labels)
    metrics.update(
        {
            "max_features": args.max_features,
            "svd_components": int(embeddings.shape[1]),
        }
    )

    predictions = dataframe.copy()
    predictions["true_label"] = true_labels
    predictions["cluster_label"] = cluster_labels

    save_metrics(metrics, metrics_path)
    save_dataframe(predictions, predictions_path)

    LOGGER.info("Saved embeddings to: %s", embeddings_path)
    LOGGER.info("Saved metrics to: %s", metrics_path)
    LOGGER.info("Saved predictions to: %s", predictions_path)


def _load_or_build_embeddings(
    texts: list[str],
    embeddings_path: Path,
    max_features: int,
    svd_components: int,
    random_state: int,
    force: bool,
):
    if embeddings_path.exists() and not force:
        LOGGER.info("Reusing cached TF-IDF SVD embeddings from: %s", embeddings_path)
        embeddings = load_embeddings(embeddings_path)
        if len(embeddings) == len(texts):
            return embeddings
        LOGGER.warning(
            "Cached TF-IDF SVD embeddings have %d rows, but input data has %d rows. "
            "Recomputing embeddings.",
            len(embeddings),
            len(texts),
        )

    LOGGER.info("Building TF-IDF matrix for %d examples", len(texts))
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)

    effective_components = _safe_svd_components(
        requested_components=svd_components,
        n_rows=tfidf_matrix.shape[0],
        n_features=tfidf_matrix.shape[1],
    )
    LOGGER.info("Applying TruncatedSVD with n_components=%d", effective_components)
    svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
    embeddings = svd.fit_transform(tfidf_matrix)

    save_embeddings(embeddings, embeddings_path)
    return embeddings


def _safe_svd_components(
    requested_components: int,
    n_rows: int,
    n_features: int,
) -> int:
    max_components = min(n_rows, n_features)
    if max_components <= 1:
        return 1
    return max(1, min(requested_components, max_components - 1))


def _validate_columns(dataframe, text_column: str, label_column: str) -> None:
    missing_columns = [
        column for column in (text_column, label_column) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")


if __name__ == "__main__":
    main()
