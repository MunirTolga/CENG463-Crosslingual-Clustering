from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.dbscan import run_dbscan
from clustering.preprocess import preprocess_embeddings_for_clustering
from data.preprocess import encode_label
from evaluation.metrics import evaluate_clustering, save_metrics
from utils.io import ensure_dir, load_json, load_numpy, load_string_array
from utils.logger import get_logger

LOGGER = get_logger(__name__)

COMPARISON_METRICS = [
    "silhouette_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "adjusted_rand_score",
    "normalized_mutual_info_score",
    "number_of_clusters",
    "noise_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final proposed method: SupCon mBERT + DBSCAN.")
    parser.add_argument("--embeddings-path", default="outputs/embeddings/supcon_mbert_embeddings.npy")
    parser.add_argument("--labels-path", default="outputs/embeddings/supcon_labels.npy")
    parser.add_argument("--languages-path", default="outputs/embeddings/supcon_languages.npy")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--reducer", choices=["none", "pca"], default="none")
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--output_suffix", default=None)

    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize", dest="normalize_embeddings", action="store_true")
    normalize_group.add_argument("--no-normalize", dest="normalize_embeddings", action="store_false")
    parser.set_defaults(normalize_embeddings=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(_resolve_path(args.output_dir))

    embeddings = load_numpy(_resolve_path(args.embeddings_path))
    true_labels = load_numpy(_resolve_path(args.labels_path))
    languages = load_string_array(_resolve_path(args.languages_path))

    LOGGER.info("Loaded SupCon embeddings: %s", embeddings.shape)
    processed_embeddings, preprocessing_info = preprocess_embeddings_for_clustering(
        embeddings,
        normalize_embeddings=args.normalize_embeddings,
        reducer=args.reducer,
        pca_components=args.pca_components,
    )

    LOGGER.info("Original embedding shape: %s", tuple(preprocessing_info["original_shape"]))
    LOGGER.info("Processed embedding shape: %s", tuple(preprocessing_info["processed_shape"]))
    LOGGER.info("DBSCAN eps: %s", args.eps)
    LOGGER.info("DBSCAN min_samples: %s", args.min_samples)
    LOGGER.info("DBSCAN metric: %s", args.metric)
    LOGGER.info("Normalize embeddings: %s", args.normalize_embeddings)
    LOGGER.info("Reducer: %s", args.reducer)

    cluster_labels, _ = run_dbscan(
        processed_embeddings,
        eps=args.eps,
        min_samples=args.min_samples,
        metric=args.metric,
    )

    metrics = evaluate_clustering(processed_embeddings, true_labels, cluster_labels)
    _add_run_metadata(metrics, args, preprocessing_info)
    _warn_if_all_noise(metrics)

    default_metrics_path = output_dir / "supcon_mbert_dbscan_metrics.json"
    default_predictions_path = output_dir / "supcon_mbert_dbscan_predictions.csv"
    specific_suffix = args.output_suffix or _build_output_suffix(args)
    specific_metrics_path = output_dir / f"supcon_mbert_dbscan_{specific_suffix}_metrics.json"
    specific_predictions_path = output_dir / f"supcon_mbert_dbscan_{specific_suffix}_predictions.csv"

    predictions = _build_predictions(true_labels, languages, cluster_labels)

    save_metrics(metrics, default_metrics_path)
    save_metrics(metrics, specific_metrics_path)
    predictions.to_csv(default_predictions_path, index=False)
    predictions.to_csv(specific_predictions_path, index=False)

    LOGGER.info("Number of clusters: %s", metrics["number_of_clusters"])
    LOGGER.info("Noise ratio: %.4f", metrics["noise_ratio"])
    LOGGER.info("Saved metrics to: %s", default_metrics_path)
    LOGGER.info("Saved metrics to: %s", specific_metrics_path)
    LOGGER.info("Saved predictions to: %s", default_predictions_path)
    LOGGER.info("Saved predictions to: %s", specific_predictions_path)
    _compare_with_mbert_dbscan(metrics, output_dir / "mbert_dbscan_metrics.json")


def _add_run_metadata(
    metrics: dict,
    args: argparse.Namespace,
    preprocessing_info: dict,
) -> None:
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


def _warn_if_all_noise(metrics: dict) -> None:
    if metrics["number_of_clusters"] == 0:
        LOGGER.warning(
            "DBSCAN produced no valid clusters. All points may have been assigned as noise. "
            "Try larger eps, lower min_samples, cosine metric, normalization, or PCA."
        )


def _build_predictions(true_labels, languages, cluster_labels) -> pd.DataFrame:
    label_names = []
    for label in true_labels:
        try:
            label_names.append(encode_label(label))
        except (TypeError, ValueError):
            label_names.append(str(label))

    return pd.DataFrame(
        {
            "true_label": true_labels,
            "label_name": label_names,
            "language": languages.astype(str),
            "cluster_label": cluster_labels,
        }
    )


def _build_output_suffix(args: argparse.Namespace) -> str:
    norm_part = "norm" if args.normalize_embeddings else "raw"
    reducer_part = args.reducer if args.reducer == "none" else f"pca{args.pca_components}"
    return f"eps{_format_float(args.eps)}_ms{args.min_samples}_{args.metric}_{norm_part}_{reducer_part}"


def _format_float(value: float) -> str:
    return str(value)


def _compare_with_mbert_dbscan(
    proposed_metrics: dict,
    baseline_metrics_path: Path,
) -> None:
    if not baseline_metrics_path.exists():
        LOGGER.info("Baseline 4 metrics not found, skipping comparison: %s", baseline_metrics_path)
        return

    baseline_metrics = load_json(baseline_metrics_path)
    rows = []
    for metric_name in COMPARISON_METRICS:
        rows.append(
            {
                "metric": metric_name,
                "baseline_4_mbert_dbscan": baseline_metrics.get(metric_name),
                "proposed_supcon_mbert_dbscan": proposed_metrics.get(metric_name),
            }
        )

    comparison = pd.DataFrame(rows)
    LOGGER.info("Comparison with Baseline 4 mBERT + DBSCAN:\n%s", comparison.to_string(index=False))


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
