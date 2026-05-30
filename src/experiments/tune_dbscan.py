from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.dbscan import run_dbscan
from clustering.preprocess import preprocess_embeddings_for_clustering
from data.preprocess import encode_label
from evaluation.metrics import evaluate_clustering
from utils.io import ensure_dir, load_numpy, load_string_array, save_json
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DBSCAN for SupCon mBERT embeddings.")
    parser.add_argument("--embeddings_path", default="outputs/embeddings/supcon_mbert_embeddings.npy")
    parser.add_argument("--labels_path", default="outputs/embeddings/supcon_labels.npy")
    parser.add_argument("--languages_path", default="outputs/embeddings/supcon_languages.npy")
    parser.add_argument("--output_dir", default="outputs/metrics")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(_resolve_path(args.output_dir))

    LOGGER.info(
        "DBSCAN parameters should be selected on validation data and reported transparently."
    )
    embeddings = load_numpy(_resolve_path(args.embeddings_path))
    labels = load_numpy(_resolve_path(args.labels_path))
    languages = load_string_array(_resolve_path(args.languages_path))

    grid = _build_grid(args.quick)
    rows = []
    preprocessing_cache = {}

    for params in grid:
        cache_key = (
            params["normalize_embeddings"],
            params["reducer"],
            params["pca_components"],
        )
        if cache_key not in preprocessing_cache:
            preprocessing_cache[cache_key] = preprocess_embeddings_for_clustering(
                embeddings,
                normalize_embeddings=params["normalize_embeddings"],
                reducer=params["reducer"],
                pca_components=params["pca_components"] or 50,
            )
        processed_embeddings, preprocessing_info = preprocessing_cache[cache_key]

        cluster_labels, _ = run_dbscan(
            processed_embeddings,
            eps=params["eps"],
            min_samples=params["min_samples"],
            metric=params["metric"],
        )
        metrics = evaluate_clustering(processed_embeddings, labels, cluster_labels)
        valid_clustering = metrics["number_of_clusters"] >= 2 and metrics["noise_ratio"] < 0.95
        selection_score = _selection_score(metrics)

        row = {
            **params,
            **metrics,
            "valid_clustering": valid_clustering,
            "selection_score": selection_score,
            "original_embedding_shape": preprocessing_info["original_shape"],
            "processed_embedding_shape": preprocessing_info["processed_shape"],
            "pca_components_used": preprocessing_info["pca_components_used"],
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    grid_path = output_dir / "supcon_dbscan_grid_search.csv"
    results.to_csv(grid_path, index=False)
    LOGGER.info("Saved DBSCAN grid search results to: %s", grid_path)

    best_row = _select_best_row(results)
    if not bool(best_row["valid_clustering"]):
        LOGGER.warning(
            "No fully valid DBSCAN configuration was found. Selecting the lowest-noise "
            "configuration for diagnosis, but it should not be reported as a successful cluster result."
        )

    best_params = best_row.to_dict()
    best_params_path = output_dir / "supcon_dbscan_best_params.json"
    save_json(_json_safe(best_params), best_params_path)
    LOGGER.info("Saved best DBSCAN parameters to: %s", best_params_path)

    best_predictions = _build_best_predictions(embeddings, labels, languages, best_params)
    predictions_path = output_dir / "supcon_mbert_dbscan_best_predictions.csv"
    best_predictions.to_csv(predictions_path, index=False)
    LOGGER.info("Saved best DBSCAN predictions to: %s", predictions_path)


def _build_grid(quick: bool) -> list[dict]:
    if quick:
        metric_options = ["cosine"]
        normalize_options = [True]
        reducer_options = ["none", "pca"]
        pca_components_options = [50]
        eps_values_cosine = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        eps_values_euclidean = []
        min_samples_values = [2, 3, 5]
    else:
        metric_options = ["cosine", "euclidean"]
        normalize_options = [True]
        reducer_options = ["none", "pca"]
        pca_components_options = [20, 50, 100]
        eps_values_cosine = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
        eps_values_euclidean = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]
        min_samples_values = [2, 3, 5, 8]

    grid = []
    for metric in metric_options:
        eps_values = eps_values_cosine if metric == "cosine" else eps_values_euclidean
        for normalize_embeddings in normalize_options:
            for reducer in reducer_options:
                pca_values = pca_components_options if reducer == "pca" else [None]
                for pca_components in pca_values:
                    for eps in eps_values:
                        for min_samples in min_samples_values:
                            grid.append(
                                {
                                    "metric": metric,
                                    "normalize_embeddings": normalize_embeddings,
                                    "reducer": reducer,
                                    "pca_components": pca_components,
                                    "eps": eps,
                                    "min_samples": min_samples,
                                }
                            )
    return grid


def _select_best_row(results: pd.DataFrame) -> pd.Series:
    valid_results = results[results["valid_clustering"] == True].copy()
    if not valid_results.empty:
        return valid_results.sort_values("selection_score", ascending=False).iloc[0]
    return results.sort_values(["noise_ratio", "number_of_clusters"], ascending=[True, False]).iloc[0]


def _selection_score(metrics: dict) -> float:
    return (
        _score_value(metrics.get("normalized_mutual_info_score"))
        + _score_value(metrics.get("v_measure_score"))
        + _score_value(metrics.get("homogeneity_score"))
        - 0.25 * _score_value(metrics.get("noise_ratio"))
    )


def _score_value(value) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value):
        return 0.0
    return value


def _build_best_predictions(embeddings, labels, languages, best_params: dict) -> pd.DataFrame:
    processed_embeddings, _ = preprocess_embeddings_for_clustering(
        embeddings,
        normalize_embeddings=bool(best_params["normalize_embeddings"]),
        reducer=str(best_params["reducer"]),
        pca_components=_safe_pca_components(best_params.get("pca_components")),
    )
    cluster_labels, _ = run_dbscan(
        processed_embeddings,
        eps=float(best_params["eps"]),
        min_samples=int(best_params["min_samples"]),
        metric=str(best_params["metric"]),
    )
    return pd.DataFrame(
        {
            "true_label": labels,
            "label_name": [_safe_label_name(label) for label in labels],
            "language": languages.astype(str),
            "cluster_label": cluster_labels,
        }
    )


def _safe_label_name(label) -> str:
    try:
        return encode_label(label)
    except (TypeError, ValueError):
        return str(label)


def _safe_pca_components(value) -> int:
    if value is None:
        return 50
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 50
    if math.isnan(value):
        return 50
    return int(value)


def _json_safe(data: dict) -> dict:
    safe = {}
    integer_fields = {
        "min_samples",
        "number_of_clusters",
        "number_of_noise_points",
        "pca_components",
        "pca_components_used",
    }
    for key, value in data.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        if key in integer_fields and value is not None:
            value = int(value)
        safe[str(key)] = value
    return safe


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
