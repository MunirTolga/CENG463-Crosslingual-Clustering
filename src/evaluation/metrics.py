from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from src.utils.io import save_json


def evaluate_clustering(
    embeddings: np.ndarray,
    true_labels: np.ndarray | list[int],
    cluster_labels: np.ndarray | list[int],
) -> dict[str, Any]:
    """Evaluate clustering assignments against gold labels."""

    true_labels_array = np.asarray(true_labels)
    cluster_labels_array = np.asarray(cluster_labels)

    if len(true_labels_array) != len(cluster_labels_array):
        raise ValueError("true_labels and cluster_labels must have the same length.")

    number_of_noise_points = int(np.sum(cluster_labels_array == -1))
    noise_ratio = (
        number_of_noise_points / len(cluster_labels_array) if len(cluster_labels_array) > 0 else 0.0
    )
    non_noise_clusters = set(cluster_labels_array[cluster_labels_array != -1].tolist())

    metrics = {
        "silhouette_score": _safe_silhouette_score(embeddings, cluster_labels_array),
        "homogeneity_score": float(homogeneity_score(true_labels_array, cluster_labels_array)),
        "completeness_score": float(completeness_score(true_labels_array, cluster_labels_array)),
        "v_measure_score": float(v_measure_score(true_labels_array, cluster_labels_array)),
        "adjusted_rand_score": float(adjusted_rand_score(true_labels_array, cluster_labels_array)),
        "normalized_mutual_info_score": float(
            normalized_mutual_info_score(true_labels_array, cluster_labels_array)
        ),
        "number_of_clusters": int(len(non_noise_clusters)),
        "noise_ratio": float(noise_ratio),
        "number_of_noise_points": number_of_noise_points,
    }
    return metrics


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save clustering metrics as JSON."""

    save_json(metrics, output_path)


def _safe_silhouette_score(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> float | None:
    if len(cluster_labels) < 2:
        return None

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(cluster_labels):
        return None

    try:
        return float(silhouette_score(embeddings, cluster_labels))
    except ValueError:
        return None
