from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    n_jobs: int | None = None,
) -> tuple[np.ndarray, DBSCAN]:
    """Run DBSCAN clustering and return labels with the fitted model."""

    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be either 'euclidean' or 'cosine'.")

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
    )
    cluster_labels = model.fit_predict(embeddings)
    return cluster_labels, model
