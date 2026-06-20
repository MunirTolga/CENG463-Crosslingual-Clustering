from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """Run K-Means clustering and return labels with the fitted model."""

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = model.fit_predict(embeddings)
    return cluster_labels, model
