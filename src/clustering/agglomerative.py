from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def run_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int = 3,
    metric: str = "cosine",
    linkage: str = "average",
) -> tuple[np.ndarray, AgglomerativeClustering]:
    """Run agglomerative clustering and return labels with the fitted model."""

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=metric,
            linkage=linkage,
        )

    cluster_labels = model.fit_predict(embeddings)
    return cluster_labels, model
