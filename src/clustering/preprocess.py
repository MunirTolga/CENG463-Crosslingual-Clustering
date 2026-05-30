from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def preprocess_embeddings_for_clustering(
    embeddings,
    normalize_embeddings: bool = True,
    reducer: str = "none",
    pca_components: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare embeddings for clustering with optional L2 normalization and PCA."""

    embeddings_array = np.asarray(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape [num_samples, num_features].")

    processed_embeddings = embeddings_array.astype(np.float32, copy=True)
    original_shape = tuple(processed_embeddings.shape)
    pca_components_used = None

    if normalize_embeddings:
        processed_embeddings = normalize(processed_embeddings, norm="l2").astype(np.float32)

    if reducer not in {"none", "pca"}:
        raise ValueError("reducer must be either 'none' or 'pca'.")

    if reducer == "pca":
        n_samples, n_features = processed_embeddings.shape
        safe_components = min(int(pca_components), n_samples - 1, n_features)
        if safe_components >= 2:
            pca = PCA(n_components=safe_components, random_state=random_state)
            processed_embeddings = pca.fit_transform(processed_embeddings).astype(np.float32)
            pca_components_used = int(safe_components)

    preprocessing_info = {
        "original_shape": list(original_shape),
        "processed_shape": list(processed_embeddings.shape),
        "normalize_embeddings": bool(normalize_embeddings),
        "reducer": reducer,
        "pca_components_used": pca_components_used,
    }
    return processed_embeddings, preprocessing_info
