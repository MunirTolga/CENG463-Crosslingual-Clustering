from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_embeddings(
    texts: Sequence[str],
    max_features: int = 10000,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Build dense TF-IDF embeddings for a collection of texts."""

    vectorizer = TfidfVectorizer(max_features=max_features)
    sparse_embeddings = vectorizer.fit_transform(texts)
    embeddings = sparse_embeddings.toarray().astype(np.float32)
    return embeddings, vectorizer
