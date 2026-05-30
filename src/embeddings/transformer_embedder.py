from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from utils.io import load_numpy, save_numpy


def encode_texts_with_sentence_transformer(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode texts with a sentence-transformers model."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings to a .npy file."""

    save_numpy(embeddings, path)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from a .npy file."""

    return load_numpy(path)
