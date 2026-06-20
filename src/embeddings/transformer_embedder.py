from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

import numpy as np

from utils.io import load_numpy, save_numpy


def encode_texts_with_sentence_transformer(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    normalize_embeddings: bool = True,
    local_files_only: bool = False,
    offline: bool = False,
) -> np.ndarray:
    """Encode texts with a sentence-transformers model."""

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        local_files_only = True

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not load SentenceTransformer model '{model_name}' because "
            "sentence-transformers is not installed. Install project dependencies "
            "with: pip install -r requirements.txt."
        ) from exc

    try:
        model = _load_sentence_transformer(
            SentenceTransformer=SentenceTransformer,
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only or offline:
            raise RuntimeError(
                f"SentenceTransformer model '{model_name}' is not available locally "
                f"(local_files_only={local_files_only}). Please run once with "
                "internet access or provide a local model path."
            ) from exc
        raise RuntimeError(
            f"Could not load SentenceTransformer model '{model_name}' "
            f"(local_files_only={local_files_only}). Install sentence-transformers, "
            "download the model once, fix internet/cache access, or provide a local path."
        ) from exc

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def _load_sentence_transformer(
    SentenceTransformer,
    model_name: str,
    device: str | None,
    local_files_only: bool,
):
    if not local_files_only:
        return SentenceTransformer(model_name, device=device)

    try:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
    except TypeError:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return SentenceTransformer(model_name, device=device)


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings to a .npy file."""

    save_numpy(embeddings, path)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from a .npy file."""

    return load_numpy(path)
