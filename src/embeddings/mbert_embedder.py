from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm


def encode_texts_with_mbert(
    texts: Sequence[str],
    model_name: str = "bert-base-multilingual-cased",
    batch_size: int = 16,
    max_length: int = 128,
    device: str | None = None,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract raw mBERT embeddings without fine-tuning."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be either 'mean' or 'cls'.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding with mBERT"):
            batch_texts = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(**encoded)
            if pooling == "mean":
                embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            else:
                embeddings = outputs.last_hidden_state[:, 0]

            all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


def _mean_pool(token_embeddings, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed_embeddings = torch.sum(token_embeddings * mask, dim=1)
    token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed_embeddings / token_counts
