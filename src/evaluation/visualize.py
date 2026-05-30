from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_umap_by_label(
    embeddings: np.ndarray,
    labels,
    output_path: str | Path = "outputs/figures/umap_by_label.png",
    title: str = "UMAP Projection Colored by NLI Label",
) -> None:
    """Create a UMAP plot colored by NLI label."""

    projection = _compute_umap(embeddings)
    _plot_projection(
        projection=projection,
        categories=labels,
        output_path=output_path,
        title=title,
        legend_title="NLI label",
    )


def plot_umap_by_language(
    embeddings: np.ndarray,
    languages,
    output_path: str | Path = "outputs/figures/umap_by_language.png",
    title: str = "UMAP Projection Colored by Language",
) -> None:
    """Create a UMAP plot colored by language."""

    projection = _compute_umap(embeddings)
    _plot_projection(
        projection=projection,
        categories=languages,
        output_path=output_path,
        title=title,
        legend_title="Language",
    )


def plot_tsne_by_label(
    embeddings: np.ndarray,
    labels,
    output_path: str | Path = "outputs/figures/tsne_by_label.png",
    title: str = "t-SNE Projection Colored by NLI Label",
) -> None:
    """Create a t-SNE plot colored by NLI label."""

    projection = _compute_tsne(embeddings)
    _plot_projection(
        projection=projection,
        categories=labels,
        output_path=output_path,
        title=title,
        legend_title="NLI label",
    )


def plot_tsne_by_language(
    embeddings: np.ndarray,
    languages,
    output_path: str | Path = "outputs/figures/tsne_by_language.png",
    title: str = "t-SNE Projection Colored by Language",
) -> None:
    """Create a t-SNE plot colored by language."""

    projection = _compute_tsne(embeddings)
    _plot_projection(
        projection=projection,
        categories=languages,
        output_path=output_path,
        title=title,
        legend_title="Language",
    )


def _compute_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(embeddings) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _compute_tsne(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30, max(1, (len(embeddings) - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _plot_projection(
    projection: np.ndarray,
    categories,
    output_path: str | Path,
    title: str,
    legend_title: str,
) -> None:
    import matplotlib.pyplot as plt

    _validate_projection_inputs(projection, categories)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories_array = np.asarray(categories).astype(str)
    unique_categories = sorted(set(categories_array.tolist()))
    color_map = plt.get_cmap("tab10", max(len(unique_categories), 1))

    plt.figure(figsize=(8, 6), dpi=150)
    for index, category in enumerate(unique_categories):
        mask = categories_array == category
        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=18,
            alpha=0.75,
            color=color_map(index),
            label=category,
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.2)
    plt.legend(title=legend_title, loc="best", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


def _validate_projection_inputs(projection: np.ndarray, categories: Any) -> None:
    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError("projection must have shape [num_samples, 2].")
    if len(projection) != len(categories):
        raise ValueError("projection and categories must have the same number of samples.")
