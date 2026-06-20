from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.preprocess import encode_labels, load_dataframe
from evaluation.visualize import (
    plot_tsne_by_label,
    plot_tsne_by_language,
    plot_umap_by_label,
    plot_umap_by_language,
)
from utils.io import ensure_dir, load_numpy, load_string_array
from utils.logger import get_logger

LOGGER = get_logger(__name__)

EMBEDDING_FILES = {
    "minilm": "minilm_embeddings.npy",
    "mbert": "mbert_embeddings.npy",
    "supcon_mbert": "supcon_mbert_embeddings.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UMAP and t-SNE embedding visualizations.")
    parser.add_argument(
        "--embedding_name",
        choices=["minilm", "mbert", "supcon_mbert"],
        required=True,
    )
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--metadata-file", default="data/processed/xnli_validation_en_tr.csv")
    parser.add_argument("--labels-path", default=None)
    parser.add_argument("--languages-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_dir = ROOT / args.embeddings_dir
    figures_dir = ensure_dir(ROOT / args.figures_dir)
    embeddings_path = embeddings_dir / EMBEDDING_FILES[args.embedding_name]

    LOGGER.info("Loading embeddings from: %s", embeddings_path)
    embeddings = load_numpy(embeddings_path)

    labels, languages = _load_labels_and_languages(args, embeddings_dir)
    _validate_lengths(embeddings, labels, languages)

    prefix = args.embedding_name
    plot_umap_by_label(
        embeddings=embeddings,
        labels=labels,
        output_path=figures_dir / f"{prefix}_umap_by_label.png",
        title=f"{prefix} UMAP Colored by NLI Label",
    )
    plot_umap_by_language(
        embeddings=embeddings,
        languages=languages,
        output_path=figures_dir / f"{prefix}_umap_by_language.png",
        title=f"{prefix} UMAP Colored by Language",
    )
    plot_tsne_by_label(
        embeddings=embeddings,
        labels=labels,
        output_path=figures_dir / f"{prefix}_tsne_by_label.png",
        title=f"{prefix} t-SNE Colored by NLI Label",
    )
    plot_tsne_by_language(
        embeddings=embeddings,
        languages=languages,
        output_path=figures_dir / f"{prefix}_tsne_by_language.png",
        title=f"{prefix} t-SNE Colored by Language",
    )

    LOGGER.info("Saved visualizations under: %s", figures_dir)


def _load_labels_and_languages(
    args: argparse.Namespace,
    embeddings_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    labels_path = Path(args.labels_path) if args.labels_path else None
    languages_path = Path(args.languages_path) if args.languages_path else None

    if labels_path is not None and not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    if languages_path is not None and not languages_path.is_absolute():
        languages_path = ROOT / languages_path

    if args.embedding_name == "supcon_mbert":
        labels_path = labels_path or embeddings_dir / "supcon_labels.npy"
        languages_path = languages_path or embeddings_dir / "supcon_languages.npy"

    if labels_path and languages_path and labels_path.exists() and languages_path.exists():
        LOGGER.info("Loading labels from: %s", labels_path)
        LOGGER.info("Loading languages from: %s", languages_path)
        labels = _format_labels(load_numpy(labels_path))
        languages = load_string_array(languages_path)
        return labels, languages

    metadata_path = ROOT / args.metadata_file
    LOGGER.info("Loading labels and languages from metadata CSV: %s", metadata_path)
    dataframe = load_dataframe(metadata_path)
    labels = _format_labels(dataframe["label_id"].to_numpy())
    languages = dataframe["language"].astype(str).to_numpy()
    return labels, languages


def _format_labels(labels) -> np.ndarray:
    formatted_labels = []
    for label in labels:
        try:
            formatted_labels.append(encode_labels(int(label)))
        except (TypeError, ValueError):
            formatted_labels.append(str(label))
    return np.asarray(formatted_labels)


def _validate_lengths(
    embeddings: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
) -> None:
    if len(embeddings) != len(labels) or len(embeddings) != len(languages):
        raise ValueError(
            "Embeddings, labels, and languages must have the same number of samples. "
            f"Got embeddings={len(embeddings)}, labels={len(labels)}, languages={len(languages)}."
        )


if __name__ == "__main__":
    main()
