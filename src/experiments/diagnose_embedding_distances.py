from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.io import ensure_dir, load_numpy
from utils.logger import get_logger

LOGGER = get_logger(__name__)

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose distance ranges for SupCon embeddings.")
    parser.add_argument("--embeddings_path", default="outputs/embeddings/supcon_mbert_embeddings.npy")
    parser.add_argument("--output_path", default="outputs/reports/embedding_distance_diagnostics.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = load_numpy(_resolve_path(args.embeddings_path))
    report = build_distance_report(embeddings)

    output_path = _resolve_path(args.output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(report, encoding="utf-8")

    print(report)
    LOGGER.info("Saved embedding distance diagnostics to: %s", output_path)


def build_distance_report(embeddings: np.ndarray) -> str:
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if len(embeddings) < 2:
        raise ValueError("At least two embeddings are required for distance diagnostics.")

    euclidean_distances = _upper_triangle_distances(pairwise_distances(embeddings, metric="euclidean"))
    cosine_distances = _upper_triangle_distances(pairwise_distances(embeddings, metric="cosine"))

    lines = [
        "Embedding distance diagnostics",
        f"Embedding shape: {tuple(embeddings.shape)}",
        "",
        "Euclidean distance percentiles:",
        *_format_percentiles(euclidean_distances),
        "",
        "Cosine distance percentiles:",
        *_format_percentiles(cosine_distances),
        "",
        "Interpretation note:",
        "DBSCAN eps values must match the actual distance scale. Transformer embeddings in",
        "768 dimensions often have Euclidean distances much larger than 0.3 to 1.2, while",
        "cosine distances after normalization usually occupy a smaller and more useful range.",
    ]
    return "\n".join(lines)


def _upper_triangle_distances(distance_matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices_from(distance_matrix, k=1)
    return distance_matrix[indices]


def _format_percentiles(distances: np.ndarray) -> list[str]:
    values = np.percentile(distances, PERCENTILES)
    return [f"  p{percentile:02d}: {value:.6f}" for percentile, value in zip(PERCENTILES, values)]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
