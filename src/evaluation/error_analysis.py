from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.preprocess import encode_labels, load_dataframe
from utils.io import ensure_dir
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def run_error_analysis(
    predictions_file: str | Path,
    output_dir: str | Path = "outputs/reports",
) -> dict[str, pd.DataFrame]:
    """Analyze cluster-label and cluster-language alignment.

    Cluster ids are arbitrary, so these tables describe alignment patterns rather
    than classification accuracy.
    """

    predictions_path = Path(predictions_file)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}. "
            "Please run run_proposed_method.py first."
        )

    LOGGER.info("Predictions file: %s", predictions_path)
    LOGGER.info("Output directory: %s", output_dir)

    predictions = load_dataframe(predictions_path)
    output_dir = ensure_dir(output_dir)
    label_column = find_column(predictions, ["true_label", "label_id", "label", "label_name"])
    language_column = find_column(predictions, ["language", "languages"])
    cluster_column = find_column(predictions, ["cluster_label", "cluster", "predicted_cluster"])

    LOGGER.info("Detected label column: %s", label_column)
    LOGGER.info("Detected language column: %s", language_column)
    LOGGER.info("Detected cluster column: %s", cluster_column)
    LOGGER.info("Rows loaded: %d", len(predictions))
    LOGGER.info("Unique clusters: %d", predictions[cluster_column].nunique(dropna=False))

    noise_points = int((predictions[cluster_column] == -1).sum())
    LOGGER.info("Noise points: %d", noise_points)
    if noise_points == len(predictions) and len(predictions) > 0:
        LOGGER.warning(
            "All samples are assigned to DBSCAN noise cluster (-1). Error analysis files "
            "were generated, but clustering is not meaningful."
        )

    analysis = predictions.copy()
    analysis["_true_label_for_analysis"] = _label_values_for_analysis(analysis, label_column)

    cluster_label_distribution = (
        analysis.groupby([cluster_column, "_true_label_for_analysis"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                cluster_column: "cluster_id",
                "_true_label_for_analysis": "true_label",
            }
        )
    )
    cluster_language_distribution = (
        analysis.groupby([cluster_column, language_column], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={cluster_column: "cluster_id", language_column: "language"})
    )

    cluster_label_crosstab = pd.crosstab(
        analysis[cluster_column],
        analysis["_true_label_for_analysis"],
        rownames=["cluster_id"],
        colnames=["true_label"],
        margins=True,
    )
    cluster_language_crosstab = pd.crosstab(
        analysis[cluster_column],
        analysis[language_column],
        rownames=["cluster_id"],
        colnames=["language"],
        margins=True,
    )

    output_paths = {
        "cluster_label_distribution": output_dir / "cluster_label_distribution.csv",
        "cluster_language_distribution": output_dir / "cluster_language_distribution.csv",
        "cluster_label_crosstab": output_dir / "cluster_label_crosstab.csv",
        "cluster_language_crosstab": output_dir / "cluster_language_crosstab.csv",
    }

    cluster_label_distribution.to_csv(output_paths["cluster_label_distribution"], index=False)
    cluster_language_distribution.to_csv(output_paths["cluster_language_distribution"], index=False)
    cluster_label_crosstab.to_csv(output_paths["cluster_label_crosstab"])
    cluster_language_crosstab.to_csv(output_paths["cluster_language_crosstab"])

    for output_path in output_paths.values():
        LOGGER.info("Saved error analysis output to: %s", output_path)
    LOGGER.info("These tables describe cluster-label alignment, not prediction accuracy.")

    return {
        "cluster_label_distribution": cluster_label_distribution,
        "cluster_language_distribution": cluster_language_distribution,
        "cluster_label_crosstab": cluster_label_crosstab,
        "cluster_language_crosstab": cluster_language_crosstab,
    }


def analyze_cluster_alignment(
    predictions_path: str | Path,
    output_dir: str | Path = "outputs/reports",
) -> dict[str, pd.DataFrame]:
    """Backward-compatible alias for run_error_analysis."""

    return run_error_analysis(predictions_path, output_dir)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """Find the first available column from a candidate list."""

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            "Missing required column. "
            f"Tried candidates: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze clustering outputs as cluster-label alignment tables."
    )
    parser.add_argument(
        "--predictions-file",
        "--predictions_file",
        dest="predictions_file",
        default="outputs/metrics/supcon_mbert_dbscan_predictions.csv",
        help="Path to predictions CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="outputs/reports",
        help="Directory where error analysis reports will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_error_analysis(
        predictions_file=_resolve_path(args.predictions_file),
        output_dir=_resolve_path(args.output_dir),
    )


def _label_values_for_analysis(predictions: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column == "label_name":
        return predictions[label_column].astype(str)

    if "label_name" in predictions.columns:
        return predictions["label_name"].astype(str)

    return predictions[label_column].apply(_safe_label_name)


def _safe_label_name(value) -> str:
    try:
        return encode_labels(value)
    except (TypeError, ValueError):
        return str(value)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
