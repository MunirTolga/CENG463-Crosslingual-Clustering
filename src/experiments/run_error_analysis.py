from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluation.error_analysis import run_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering error analysis reports.")
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


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
