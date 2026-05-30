from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.io import ensure_dir, load_json
from utils.logger import get_logger

LOGGER = get_logger(__name__)

SUMMARY_COLUMNS = [
    "model_name",
    "silhouette_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "adjusted_rand_score",
    "normalized_mutual_info_score",
    "number_of_clusters",
    "noise_ratio",
]


@dataclass(frozen=True)
class StepResult:
    name: str
    return_code: int
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete CENG463 experiment pipeline.")
    parser.add_argument("--quick", action="store_true", help="Use small subsets and one epoch.")
    parser.add_argument("--train-samples-per-language", type=int, default=None)
    parser.add_argument("--eval-samples-per-language", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument(
        "--msimcse_model_name",
        default=None,
        help="Optional sentence-transformers model name for Baseline 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _resolve_settings(args)
    results: list[StepResult] = []

    LOGGER.info("Starting final experiment pipeline")
    if args.quick:
        LOGGER.info("Quick mode enabled: %s", settings)

    results.extend(_prepare_data(settings))
    results.append(_run_all_baselines(args, settings))
    results.append(_train_supcon(settings))
    results.append(_extract_supcon_embeddings(settings))
    results.append(_run_proposed_method(args, settings))
    results.extend(_run_visualizations(settings))
    results.append(_run_error_analysis())

    summary_path = _save_final_summary(results)
    LOGGER.info("Final experiment pipeline finished")
    LOGGER.info("Saved final summary table to: %s", summary_path)


def _resolve_settings(args: argparse.Namespace) -> dict[str, int | float | None]:
    common = {
        "train_file": "data/processed/xnli_train.csv",
        "validation_file": "data/processed/xnli_validation.csv",
        "alternate_validation_file": "data/processed/xnli_valid.csv",
        "embeddings_dir": "outputs/embeddings/final_quick" if args.quick else "outputs/embeddings/final",
    }

    if args.quick:
        return common | {
            "train_samples_per_language": args.train_samples_per_language or 20,
            "eval_samples_per_language": args.eval_samples_per_language or 20,
            "epochs": args.epochs or 1,
            "batch_size": args.batch_size or 4,
            "learning_rate": args.learning_rate,
        }

    return common | {
        "train_samples_per_language": args.train_samples_per_language or 500,
        "eval_samples_per_language": args.eval_samples_per_language or 500,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }


def _prepare_data(settings: dict[str, int | float | None]) -> list[StepResult]:
    LOGGER.info("Step 1/8: Preparing XNLI train and validation data")
    max_samples = _prepare_data_sample_limit(settings)
    command = [
        sys.executable,
        "-m",
        "scripts.prepare_data",
        "--source_splits",
        "train",
        "validation",
        "--max_samples_per_language",
        str(max_samples),
    ]
    result = _run_step(name="prepare_data", command=command)
    _verify_processed_data_files(settings)
    return [result]


def _run_all_baselines(args: argparse.Namespace, settings: dict[str, int | float | None]) -> StepResult:
    LOGGER.info("Step 2/8: Running all baseline models")
    command = [
        sys.executable,
        "-m",
        "scripts.run_baselines",
        "--input-file",
        _path_arg(_validation_file(settings)),
        "--embeddings-dir",
        str(settings["embeddings_dir"]),
        "--eps",
        str(args.eps),
        "--min_samples",
        str(args.min_samples),
    ]
    if args.msimcse_model_name:
        command.extend(["--msimcse_model_name", args.msimcse_model_name])
    else:
        LOGGER.info("No --msimcse_model_name provided; Baseline 5 will be logged as failed.")
    return _run_step("run_all_baselines", command)


def _train_supcon(settings: dict[str, int | float | None]) -> StepResult:
    LOGGER.info("Step 3/8: Training supervised contrastive mBERT")
    command = [
        sys.executable,
        "-m",
        "src.training.train_supcon",
        "--train_file",
        _path_arg(_train_file(settings)),
    ]

    if settings["epochs"] is not None:
        command.extend(["--epochs", str(settings["epochs"])])
    if settings["batch_size"] is not None:
        command.extend(["--batch_size", str(settings["batch_size"])])
    if settings["learning_rate"] is not None:
        command.extend(["--learning_rate", str(settings["learning_rate"])])

    return _run_step("train_supcon_mbert", command)


def _extract_supcon_embeddings(settings: dict[str, int | float | None]) -> StepResult:
    LOGGER.info("Step 4/8: Extracting SupCon mBERT encoder embeddings")
    command = [
        sys.executable,
        "-m",
        "src.experiments.extract_supcon_embeddings",
        "--input-file",
        _path_arg(_validation_file(settings)),
        "--output-dir",
        str(settings["embeddings_dir"]),
    ]
    if settings["batch_size"] is not None:
        command.extend(["--batch-size", str(settings["batch_size"])])
    return _run_step("extract_supcon_embeddings", command)


def _run_proposed_method(args: argparse.Namespace, settings: dict[str, int | float | None]) -> StepResult:
    LOGGER.info("Step 5/8: Running SupCon mBERT + DBSCAN")
    embeddings_dir = Path(str(settings["embeddings_dir"]))
    return _run_step(
        name="run_proposed_method",
        command=[
            sys.executable,
            "-m",
            "src.experiments.run_proposed_method",
            "--embeddings-path",
            str(embeddings_dir / "supcon_mbert_embeddings.npy"),
            "--labels-path",
            str(embeddings_dir / "supcon_labels.npy"),
            "--languages-path",
            str(embeddings_dir / "supcon_languages.npy"),
            "--eps",
            str(args.eps),
            "--min_samples",
            str(args.min_samples),
        ],
    )


def _run_visualizations(settings: dict[str, int | float | None]) -> list[StepResult]:
    LOGGER.info("Step 6/8: Generating embedding visualizations")
    results = []
    for embedding_name in ["minilm", "mbert", "supcon_mbert"]:
        results.append(
            _run_step(
                name=f"visualize_{embedding_name}",
                command=[
                    sys.executable,
                    "-m",
                    "src.experiments.run_visualizations",
                    "--embedding_name",
                    embedding_name,
                    "--embeddings-dir",
                    str(settings["embeddings_dir"]),
                    "--metadata-file",
                    _path_arg(_validation_file(settings)),
                ],
                required=False,
            )
        )
    return results


def _run_error_analysis() -> StepResult:
    LOGGER.info("Step 7/8: Running clustering error analysis")
    return _run_step(
        name="error_analysis",
        command=[
            sys.executable,
            "-m",
            "src.evaluation.error_analysis",
            "--predictions-file",
            "outputs/metrics/supcon_mbert_dbscan_predictions.csv",
        ],
    )


def _save_final_summary(results: list[StepResult]) -> Path:
    LOGGER.info("Step 8/8: Saving final summary table")
    metrics_dir = ensure_dir(ROOT / "outputs/metrics")
    summary_path = metrics_dir / "final_summary.csv"

    rows = _load_metric_rows(metrics_dir)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def _load_metric_rows(metrics_dir: Path) -> list[dict]:
    rows = []
    metric_files = {
        "tfidf_kmeans": "tfidf_kmeans_metrics.json",
        "minilm_kmeans": "minilm_kmeans_metrics.json",
        "mbert_kmeans": "mbert_kmeans_metrics.json",
        "mbert_dbscan": "mbert_dbscan_metrics.json",
        "msimcse_dbscan": "msimcse_dbscan_metrics.json",
        "supcon_mbert_dbscan": "supcon_mbert_dbscan_metrics.json",
    }

    for model_name, file_name in metric_files.items():
        row = {column: None for column in SUMMARY_COLUMNS}
        row["model_name"] = model_name
        metrics_path = metrics_dir / file_name
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            for column in SUMMARY_COLUMNS:
                if column != "model_name":
                    row[column] = metrics.get(column)
        rows.append(row)

    return rows


def _prepare_data_sample_limit(settings: dict[str, int | float | None]) -> int:
    train_limit = int(settings["train_samples_per_language"])
    eval_limit = int(settings["eval_samples_per_language"])
    if train_limit != eval_limit:
        chosen_limit = min(train_limit, eval_limit)
        LOGGER.info(
            "prepare_data.py uses one max_samples_per_language value for all project splits. "
            "Using %d because train=%d and eval=%d.",
            chosen_limit,
            train_limit,
            eval_limit,
        )
        return chosen_limit
    return train_limit


def _verify_processed_data_files(settings: dict[str, int | float | None]) -> None:
    train_file = _train_file(settings)
    validation_file = _validation_file(settings)

    LOGGER.info("Prepared data files:")
    LOGGER.info("train: %s exists=%s", _path_arg(train_file), train_file.exists())
    LOGGER.info("validation: %s exists=%s", _path_arg(validation_file), validation_file.exists())

    missing = [path for path in [train_file, validation_file] if not path.exists()]
    if missing:
        expected = ", ".join(_path_arg(path) for path in [train_file, validation_file])
        missing_text = ", ".join(_path_arg(path) for path in missing)
        raise FileNotFoundError(
            f"Processed data preparation did not create required files. "
            f"Missing: {missing_text}. Expected: {expected}."
        )


def resolve_processed_file(preferred_paths: list[str | Path]) -> Path:
    candidates = [_resolve_root_path(path) for path in preferred_paths]
    for path in candidates:
        if path.exists():
            return path

    expected = ", ".join(_path_arg(path) for path in candidates)
    raise FileNotFoundError(f"No processed data file found. Expected one of: {expected}")


def _train_file(settings: dict[str, int | float | None]) -> Path:
    return resolve_processed_file([settings["train_file"]])


def _validation_file(settings: dict[str, int | float | None]) -> Path:
    return resolve_processed_file(
        [
            settings["validation_file"],
            settings["alternate_validation_file"],
        ]
    )


def _resolve_root_path(path: str | Path) -> Path:
    path = Path(str(path))
    if path.is_absolute():
        return path
    return ROOT / path


def _path_arg(path: str | Path) -> str:
    path = _resolve_root_path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run_step(
    name: str,
    command: list[str],
    required: bool = True,
) -> StepResult:
    LOGGER.info("Running step: %s", name)
    LOGGER.info("Command: %s", " ".join(command))

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout.strip():
        LOGGER.info("%s stdout:\n%s", name, result.stdout.strip())
    if result.stderr.strip():
        LOGGER.error("%s stderr:\n%s", name, result.stderr.strip())

    if result.returncode != 0:
        LOGGER.error("Step failed: %s (return code %d)", name, result.returncode)
        if required:
            raise RuntimeError(f"Required pipeline step failed: {name}")
    else:
        LOGGER.info("Step completed: %s", name)

    return StepResult(name=name, return_code=result.returncode, command=command)


if __name__ == "__main__":
    main()
