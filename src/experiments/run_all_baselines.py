from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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
class BaselineSpec:
    name: str
    module: str
    metrics_file: str
    command_args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all baseline clustering experiments.")
    parser.add_argument("--input-file", default="data/processed/xnli_validation_en_tr.csv")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", "--min-samples", dest="min_samples", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--minilm-model-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--mbert-model-name", default="bert-base-multilingual-cased")
    parser.add_argument(
        "--msimcse_model_name",
        "--msimcse-model-name",
        dest="msimcse_model_name",
        default=None,
        help="Required to run Baseline 5 with a chosen sentence-transformers model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = ensure_dir(ROOT / args.metrics_dir)
    ensure_dir(ROOT / args.embeddings_dir)

    baseline_specs = _build_baseline_specs(args)
    successful_baselines: set[str] = set()

    for spec in baseline_specs:
        if spec.name == "msimcse_dbscan" and not args.msimcse_model_name:
            LOGGER.error("Baseline failed: %s. Missing --msimcse_model_name.", spec.name)
            continue

        success = _run_baseline(spec)
        if success:
            successful_baselines.add(spec.name)

    summary = _build_summary(baseline_specs, successful_baselines, metrics_dir)
    summary_path = metrics_dir / "baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved baseline summary to: %s", summary_path)


def _build_baseline_specs(args: argparse.Namespace) -> list[BaselineSpec]:
    metrics_dir = args.metrics_dir
    embeddings_dir = args.embeddings_dir

    common_args = [
        "--input-file",
        args.input_file,
        "--output-dir",
        metrics_dir,
        "--random-state",
        str(args.random_state),
    ]

    return [
        BaselineSpec(
            name="tfidf_kmeans",
            module="src.baselines.run_tfidf_kmeans",
            metrics_file="tfidf_kmeans_metrics.json",
            command_args=common_args,
        ),
        BaselineSpec(
            name="minilm_kmeans",
            module="src.baselines.run_minilm_kmeans",
            metrics_file="minilm_kmeans_metrics.json",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/minilm_embeddings.npy",
                "--model-name",
                args.minilm_model_name,
            ],
        ),
        BaselineSpec(
            name="mbert_kmeans",
            module="src.baselines.run_mbert_kmeans",
            metrics_file="mbert_kmeans_metrics.json",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/mbert_embeddings.npy",
                "--model-name",
                args.mbert_model_name,
            ],
        ),
        BaselineSpec(
            name="mbert_dbscan",
            module="src.baselines.run_mbert_dbscan",
            metrics_file="mbert_dbscan_metrics.json",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/mbert_embeddings.npy",
                "--model-name",
                args.mbert_model_name,
                "--eps",
                str(args.eps),
                "--min_samples",
                str(args.min_samples),
            ],
        ),
        BaselineSpec(
            name="msimcse_dbscan",
            module="src.baselines.run_msimcse_dbscan",
            metrics_file="msimcse_dbscan_metrics.json",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/msimcse_embeddings.npy",
                "--model_name",
                args.msimcse_model_name or "",
                "--eps",
                str(args.eps),
                "--min_samples",
                str(args.min_samples),
            ],
        ),
    ]


def _run_baseline(spec: BaselineSpec) -> bool:
    command = [sys.executable, "-m", spec.module, *spec.command_args]
    LOGGER.info("Running baseline: %s", spec.name)

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout.strip():
        LOGGER.info("%s output:\n%s", spec.name, result.stdout.strip())

    if result.returncode != 0:
        LOGGER.error("Baseline failed: %s", spec.name)
        if result.stderr.strip():
            LOGGER.error("%s error:\n%s", spec.name, result.stderr.strip())
        return False

    return True


def _build_summary(
    baseline_specs: list[BaselineSpec],
    successful_baselines: set[str],
    metrics_dir: Path,
) -> pd.DataFrame:
    rows = []

    for spec in baseline_specs:
        row = {column: None for column in SUMMARY_COLUMNS}
        row["model_name"] = spec.name

        metrics_path = metrics_dir / spec.metrics_file
        if spec.name in successful_baselines and metrics_path.exists():
            metrics = load_json(metrics_path)
            for column in SUMMARY_COLUMNS:
                if column != "model_name":
                    row[column] = metrics.get(column)
        elif spec.name in successful_baselines:
            LOGGER.error("Metrics file missing for baseline: %s", spec.name)

        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


if __name__ == "__main__":
    main()
