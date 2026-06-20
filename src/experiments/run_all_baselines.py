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
    "status",
    "reason",
    "embedding_type",
    "clustering_method",
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
    embedding_type: str
    clustering_method: str
    command_args: list[str]


@dataclass(frozen=True)
class BaselineRunResult:
    status: str
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all baseline clustering experiments.")
    parser.add_argument("--input-file", default="data/processed/xnli_validation_en_tr.csv")
    parser.add_argument(
        "--metrics-dir",
        "--output-dir",
        dest="metrics_dir",
        default="outputs/metrics",
    )
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", "--min-samples", dest="min_samples", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--minilm-model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--mbert-model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--labse-model-name", default="sentence-transformers/LaBSE")
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-labse",
        dest="include_labse",
        action="store_true",
        default=True,
        help="Run the LaBSE + K-Means baseline. Enabled by default.",
    )
    parser.add_argument(
        "--skip-labse",
        dest="include_labse",
        action="store_false",
        help="Skip the LaBSE + K-Means baseline.",
    )
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
    baseline_results: dict[str, BaselineRunResult] = {}

    for spec in baseline_specs:
        if spec.name == "msimcse_dbscan" and not args.msimcse_model_name:
            reason = "No --msimcse_model_name provided"
            LOGGER.warning("Baseline skipped: %s. %s.", spec.name, reason)
            baseline_results[spec.name] = BaselineRunResult(
                status="skipped",
                reason=reason,
            )
            continue
        if spec.name == "labse_kmeans" and not args.include_labse:
            reason = "LaBSE baseline disabled with --skip-labse"
            LOGGER.warning("Baseline skipped: %s. %s.", spec.name, reason)
            baseline_results[spec.name] = BaselineRunResult(
                status="skipped",
                reason=reason,
            )
            continue

        baseline_results[spec.name] = _run_baseline(spec)

    summary = _build_summary(baseline_specs, baseline_results, metrics_dir)
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
            embedding_type="lexical",
            clustering_method="kmeans",
            command_args=common_args,
        ),
        BaselineSpec(
            name="tfidf_svd_kmeans",
            module="src.baselines.run_tfidf_svd_kmeans",
            metrics_file="tfidf_svd_kmeans_metrics.json",
            embedding_type="lexical_svd",
            clustering_method="kmeans",
            command_args=[
                *common_args,
                "--embeddings-dir",
                embeddings_dir,
                *_force_args(args),
            ],
        ),
        BaselineSpec(
            name="minilm_kmeans",
            module="src.baselines.run_minilm_kmeans",
            metrics_file="minilm_kmeans_metrics.json",
            embedding_type="sentence_transformer",
            clustering_method="kmeans",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/minilm_embeddings.npy",
                "--model-name",
                args.minilm_model_name,
                *_sentence_transformer_cache_args(args),
                *_force_args(args),
            ],
        ),
        BaselineSpec(
            name="minilm_agglomerative",
            module="src.baselines.run_minilm_agglomerative",
            metrics_file="minilm_agglomerative_metrics.json",
            embedding_type="sentence_transformer",
            clustering_method="agglomerative",
            command_args=[
                *common_args,
                "--embeddings-dir",
                embeddings_dir,
                "--model-name",
                args.minilm_model_name,
                *_sentence_transformer_cache_args(args),
                *_force_args(args),
            ],
        ),
        BaselineSpec(
            name="mbert_kmeans",
            module="src.baselines.run_mbert_kmeans",
            metrics_file="mbert_kmeans_metrics.json",
            embedding_type="transformer_encoder",
            clustering_method="kmeans",
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
            embedding_type="transformer_encoder",
            clustering_method="dbscan",
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
            name="labse_kmeans",
            module="src.baselines.run_labse_kmeans",
            metrics_file="labse_kmeans_metrics.json",
            embedding_type="cross_lingual_sentence_embedding",
            clustering_method="kmeans",
            command_args=[
                *common_args,
                "--embeddings-dir",
                embeddings_dir,
                "--model-name",
                args.labse_model_name,
                *_sentence_transformer_cache_args(args),
                *_force_args(args),
            ],
        ),
        BaselineSpec(
            name="msimcse_dbscan",
            module="src.baselines.run_msimcse_dbscan",
            metrics_file="msimcse_dbscan_metrics.json",
            embedding_type="contrastive_sentence_embedding",
            clustering_method="dbscan",
            command_args=[
                *common_args,
                "--embeddings-path",
                f"{embeddings_dir}/msimcse_embeddings.npy",
                "--model_name",
                args.msimcse_model_name or "",
                *_sentence_transformer_cache_args(args),
                *_force_args(args),
                "--eps",
                str(args.eps),
                "--min_samples",
                str(args.min_samples),
            ],
        ),
    ]


def _sentence_transformer_cache_args(args: argparse.Namespace) -> list[str]:
    command_args = []
    if args.local_files_only:
        command_args.append("--local_files_only")
    if args.offline:
        command_args.append("--offline")
    return command_args


def _force_args(args: argparse.Namespace) -> list[str]:
    return ["--force"] if args.force else []


def _run_baseline(spec: BaselineSpec) -> BaselineRunResult:
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
        return BaselineRunResult(
            status="failed",
            reason=_summarize_failure(result.stderr or result.stdout),
        )

    return BaselineRunResult(status="completed")


def _summarize_failure(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Command failed without stderr output"

    for line in reversed(lines):
        if "SSLCertVerificationError" in line or "requests.exceptions.SSLError" in line:
            return (
                "Hugging Face model download failed because SSL/certificate "
                "verification failed. Use a working cache, local model path, or "
                "fix the Python certificate configuration."
            )
        if "SentenceTransformer model is not available locally" in line:
            return (
                "SentenceTransformer model is not available locally. "
                "Run once with internet access or provide a local model path."
            )
        if "sentence-transformers is not installed" in line:
            return "sentence-transformers is not installed"
        if "Could not load SentenceTransformer model" in line:
            return line.replace("RuntimeError: ", "")
        if "ModuleNotFoundError" in line:
            return line
        if "RuntimeError:" in line:
            return line

    return lines[-1][:300]


def _build_summary(
    baseline_specs: list[BaselineSpec],
    baseline_results: dict[str, BaselineRunResult],
    metrics_dir: Path,
) -> pd.DataFrame:
    rows = []

    for spec in baseline_specs:
        row = {column: None for column in SUMMARY_COLUMNS}
        row["model_name"] = spec.name
        row["embedding_type"] = spec.embedding_type
        row["clustering_method"] = spec.clustering_method
        result = baseline_results.get(
            spec.name,
            BaselineRunResult(status="not_run", reason="Baseline was not executed"),
        )
        row["status"] = result.status
        row["reason"] = result.reason

        metrics_path = metrics_dir / spec.metrics_file
        if result.status == "completed" and metrics_path.exists():
            metrics = load_json(metrics_path)
            for column in SUMMARY_COLUMNS:
                if column not in {
                    "model_name",
                    "status",
                    "reason",
                    "embedding_type",
                    "clustering_method",
                }:
                    row[column] = metrics.get(column)
        elif result.status == "completed":
            LOGGER.error("Metrics file missing for baseline: %s", spec.name)
            row["status"] = "failed"
            row["reason"] = "Metrics file missing after baseline completed"

        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


if __name__ == "__main__":
    main()
