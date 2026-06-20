from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.dbscan import run_dbscan
from data.dataset import XNLISupConDataset, xnli_supcon_collate_fn
from data.load_xnli import load_xnli_dataframe
from data.preprocess import save_dataframe
from evaluation.metrics import evaluate_clustering, save_metrics
from experiments.extract_supcon_embeddings import extract_encoder_embeddings
from models.supcon_model import SupConMBertModel
from training.losses import SupervisedContrastiveLoss
from training.train_supcon import train_supcon_epochs
from utils.io import ensure_dir
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)

DEFAULT_ABLATIONS = [
    (["en", "de", "fr"], "tr"),
    (["en", "tr", "fr"], "de"),
    (["en", "tr", "de"], "ar"),
]


@dataclass(frozen=True)
class AblationConfig:
    train_languages: list[str]
    test_language: str
    dataset_name: str
    train_split: str
    test_split: str
    max_train_samples_per_language: int
    max_test_samples_per_language: int
    model_name: str
    pooling: str
    projection_dim: int
    temperature: float
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    eps: float
    min_samples: int
    seed: int
    checkpoints_dir: Path
    metrics_dir: Path
    reports_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run language ablation for SupCon mBERT clustering."
    )
    parser.add_argument("--train_languages", default="en,de,fr")
    parser.add_argument("--test_language", default="tr")
    parser.add_argument("--run_all", action="store_true", help="Run the three predefined ablations.")
    parser.add_argument("--dataset_name", default="facebook/xnli")
    parser.add_argument("--train_split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--test_split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_train_samples_per_language", type=int, default=50)
    parser.add_argument("--max_test_samples_per_language", type=int, default=100)
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints_dir", default="outputs/checkpoints")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--reports_dir", default="outputs/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ablations = DEFAULT_ABLATIONS if args.run_all else [(_parse_languages(args.train_languages), args.test_language)]

    for train_languages, test_language in ablations:
        config = AblationConfig(
            train_languages=train_languages,
            test_language=test_language,
            dataset_name=args.dataset_name,
            train_split=args.train_split,
            test_split=args.test_split,
            max_train_samples_per_language=args.max_train_samples_per_language,
            max_test_samples_per_language=args.max_test_samples_per_language,
            model_name=args.model_name,
            pooling=args.pooling,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            eps=args.eps,
            min_samples=args.min_samples,
            seed=args.seed,
            checkpoints_dir=ensure_dir(ROOT / args.checkpoints_dir),
            metrics_dir=ensure_dir(ROOT / args.metrics_dir),
            reports_dir=ensure_dir(ROOT / args.reports_dir),
        )
        run_language_ablation(config)


def run_language_ablation(config: AblationConfig) -> dict:
    LOGGER.info(
        "Running ablation: train_languages=%s, test_language=%s",
        ",".join(config.train_languages),
        config.test_language,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    train_df = load_xnli_dataframe(
        dataset_name=config.dataset_name,
        languages=config.train_languages,
        split=config.train_split,
        max_samples_per_language=config.max_train_samples_per_language,
        random_seed=config.seed,
    )
    test_df = load_xnli_dataframe(
        dataset_name=config.dataset_name,
        languages=[config.test_language],
        split=config.test_split,
        max_samples_per_language=config.max_test_samples_per_language,
        random_seed=config.seed,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_loader = _build_dataloader(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = _build_dataloader(
        dataframe=test_df,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = SupConMBertModel(
        model_name=config.model_name,
        pooling=config.pooling,
        projection_dim=config.projection_dim,
    ).to(device)
    loss_fn = SupervisedContrastiveLoss(temperature=config.temperature)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    run_name = f"ablation_{config.test_language}"
    checkpoint_dir = ensure_dir(config.checkpoints_dir / run_name)
    train_supcon_epochs(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        output_dir=checkpoint_dir,
        config=_checkpoint_config(config),
    )

    embeddings = extract_encoder_embeddings(model, test_loader, device)
    true_labels = test_df["label_id"].to_numpy()
    cluster_labels, _ = run_dbscan(
        embeddings,
        eps=config.eps,
        min_samples=config.min_samples,
    )

    metrics = evaluate_clustering(embeddings, true_labels, cluster_labels)
    metrics.update(
        {
            "train_languages": config.train_languages,
            "test_language": config.test_language,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "eps": config.eps,
            "min_samples": config.min_samples,
        }
    )

    metrics_path = config.metrics_dir / f"ablation_{config.test_language}.json"
    predictions_path = config.reports_dir / f"ablation_{config.test_language}_predictions.csv"

    save_metrics(metrics, metrics_path)
    predictions = test_df.copy()
    predictions["cluster_label"] = cluster_labels
    save_dataframe(predictions, predictions_path)

    LOGGER.info("Saved ablation metrics to: %s", metrics_path)
    LOGGER.info("Saved ablation predictions to: %s", predictions_path)
    return metrics


def _build_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = XNLISupConDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=xnli_supcon_collate_fn,
    )


def _checkpoint_config(config: AblationConfig) -> dict:
    return {
        "model_name": config.model_name,
        "pooling": config.pooling,
        "projection_dim": config.projection_dim,
        "temperature": config.temperature,
        "train_languages": config.train_languages,
        "test_language": config.test_language,
        "max_length": config.max_length,
    }


def _parse_languages(value: str) -> list[str]:
    return [language.strip() for language in value.split(",") if language.strip()]


if __name__ == "__main__":
    main()
