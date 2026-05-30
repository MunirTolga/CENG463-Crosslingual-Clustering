from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import XNLISupConDataset, xnli_supcon_collate_fn
from src.models.supcon_model import SupConMBertModel
from src.training.losses import SupervisedContrastiveLoss
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


LABEL_NAME_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

LABEL_ID_TO_NAME = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train mBERT with supervised contrastive loss."
    )

    parser.add_argument(
        "--train_file",
        default="data/processed/xnli_train.csv",
        help="Path to processed training CSV file.",
    )
    parser.add_argument(
        "--config",
        default="configs/supcon_mbert.yaml",
        help="Path to supervised contrastive training config.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/checkpoints",
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--filter_source_split",
        default=None,
        help=(
            "Optional source_split filter. Example: --filter_source_split train. "
            "Leave empty if your prepared train CSV intentionally contains mixed source_split values."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_path(args.config)
    train_file = _resolve_path(args.train_file)
    output_dir = _ensure_dir(_resolve_path(args.output_dir))
    metrics_dir = _ensure_dir(ROOT / "outputs" / "metrics")

    config = load_config(config_path)

    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    model_name = str(config.get("model_name", "bert-base-multilingual-cased"))
    batch_size = args.batch_size or int(config.get("batch_size", 16))
    learning_rate = args.learning_rate or float(config.get("learning_rate", 2e-5))
    epochs = args.epochs or int(config.get("num_epochs", 1))
    max_length = args.max_length or int(config.get("max_length", 128))
    temperature = float(config.get("temperature", 0.07))
    projection_dim = int(config.get("projection_dim", 128))
    pooling = str(config.get("pooling", "mean"))
    max_samples_per_language = config.get("max_samples_per_language")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Loading config from: %s", config_path)
    LOGGER.info("Loading training data from: %s", train_file)

    dataframe = _load_training_dataframe(
        train_file=train_file,
        filter_source_split=args.filter_source_split,
    )

    dataframe = _limit_samples_per_language(
        dataframe=dataframe,
        max_samples_per_language=max_samples_per_language,
        seed=seed,
    )

    _log_dataframe_summary(dataframe)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    LOGGER.info("Creating XNLISupConDataset...")
    dataset = XNLISupConDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=xnli_supcon_collate_fn,
    )

    LOGGER.info("Initializing model: %s", model_name)

    model = SupConMBertModel(
        model_name=model_name,
        pooling=pooling,
        projection_dim=projection_dim,
    ).to(device)

    loss_fn = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    history = train_supcon_epochs(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        output_dir=output_dir,
        config=config,
    )

    loss_path = metrics_dir / "supcon_training_loss.csv"
    pd.DataFrame(history).to_csv(loss_path, index=False)
    LOGGER.info("Saved training loss history to: %s", loss_path)


def train_supcon_epochs(
    model: SupConMBertModel,
    dataloader: DataLoader,
    loss_fn: SupervisedContrastiveLoss,
    optimizer: AdamW,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []

    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_id"].to(device)

            optimizer.zero_grad(set_to_none=True)

            _, projected_embeddings = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn(projected_embeddings, labels)

            if torch.isnan(loss):
                raise RuntimeError(
                    "Training stopped because loss became NaN. "
                    "Try increasing batch_size or lowering learning_rate."
                )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

            progress.set_postfix(loss=total_loss / max(num_batches, 1))

        average_loss = total_loss / max(num_batches, 1)

        print(f"Epoch {epoch}/{epochs} average loss: {average_loss:.6f}")
        LOGGER.info("Epoch %d/%d average loss: %.6f", epoch, epochs, average_loss)

        checkpoint_path = output_dir / f"supcon_mbert_epoch_{epoch}.pt"

        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            average_loss=average_loss,
            config=config,
        )

        history.append(
            {
                "epoch": float(epoch),
                "average_loss": float(average_loss),
            }
        )

    final_checkpoint_path = output_dir / "supcon_mbert_final.pt"

    _save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        average_loss=history[-1]["average_loss"] if history else 0.0,
        config=config,
    )

    return history


def _load_training_dataframe(
    train_file: Path,
    filter_source_split: str | None = None,
) -> pd.DataFrame:
    if not train_file.exists():
        raise FileNotFoundError(f"Training file does not exist: {train_file}")

    dataframe = pd.read_csv(train_file)

    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    LOGGER.info("Original dataframe columns: %s", dataframe.columns.tolist())
    LOGGER.info("Original dataframe shape: %s", dataframe.shape)

    if filter_source_split is not None:
        if "source_split" not in dataframe.columns:
            raise ValueError(
                "--filter_source_split was provided, but the CSV has no source_split column."
            )

        before_count = len(dataframe)
        dataframe = dataframe[
            dataframe["source_split"].astype(str) == str(filter_source_split)
        ].copy()

        LOGGER.info(
            "Filtered source_split=%s: %d -> %d rows",
            filter_source_split,
            before_count,
            len(dataframe),
        )

        if dataframe.empty:
            raise ValueError(
                f"No rows left after filtering source_split={filter_source_split}."
            )

    dataframe = _prepare_required_columns(dataframe)
    dataframe = _remove_invalid_rows(dataframe)

    if dataframe.empty:
        raise ValueError("Training dataframe is empty after preprocessing.")

    return dataframe.reset_index(drop=True)


def _prepare_required_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "text" not in dataframe.columns:
        if {"premise", "hypothesis"}.issubset(dataframe.columns):
            dataframe["text"] = (
                dataframe["premise"].fillna("").astype(str)
                + " [SEP] "
                + dataframe["hypothesis"].fillna("").astype(str)
            )
        else:
            raise ValueError(
                "Training data must contain either 'text' column or both "
                "'premise' and 'hypothesis' columns."
            )

    if "label_id" not in dataframe.columns:
        if "label" in dataframe.columns:
            dataframe["label_id"] = dataframe["label"]
        elif "label_name" in dataframe.columns:
            dataframe["label_id"] = (
                dataframe["label_name"]
                .astype(str)
                .str.lower()
                .map(LABEL_NAME_TO_ID)
            )
        else:
            raise ValueError(
                "Training data must contain one of these columns: "
                "'label_id', 'label', or 'label_name'."
            )

    if "label_name" not in dataframe.columns:
        dataframe["label_name"] = dataframe["label_id"].map(LABEL_ID_TO_NAME)

    if "language" not in dataframe.columns:
        LOGGER.warning(
            "Column 'language' is missing. Filling it with 'unknown'. "
            "Training can continue, but language-based analysis will be weak."
        )
        dataframe["language"] = "unknown"

    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    dataframe["language"] = dataframe["language"].fillna("unknown").astype(str)

    dataframe["label_id"] = pd.to_numeric(
        dataframe["label_id"],
        errors="coerce",
    )

    return dataframe


def _remove_invalid_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    before_count = len(dataframe)

    dataframe = dataframe.dropna(subset=["text", "label_id"]).copy()
    dataframe = dataframe[dataframe["text"].str.strip() != ""].copy()
    dataframe["label_id"] = dataframe["label_id"].astype(int)

    valid_label_ids = set(LABEL_ID_TO_NAME.keys())
    dataframe = dataframe[dataframe["label_id"].isin(valid_label_ids)].copy()

    after_count = len(dataframe)

    if after_count < before_count:
        LOGGER.warning(
            "Removed %d invalid rows from training dataframe.",
            before_count - after_count,
        )

    return dataframe


def _limit_samples_per_language(
    dataframe: pd.DataFrame,
    max_samples_per_language: Any,
    seed: int,
) -> pd.DataFrame:
    if max_samples_per_language is None:
        return dataframe

    if "language" not in dataframe.columns:
        LOGGER.warning(
            "Cannot limit samples per language because 'language' column is missing."
        )
        return dataframe

    max_samples = int(max_samples_per_language)

    if max_samples <= 0:
        return dataframe

    limited_frames = []

    for _, group in dataframe.groupby("language", sort=False):
        sample_size = min(len(group), max_samples)
        limited_frames.append(group.sample(n=sample_size, random_state=seed))

    limited_dataframe = pd.concat(limited_frames, ignore_index=True)

    LOGGER.info(
        "Applied max_samples_per_language=%d: %d -> %d rows",
        max_samples,
        len(dataframe),
        len(limited_dataframe),
    )

    return limited_dataframe


def _log_dataframe_summary(dataframe: pd.DataFrame) -> None:
    LOGGER.info("Final training dataframe columns: %s", dataframe.columns.tolist())
    LOGGER.info("Final training dataframe shape: %s", dataframe.shape)
    LOGGER.info("Training examples: %d", len(dataframe))

    if "language" in dataframe.columns:
        LOGGER.info(
            "Language distribution:\n%s",
            dataframe["language"].value_counts().to_string(),
        )

    if "label_name" in dataframe.columns:
        LOGGER.info(
            "Label distribution:\n%s",
            dataframe["label_name"].value_counts().to_string(),
        )
    elif "label_id" in dataframe.columns:
        LOGGER.info(
            "Label distribution:\n%s",
            dataframe["label_id"].value_counts().to_string(),
        )

    if "source_split" in dataframe.columns:
        LOGGER.warning(
            "source_split distribution:\n%s",
            dataframe["source_split"].value_counts().to_string(),
        )
        LOGGER.warning(
            "If this file is used for training, make sure you are not accidentally "
            "training on validation/test data."
        )

    LOGGER.info("Training dataframe preview:\n%s", dataframe.head().to_string())


def _save_checkpoint(
    checkpoint_path: Path,
    model: SupConMBertModel,
    optimizer: AdamW,
    epoch: int,
    average_loss: float,
    config: dict[str, Any],
) -> None:
    _ensure_dir(checkpoint_path.parent)

    torch.save(
        {
            "epoch": epoch,
            "average_loss": average_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )

    LOGGER.info("Saved checkpoint to: %s", checkpoint_path)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()