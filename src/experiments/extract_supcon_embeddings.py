from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.dataset import XNLISupConDataset, xnli_supcon_collate_fn
from data.preprocess import load_dataframe
from embeddings.transformer_embedder import save_embeddings
from models.supcon_model import SupConMBertModel
from utils.io import ensure_dir, save_numpy, save_string_array
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract encoder embeddings from a trained SupCon mBERT checkpoint."
    )
    parser.add_argument("--input-file", default="data/processed/xnli_validation_en_tr.csv")
    parser.add_argument("--checkpoint-path", default="outputs/checkpoints/supcon_mbert_final.pt")
    parser.add_argument("--output-dir", default="outputs/embeddings")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = ensure_dir(ROOT / args.output_dir)
    checkpoint_path = ROOT / args.checkpoint_path
    input_path = ROOT / args.input_file

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Loading checkpoint from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", "bert-base-multilingual-cased")
    pooling = config.get("pooling", "mean")
    projection_dim = int(config.get("projection_dim", 128))
    max_length = args.max_length or int(config.get("max_length", 128))

    LOGGER.info("Loading data from: %s", input_path)
    dataframe = load_dataframe(input_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = XNLISupConDataset(dataframe, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=xnli_supcon_collate_fn,
    )

    model = SupConMBertModel(
        model_name=model_name,
        pooling=pooling,
        projection_dim=projection_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings = extract_encoder_embeddings(model, dataloader, device)
    labels = dataframe["label_id"].to_numpy(dtype=np.int64)
    languages = dataframe["language"].astype(str).to_numpy()

    embeddings_path = output_dir / "supcon_mbert_embeddings.npy"
    labels_path = output_dir / "supcon_labels.npy"
    languages_path = output_dir / "supcon_languages.npy"

    save_embeddings(embeddings, embeddings_path)
    save_numpy(labels, labels_path)
    save_string_array(languages, languages_path)

    LOGGER.info("Saved encoder embeddings to: %s", embeddings_path)
    LOGGER.info("Saved labels to: %s", labels_path)
    LOGGER.info("Saved languages to: %s", languages_path)


@torch.no_grad()
def extract_encoder_embeddings(
    model: SupConMBertModel,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_embeddings: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Extracting SupCon encoder embeddings"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        encoder_embeddings, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        all_embeddings.append(encoder_embeddings.cpu().numpy())

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


if __name__ == "__main__":
    main()
