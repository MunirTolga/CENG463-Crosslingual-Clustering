from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.io import repair_string_array_file
from utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_LANGUAGE_FILES = [
    "outputs/embeddings/supcon_languages.npy",
    "outputs/embeddings/mbert_languages.npy",
    "outputs/embeddings/minilm_languages.npy",
    "outputs/embeddings/msimcse_languages.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair legacy object-dtype language .npy arrays.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_LANGUAGE_FILES,
        help="Language .npy files to repair. Missing files are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for file_name in args.files:
        path = _resolve_path(file_name)
        if not path.exists():
            LOGGER.info("Skipped missing language array: %s", path)
            continue

        repaired_array = repair_string_array_file(path)
        LOGGER.info("Repaired language array: %s", path)
        LOGGER.info("dtype after repair: %s", repaired_array.dtype)
        LOGGER.info("first 10 values: %s", repaired_array[:10].tolist())
        print(f"{path}")
        print(f"dtype after repair: {repaired_array.dtype}")
        print(f"first 10 values: {repaired_array[:10].tolist()}")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
