from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a JSON file."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {input_path}")

    return data


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to a .npy file."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.save(output_path, array)


def load_numpy(path, allow_pickle=False):
    path = Path(path)
    return np.load(path, allow_pickle=allow_pickle)


def save_string_array(values, path: str | Path) -> None:
    """Save string values as a safe NumPy Unicode array, not an object array."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    array = np.asarray(values, dtype=str)
    np.save(output_path, array)


def load_string_array(path: str | Path) -> np.ndarray:
    """Load a NumPy string array safely.

    Legacy language arrays in this project may have been saved with dtype=object.
    For normal string arrays we keep allow_pickle=False. Only when NumPy raises
    the known object-array error do we fall back to allow_pickle=True, cast to
    strings immediately, and return a safe string array.
    """

    input_path = Path(path)
    try:
        return np.load(input_path, allow_pickle=False).astype(str)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise

        legacy_array = np.load(input_path, allow_pickle=True)
        return legacy_array.astype(str)


def repair_string_array_file(path: str | Path) -> np.ndarray:
    """Convert a legacy object-dtype string .npy file into a safe Unicode array."""

    repaired_array = load_string_array(path)
    save_string_array(repaired_array, path)
    return repaired_array
