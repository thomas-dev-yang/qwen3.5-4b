from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def torch_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by config.toml, but torch cannot see a CUDA device")
    return device


def environment_metadata(device: torch.device) -> dict[str, object]:
    import transformers

    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return metadata


def load_messages(path: Path, fixture: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        fixtures = json.load(file)
    try:
        return fixtures[fixture]
    except KeyError as error:
        raise KeyError(
            f"Unknown prompt fixture {fixture!r}; choices: {sorted(fixtures)}"
        ) from error
