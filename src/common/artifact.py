from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1


@dataclass
class TraceArtifact:
    path: Path
    manifest: dict[str, object]
    tensors: dict[str, torch.Tensor]


def _tensor_manifest(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in sorted(tensors.items())
    }


def save_trace(
    path: str | Path, manifest: dict[str, object], tensors: dict[str, torch.Tensor]
) -> Path:
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    cpu_tensors = {name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()}
    manifest = {
        **manifest,
        "schema_version": SCHEMA_VERSION,
        "tensors": _tensor_manifest(cpu_tensors),
    }
    save_file(cpu_tensors, temporary / "tensors.safetensors")
    with (temporary / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")

    if output.exists():
        shutil.rmtree(output)
    temporary.replace(output)
    return output


def load_trace(path: str | Path) -> TraceArtifact:
    artifact_path = Path(path)
    with (artifact_path / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        actual_schema = manifest.get("schema_version")
        raise ValueError(f"Unsupported trace schema {actual_schema!r}; expected {SCHEMA_VERSION}")
    tensors = load_file(artifact_path / "tensors.safetensors", device="cpu")
    return TraceArtifact(path=artifact_path, manifest=manifest, tensors=tensors)
