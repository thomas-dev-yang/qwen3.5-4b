from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from common.artifact import TraceArtifact
from common.config import CompareSettings


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    compared_tensors: int
    failures: int
    report: dict[str, object]


def _canonical_cache_metadata(manifest: dict[str, object]) -> list[dict[str, object]]:
    stages = manifest.get("stages", [])
    canonical: list[dict[str, object]] = []
    for stage in stages:
        layers = []
        for layer in stage.get("cache", []):
            layers.append(
                {
                    "index": layer.get("index"),
                    "type": layer.get("type"),
                    "states": layer.get("states"),
                }
            )
        canonical.append({"name": stage.get("name"), "cache": layers})
    return canonical


def _compare_tensor(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    settings: CompareSettings,
) -> dict[str, object]:
    is_logits = name.endswith(".logits")
    atol = settings.logits_atol if is_logits else settings.state_atol
    rtol = settings.logits_rtol if is_logits else settings.state_rtol
    result: dict[str, object] = {
        "name": name,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "candidate_dtype": str(candidate.dtype).removeprefix("torch."),
        "atol": atol,
        "rtol": rtol,
    }
    if reference.shape != candidate.shape:
        return {**result, "passed": False, "reason": "shape mismatch"}

    if not (reference.is_floating_point() or reference.is_complex()):
        passed = bool(torch.equal(reference, candidate))
        return {**result, "passed": passed, "mismatch_count": int((reference != candidate).sum())}

    reference_float = reference.float()
    candidate_float = candidate.float()
    absolute = (reference_float - candidate_float).abs()
    allowed = atol + rtol * reference_float.abs()
    close = (absolute <= allowed) | (torch.isnan(reference_float) & torch.isnan(candidate_float))
    return {
        **result,
        "passed": bool(close.all()),
        "max_abs": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "mismatch_count": int((~close).sum().item()),
    }


def compare_traces(
    reference: TraceArtifact,
    candidate: TraceArtifact,
    settings: CompareSettings,
) -> ComparisonResult:
    manifest_checks = {
        "input_ids": reference.manifest.get("input_ids") == candidate.manifest.get("input_ids"),
        "attention_mask": (
            reference.manifest.get("attention_mask") == candidate.manifest.get("attention_mask")
        ),
        "decode_token_ids": (
            reference.manifest.get("decode_token_ids") == candidate.manifest.get("decode_token_ids")
        ),
        "model_repo_id": (
            reference.manifest.get("model", {}).get("repo_id")
            == candidate.manifest.get("model", {}).get("repo_id")
        ),
        "model_revision": (
            reference.manifest.get("model", {}).get("revision")
            == candidate.manifest.get("model", {}).get("revision")
        ),
        "cache_metadata": (
            _canonical_cache_metadata(reference.manifest)
            == _canonical_cache_metadata(candidate.manifest)
        ),
    }
    reference_names = set(reference.tensors)
    candidate_names = set(candidate.tensors)
    missing = sorted(reference_names - candidate_names)
    unexpected = sorted(candidate_names - reference_names)

    tensor_results = [
        _compare_tensor(name, reference.tensors[name], candidate.tensors[name], settings)
        for name in sorted(reference_names & candidate_names)
    ]
    failed_tensors = [result for result in tensor_results if not result["passed"]]
    failed_manifest_checks = [name for name, passed in manifest_checks.items() if not passed]
    passed = all(manifest_checks.values()) and not missing and not unexpected and not failed_tensors
    report = {
        "passed": passed,
        "reference": str(reference.path),
        "candidate": str(candidate.path),
        "manifest_checks": manifest_checks,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "summary": {
            "compared_tensors": len(tensor_results),
            "failed_tensors": len(failed_tensors),
            "failed_manifest_checks": failed_manifest_checks,
        },
        "tensors": tensor_results,
    }
    return ComparisonResult(
        passed=passed,
        compared_tensors=len(tensor_results),
        failures=(
            len(failed_manifest_checks) + len(failed_tensors) + len(missing) + len(unexpected)
        ),
        report=report,
    )


def save_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
