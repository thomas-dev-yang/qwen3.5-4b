from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from common.artifact import load_trace, save_trace
from common.compare import ComparisonResult, compare_traces, save_report
from common.config import Settings
from common.system import environment_metadata, load_messages, require_device, seed_everything
from common.trace import run_trace
from hf_impl.model import REVISION_FILE, load_hf_engine, load_tokenizer, tokenize_fixture
from torch_impl.model import load_torch_engine


def model_manifest(settings: Settings) -> dict[str, object]:
    return {
        "repo_id": settings.model.repo_id,
        "revision": settings.model.revision,
        "text_only": True,
        "dtype": settings.runtime.dtype,
        "attention_backend": settings.runtime.attention_backend,
    }


def download_checkpoint(settings: Settings) -> Path:
    settings.model.local_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=settings.model.repo_id,
        revision=settings.model.revision,
        local_dir=settings.model.local_dir,
    )
    (settings.model.local_dir / REVISION_FILE).write_text(
        settings.model.revision + "\n",
        encoding="utf-8",
    )
    return Path(path)


def write_reference_trace(settings: Settings, output: Path, fixture: str) -> Path:
    seed_everything(settings.runtime.seed)
    device = require_device(settings.runtime.device)
    messages = load_messages(settings.root / "fixtures" / "prompts.json", fixture)
    tokenizer = load_tokenizer(settings)
    input_ids, attention_mask = tokenize_fixture(tokenizer, messages, settings)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    engine = load_hf_engine(settings, name="huggingface-reference")
    trace = run_trace(
        engine=engine,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=settings.trace.decode_steps,
        decode_token_ids=None,
        cache_capture=settings.trace.capture_cache,
        capture_hidden_states=settings.trace.capture_hidden_states,
    )
    manifest = {
        **trace.manifest,
        "kind": "reference",
        "fixture": fixture,
        "messages": messages,
        "model": model_manifest(settings),
        "environment": environment_metadata(device),
    }
    return save_trace(output, manifest, trace.tensors)


def write_candidate_trace(
    settings: Settings,
    reference_path: Path,
    output: Path,
) -> Path:
    seed_everything(settings.runtime.seed)
    device = require_device(settings.runtime.device)
    reference = load_trace(reference_path)
    input_ids = torch.tensor(reference.manifest["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        reference.manifest["attention_mask"], dtype=torch.long, device=device
    )
    decode_token_ids = [int(token) for token in reference.manifest["decode_token_ids"]]
    engine = load_torch_engine(settings, name="torch-candidate")
    trace = run_trace(
        engine=engine,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=len(decode_token_ids),
        decode_token_ids=decode_token_ids,
        cache_capture=settings.trace.capture_cache,
        capture_hidden_states=settings.trace.capture_hidden_states,
    )
    manifest = {
        **trace.manifest,
        "kind": "candidate",
        "candidate_backend": "torch",
        "model": model_manifest(settings),
        "environment": environment_metadata(device),
    }
    return save_trace(output, manifest, trace.tensors)


def compare_artifacts(
    settings: Settings,
    reference_path: Path,
    candidate_path: Path,
    report_path: Path,
) -> ComparisonResult:
    result = compare_traces(
        load_trace(reference_path), load_trace(candidate_path), settings.compare
    )
    save_report(report_path, result.report)
    return result
