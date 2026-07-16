from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from common.benchmark import benchmark_comparison
from common.config import load_settings
from cuda_impl.correctness import validate_cuda_model
from hf_impl.model import model_source, validate_checkpoint_config
from workflows import (
    compare_artifacts,
    download_checkpoint,
    write_candidate_trace,
    write_reference_trace,
)


def _path(value: str) -> Path:
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen35", description="Qwen3.5-4B differential execution lab"
    )
    parser.add_argument("--config", default="config.toml", help="Path to repository config.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "doctor", help="Check Python, Torch, CUDA, Transformers, and local model state"
    )
    subparsers.add_parser("download", help="Download the pinned Qwen3.5-4B snapshot")

    reference = subparsers.add_parser("reference", help="Write a Hugging Face reference trace")
    reference.add_argument("--fixture", default="default")
    reference.add_argument("--output", type=_path, default=_path("artifacts/correctness/reference"))

    candidate = subparsers.add_parser("candidate", help="Write a candidate trace")
    candidate.add_argument(
        "--reference", type=_path, default=_path("artifacts/correctness/reference")
    )
    candidate.add_argument("--output", type=_path, default=_path("artifacts/correctness/candidate"))

    compare = subparsers.add_parser("compare", help="Compare reference and candidate traces")
    compare.add_argument(
        "--reference", type=_path, default=_path("artifacts/correctness/reference")
    )
    compare.add_argument(
        "--candidate", type=_path, default=_path("artifacts/correctness/candidate")
    )
    compare.add_argument("--report", type=_path, default=_path("artifacts/correctness/report.json"))

    benchmark = subparsers.add_parser(
        "benchmark", help="Compare Transformers and custom-attention Qwen decode"
    )
    benchmark.add_argument("--prompt-length", type=int, default=1024)
    benchmark.add_argument("--decode-steps", type=int, default=None)
    benchmark.add_argument(
        "--attention-version", choices=("v1", "v2", "v3", "v4", "v5"), default="v5"
    )
    benchmark.add_argument("--output", type=_path, default=None)

    validate_cuda = subparsers.add_parser(
        "validate-cuda", help="Trace the custom attention through every Qwen layer"
    )
    validate_cuda.add_argument(
        "--attention-version", choices=("v1", "v2", "v3", "v4", "v5"), default="v5"
    )
    validate_cuda.add_argument("--prompt-length", type=int, default=128)
    validate_cuda.add_argument("--decode-steps", type=int, default=4)
    validate_cuda.add_argument(
        "--output", type=_path, default=_path("artifacts/correctness/cuda-model.json")
    )
    return parser


def _doctor(settings) -> int:
    result: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "configured_device": settings.runtime.device,
        "model_dir": str(settings.model.local_dir),
        "model_downloaded": (settings.model.local_dir / "config.json").is_file(),
    }
    try:
        import transformers
        from transformers import Qwen3_5ForConditionalGeneration

        result["transformers"] = transformers.__version__
        result["qwen35_class"] = Qwen3_5ForConditionalGeneration.__name__
        if settings.runtime.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("config.toml requires CUDA, but torch.cuda.is_available() is false")
        if result["model_downloaded"]:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_source(settings))
            validate_checkpoint_config(config, settings)
            result["checkpoint_config"] = "ok"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)

    if args.command == "doctor":
        return _doctor(settings)
    if args.command == "download":
        print(download_checkpoint(settings))
        return 0
    if args.command == "reference":
        print(write_reference_trace(settings, args.output, args.fixture))
        return 0
    if args.command == "candidate":
        print(write_candidate_trace(settings, args.reference, args.output))
        return 0
    if args.command == "compare":
        result = compare_artifacts(settings, args.reference, args.candidate, args.report)
        summary = {
            "passed": result.passed,
            "compared_tensors": result.compared_tensors,
            "failures": result.failures,
            "report": str(args.report),
        }
        if not result.passed:
            failed = [tensor for tensor in result.report["tensors"] if not tensor["passed"]]
            summary["first_failure"] = failed[0] if failed else result.report["summary"]
        print(json.dumps(summary, indent=2))
        return 0 if result.passed else 1
    if args.command == "benchmark":
        decode_steps = args.decode_steps or settings.benchmark.decode_steps
        report = benchmark_comparison(
            settings, args.prompt_length, decode_steps, args.attention_version
        )
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if report["correctness"]["passed"] else 1
    if args.command == "validate-cuda":
        report = validate_cuda_model(
            settings,
            version=args.attention_version,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
        )
        rendered = json.dumps(report, indent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    **report["summary"],
                    "report": str(args.output),
                },
                indent=2,
            )
        )
        return 0 if report["passed"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
