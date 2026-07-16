#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_arguments(
    arguments: list[str],
    source: Path,
    cuda_root: Path,
    python_include: Path,
) -> list[str]:
    normalized: list[str] = ["/usr/bin/clang++"]
    skip = {
        "--generate-dependencies-with-compile",
        "--dependency-output",
        "--compiler-options",
        "--expt-relaxed-constexpr",
    }

    for argument in arguments[1:]:
        if argument in skip or argument.startswith("-gencode="):
            continue
        if argument == "--use_fast_math":
            normalized.append("-ffast-math")
            continue
        if argument == "/usr/local/cuda/include":
            normalized.append(str(cuda_root / "include"))
            continue
        if argument in {"/usr/include/python3.11", "/opt/conda/include/python3.11"}:
            normalized.append(str(python_include))
            continue
        normalized.append(argument)

    if source.suffix == ".cu":
        normalized[1:1] = [
            "-x",
            "cuda",
            f"--cuda-path={cuda_root}",
            "--cuda-gpu-arch=sm_90a",
            "-nocudalib",
            "-isystem",
            str(python_include.parent),
        ]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--python-include", type=Path, required=True)
    args = parser.parse_args()

    database = json.loads(args.database.read_text(encoding="utf-8"))
    for entry in database:
        source = Path(entry["file"])
        entry["directory"] = str(args.project_root)
        entry["arguments"] = normalize_arguments(
            entry["arguments"],
            source,
            args.cuda_root,
            args.python_include,
        )
        entry.pop("command", None)

    args.database.write_text(json.dumps(database, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
