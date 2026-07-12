.PHONY: setup doctor download test lint check benchmark pack

setup:
	bash scripts/setup_cloud.sh

doctor:
	uv run --locked qwen35 doctor

download:
	uv run --locked qwen35 download

test:
	uv run --locked pytest

lint:
	uv run --locked ruff check .

check:
	bash scripts/check_correctness.sh

benchmark:
	uv run --locked qwen35 benchmark --prompt-length 128 --decode-steps 32

pack:
	bash scripts/pack_for_cloud.sh
