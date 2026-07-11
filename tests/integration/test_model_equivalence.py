import os

import pytest

from common.config import load_settings
from workflows import compare_artifacts, write_candidate_trace, write_reference_trace

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("QWEN35_RUN_INTEGRATION") != "1",
    reason="set QWEN35_RUN_INTEGRATION=1 on a CUDA machine with the checkpoint downloaded",
)
def test_hf_smoke_trace_matches_reference(tmp_path) -> None:
    settings = load_settings()
    reference = write_reference_trace(settings, tmp_path / "reference", "default")
    candidate = write_candidate_trace(
        settings,
        reference,
        tmp_path / "candidate",
    )
    result = compare_artifacts(settings, reference, candidate, tmp_path / "report.json")

    assert result.passed
