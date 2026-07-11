import torch

from common.artifact import load_trace, save_trace
from common.compare import compare_traces
from common.config import load_settings


def _manifest():
    return {
        "backend": "test",
        "input_ids": [[1, 2]],
        "attention_mask": [[1, 1]],
        "decode_token_ids": [3],
        "model": {"revision": "abc"},
    }


def test_artifact_round_trip_and_exact_compare(tmp_path) -> None:
    tensors = {
        "stage.000_prefill.logits": torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16),
        "stage.000_prefill.cache.layer_00.linear.recurrent": torch.tensor([3.0]),
    }
    reference_path = save_trace(tmp_path / "reference", _manifest(), tensors)
    candidate_path = save_trace(tmp_path / "candidate", _manifest(), tensors)

    reference = load_trace(reference_path)
    candidate = load_trace(candidate_path)
    result = compare_traces(reference, candidate, load_settings().compare)

    assert result.passed
    assert result.compared_tensors == 2


def test_compare_reports_first_numerical_boundary(tmp_path) -> None:
    reference_path = save_trace(
        tmp_path / "reference",
        _manifest(),
        {"stage.000_prefill.logits": torch.tensor([[[1.0]]])},
    )
    candidate_path = save_trace(
        tmp_path / "candidate",
        _manifest(),
        {"stage.000_prefill.logits": torch.tensor([[[2.0]]])},
    )

    result = compare_traces(
        load_trace(reference_path),
        load_trace(candidate_path),
        load_settings().compare,
    )

    assert not result.passed
    assert result.report["tensors"][0]["name"] == "stage.000_prefill.logits"
    assert result.report["tensors"][0]["max_abs"] == 1.0


def test_metadata_only_cache_shape_mismatch_fails(tmp_path) -> None:
    reference_manifest = {
        **_manifest(),
        "stages": [
            {
                "name": "stage.000_prefill",
                "cache": [
                    {
                        "index": 0,
                        "type": "linear_attention",
                        "cache_class": "ReferenceCache",
                        "states": {"linear.recurrent": {"shape": [1, 32, 128, 128]}},
                    }
                ],
            }
        ],
    }
    candidate_manifest = {
        **reference_manifest,
        "stages": [
            {
                "name": "stage.000_prefill",
                "cache": [
                    {
                        "index": 0,
                        "type": "linear_attention",
                        "cache_class": "NativeCache",
                        "states": {"linear.recurrent": {"shape": [1, 32, 64, 128]}},
                    }
                ],
            }
        ],
    }
    tensor = {"stage.000_prefill.logits": torch.tensor([[[1.0]]])}
    reference = load_trace(save_trace(tmp_path / "reference", reference_manifest, tensor))
    candidate = load_trace(save_trace(tmp_path / "candidate", candidate_manifest, tensor))

    result = compare_traces(reference, candidate, load_settings().compare)

    assert not result.passed
    assert result.failures == 1
    assert result.report["summary"]["failed_manifest_checks"] == ["cache_metadata"]
