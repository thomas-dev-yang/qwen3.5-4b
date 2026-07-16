import pytest
import torch

from cuda_impl.correctness import tensor_error


def test_tensor_error_reports_location_and_top_token() -> None:
    reference = torch.tensor([[[1.0, 3.0, 2.0]]])
    candidate = torch.tensor([[[1.0, 2.5, 4.0]]])

    result = tensor_error(
        reference,
        candidate,
        atol=0.01,
        rtol=0.01,
        include_top_token=True,
    )

    assert result["passed"] is False
    assert result["max_abs"] == pytest.approx(2.0)
    assert result["max_location"] == [0, 0, 2]
    assert result["reference_top_token"] == 1
    assert result["candidate_top_token"] == 2
    assert result["top_token_agrees"] is False


def test_tensor_error_accepts_values_within_tolerance() -> None:
    reference = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[1.001, 1.999]])

    result = tensor_error(
        reference,
        candidate,
        atol=0.01,
        rtol=0.0,
        include_top_token=False,
    )

    assert result["passed"] is True
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
