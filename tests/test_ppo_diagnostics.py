import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

pytestmark = pytest.mark.skipif(
    torch is None, reason="PPO diagnostic helpers require PyTorch"
)

if torch is not None:
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from train.ppo import (  # noqa: E402
        _explained_variance,
        _gradient_cosine,
        _tensor_group_norm,
    )


def test_tensor_group_norm_ignores_absent_gradients():
    device = torch.device("cpu")
    norm = _tensor_group_norm(
        (torch.tensor([3.0]), None, torch.tensor([4.0])), device
    )
    assert torch.isclose(norm, torch.tensor(5.0))


def test_gradient_cosine_uses_matching_gradient_tensors():
    device = torch.device("cpu")
    left = (torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 1.0]))
    same = (torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 1.0]))
    opposite = (torch.tensor([-1.0, 0.0]), None, torch.tensor([0.0, -1.0]))
    assert torch.isclose(_gradient_cosine(left, same, device), torch.tensor(1.0))
    assert torch.isclose(
        _gradient_cosine(left, opposite, device), torch.tensor(-1.0)
    )


def test_explained_variance_perfect_constant_and_worse_predictions():
    target = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isclose(_explained_variance(target, target), torch.tensor(1.0))
    assert torch.isclose(
        _explained_variance(torch.full_like(target, 2.0), target),
        torch.tensor(0.0),
    )
    assert _explained_variance(-target, target) < 0.0
