"""Our autograd must agree with (1) finite differences and (2) PyTorch."""
import math

import pytest
import torch

from shoprank.scratch.autograd import Value
from shoprank.scratch.mlp import MLP, train_step


def expression(a, b, c):
    """Uses every operation: + * ** relu exp log - /"""
    d = a * b + c ** 2
    e = (d.relu() + 1.0).log() if isinstance(d, Value) else torch.log(torch.relu(d) + 1.0)
    f = (a / 3.0 - b).exp() if isinstance(a, Value) else torch.exp(a / 3.0 - b)
    return e * f + a


def test_gradients_match_pytorch():
    a, b, c = Value(2.0), Value(-0.5), Value(1.5)
    out = expression(a, b, c)
    out.backward()

    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (2.0, -0.5, 1.5))
    tout = expression(ta, tb, tc)
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    for ours, theirs in ((a, ta), (b, tb), (c, tc)):
        assert ours.grad == pytest.approx(theirs.grad.item(), rel=1e-9)


def test_gradient_matches_finite_difference():
    """Nudge the input a tiny bit and see how much the output moves: the definition of a derivative."""
    def f(x):
        return expression(Value(x), Value(-0.5), Value(1.5)).data

    a = Value(2.0)
    expression(a, Value(-0.5), Value(1.5)).backward()
    h = 1e-6
    numeric = (f(2.0 + h) - f(2.0 - h)) / (2 * h)
    assert a.grad == pytest.approx(numeric, rel=1e-5)


def test_reused_value_accumulates_gradient():
    x = Value(3.0)
    y = x * x + x  # dy/dx = 2x + 1 = 7; x is used 3 times, so gradients must ADD up
    y.backward()
    assert x.grad == 7.0


def test_mlp_learns_a_simple_function():
    # y = 2*x1 - x2 + 0.5: a 2-4-1 network should fit it; loss must fall a lot.
    xs = [[x1 / 2, x2 / 2] for x1 in range(-2, 3) for x2 in range(-2, 3)]
    ys = [2 * x1 - x2 + 0.5 for x1, x2 in xs]
    model = MLP([2, 4, 1], seed=0)
    losses = [train_step(model, xs, ys, lr=0.05) for _ in range(300)]
    assert losses[-1] < 0.05 * losses[0]
    assert all(math.isfinite(v) for v in losses)
