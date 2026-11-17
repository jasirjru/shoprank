"""A tiny neural network built on our scratch autograd. Week 3, Part B.

Neuron:  out = relu(w1*x1 + w2*x2 + ... + b)     (a weighted vote, then "only fire if positive")
Layer:   many neurons looking at the same inputs
MLP:     layers stacked: input -> hidden -> ... -> output   ("multi-layer perceptron")
"""
from __future__ import annotations

import random

from shoprank.scratch.autograd import Value


class Neuron:
    def __init__(self, n_inputs: int, rng: random.Random, linear: bool = False) -> None:
        scale = (2.0 / n_inputs) ** 0.5  # He initialization: keeps signal size stable through ReLU layers
        self.w = [Value(rng.gauss(0.0, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.linear = linear  # the last layer has no ReLU so it can output any number

    def __call__(self, x: list[Value | float]) -> Value:
        total = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return total if self.linear else total.relu()

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        self.layers = [
            [Neuron(sizes[i], rng, linear=(i == len(sizes) - 2)) for _ in range(sizes[i + 1])]
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: list[float]) -> list[Value]:
        out: list[Value | float] = x
        for layer in self.layers:
            out = [neuron(out) for neuron in layer]
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for neuron in layer for p in neuron.parameters()]


def train_step(model: MLP, xs: list[list[float]], ys: list[float], lr: float) -> float:
    """One full training step on mean squared error: predict, loss, backward, update."""
    preds = [model(x)[0] for x in xs]                                     # 1. predict
    loss = sum(((p - y) ** 2 for p, y in zip(preds, ys, strict=True)), Value(0.0)) / len(ys)  # 2. loss
    for p in model.parameters():
        p.grad = 0.0                                                      # forget old gradients
    loss.backward()                                                       # 3. gradients
    for p in model.parameters():
        p.data -= lr * p.grad                                             # 4. step downhill
    return loss.data
