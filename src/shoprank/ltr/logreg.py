"""Logistic regression from scratch with NumPy (gradient descent + L2).

Why build it: it is the simplest learned model, and the same 4 steps (predict → loss → gradient →
update) are how every neural network in Weeks 3–7 is trained.

Model:    p = sigmoid(X @ w + b)                      probability the product is relevant
Loss:     mean(-y*log(p) - (1-y)*log(1-p)) + (l2/2)*|w|^2   (log loss + penalty on big weights)
Gradient: dL/dw = X.T @ (p - y) / n + l2*w ;  dL/db = mean(p - y)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip avoids overflow in exp() for very large |z|.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass(frozen=True)
class Standardizer:
    """Rescale each feature to mean 0, std 1 using TRAIN statistics only (reused unchanged on dev/test)."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> Standardizer:
        std = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(std > 0, std, 1.0))  # constant column -> leave as is

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


@dataclass(frozen=True)
class LogisticRegression:
    weights: np.ndarray
    bias: float
    losses: tuple[float, ...]  # loss after each epoch, to plot the learning curve

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 2000,
            l2: float = 0.0) -> LogisticRegression:
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        losses = []
        for _ in range(epochs):
            p = sigmoid(X @ w + b)                          # 1. predict
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + l2 / 2 * w @ w
            losses.append(float(loss))                      # 2. measure how wrong
            grad_w = X.T @ (p - y) / n + l2 * w             # 3. which direction reduces loss
            grad_b = float(np.mean(p - y))
            w = w - lr * grad_w                             # 4. step downhill
            b = b - lr * grad_b
        return cls(w, b, tuple(losses))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)
