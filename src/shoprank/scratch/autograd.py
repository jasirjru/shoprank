"""A tiny autograd engine (after Karpathy's micrograd). Week 3, Part A.

Idea: every number remembers HOW it was made (which inputs, which operation). Calling .backward()
on the final result walks that recipe backwards and applies the chain rule, so every input learns
"if I nudge you up a little, how much does the result change?" — its gradient.

PyTorch's autograd does exactly this, for tensors instead of single numbers, on the GPU.
"""
from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    def __init__(self, data: float, parents: tuple[Value, ...] = (), op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0                                  # d(final result) / d(this value)
        self._parents = parents                          # values this one was computed from
        self._op = op                                    # for debugging: "+", "*", "relu", ...
        self._backward: Callable[[], None] = lambda: None  # pushes self.grad to parents

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- operations: each defines the forward result AND its local derivative rule ---

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            # d(a+b)/da = 1 and d(a+b)/db = 1: the gradient passes through unchanged.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            # d(a*b)/da = b and d(a*b)/db = a.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __pow__(self, exponent: float) -> Value:
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad  # slope 1 if active, else 0
        out._backward = backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def backward() -> None:
            self.grad += out.data * out.grad  # d(e^x)/dx = e^x
        out._backward = backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += (1.0 / self.data) * out.grad
        out._backward = backward
        return out

    # Convenience forms built from the operations above (so they need no backward rule of their own).
    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other ** -1.0 if isinstance(other, Value) else 1.0 / other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    # --- the chain rule over the whole recipe ---

    def backward(self) -> None:
        """Compute gradients of THIS value with respect to every value it depends on."""
        order: list[Value] = []
        seen: set[int] = set()

        def visit(v: Value) -> None:  # topological order: every value after the values it needs
            if id(v) not in seen:
                seen.add(id(v))
                for parent in v._parents:
                    visit(parent)
                order.append(v)
        visit(self)

        self.grad = 1.0  # d(result)/d(result) = 1
        for v in reversed(order):  # from the result back to the inputs
            v._backward()
