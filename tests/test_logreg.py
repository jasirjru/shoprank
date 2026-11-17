import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLogReg

from shoprank.ltr.logreg import LogisticRegression, Standardizer, sigmoid


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = (rng.uniform(size=500) < sigmoid(X @ true_w + 0.3)).astype(float)
    return X, y


def test_matches_sklearn(data):
    X, y = data
    l2 = 0.01
    ours = LogisticRegression.fit(X, y, lr=0.5, epochs=5000, l2=l2)
    # sklearn minimizes C*sum(logloss) + 0.5*|w|^2  ==  mean(logloss) + (l2/2)*|w|^2 when C = 1/(l2*n)
    ref = SkLogReg(C=1 / (l2 * len(y)), tol=1e-10, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.weights, ref.coef_[0], atol=1e-4)
    assert ours.bias == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_loss_goes_down(data):
    X, y = data
    model = LogisticRegression.fit(X, y, lr=0.5, epochs=200)
    assert model.losses[-1] < model.losses[0]
    assert all(b <= a + 1e-12 for a, b in zip(model.losses, model.losses[1:], strict=False))


def test_standardizer_uses_train_statistics_only():
    train = np.array([[1.0, 10.0], [3.0, 10.0]])
    s = Standardizer.fit(train)
    np.testing.assert_allclose(s.transform(train), [[-1.0, 0.0], [1.0, 0.0]])  # constant column stays 0
    np.testing.assert_allclose(s.transform(np.array([[5.0, 10.0]])), [[3.0, 0.0]])  # dev reuses train mean/std


def test_sigmoid_is_stable_for_extreme_inputs():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out)) and out[1] == 0.5
