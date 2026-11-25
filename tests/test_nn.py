import numpy as np

from trader import nn


def test_gradient_check():
    """Backprop must match numerical derivatives. This proves the calculus is right."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 5))
    y = (rng.random(32) > 0.5).astype(float)
    model = nn.MLP(5, [6, 4], seed=1)
    # ReLU has a corner at exactly 0 where the derivative is undefined. With zero
    # biases some units sit right on it, so nudge biases off zero before checking.
    for layer in model.layers:
        if isinstance(layer, nn.Dense):
            layer.b[:] = rng.normal(0, 0.1, layer.b.shape)
    assert nn.gradient_check(model, X, y, l2=1e-3, samples_per_param=20) < 1e-6


def test_solves_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 50, dtype=float)
    y = np.array([0, 1, 1, 0] * 50, dtype=float)
    model = nn.MLP(2, [8], seed=3)
    nn.train(model, X, y, X, y, epochs=400, batch_size=32, lr=0.02, l2=0.0, patience=400)
    assert ((model.predict_proba(X[:4]) > 0.5) == y[:4]).all()


def test_logistic_cannot_solve_xor():
    """A single neuron draws a straight line — XOR needs a hidden layer."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 50, dtype=float)
    y = np.array([0, 1, 1, 0] * 50, dtype=float)
    model = nn.MLP(2, [], seed=0)
    nn.train(model, X, y, X, y, epochs=200, batch_size=32, lr=0.02, l2=0.0, patience=200)
    assert ((model.predict_proba(X[:4]) > 0.5) == y[:4]).mean() < 1


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    model = nn.MLP(3, [4], seed=0)
    scaler = nn.Standardizer().fit(X)
    path = tmp_path / "m.npz"
    nn.save(path, model, scaler, {"features": ["a", "b", "c"]})
    model2, scaler2, meta = nn.load(path)
    np.testing.assert_allclose(model.predict_proba(scaler.transform(X)),
                               model2.predict_proba(scaler2.transform(X)))
    assert meta["features"] == ["a", "b", "c"]


def test_standardizer_handles_constant_column():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    Z = nn.Standardizer().fit(X).transform(X)
    assert np.isfinite(Z).all()
