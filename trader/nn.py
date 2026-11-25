"""A neural network written from scratch with NumPy.

No PyTorch — every forward pass, gradient and weight update is here, so you can
read exactly how backpropagation works.

    input -> Dense -> ReLU -> Dense -> ReLU -> Dense -> (logit) -> sigmoid = P(up)
"""
import json

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class Dense:
    """y = x @ W + b"""

    def __init__(self, n_in, n_out, rng):
        # He initialization keeps signals a sensible size through ReLU layers
        self.W = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.x = x                     # remember the input; backward needs it
        return x @ self.W + self.b

    def backward(self, grad_out):
        # Chain rule: how much did each weight contribute to the loss?
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T     # pass the gradient to the previous layer

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask    # slope is 1 where x > 0, else 0

    def params(self):
        return []

    def grads(self):
        return []


def bce_with_logits(z, y):
    """Binary cross-entropy computed from the raw logit z (numerically stable).
    Returns (loss, dLoss/dz)."""
    loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    grad = (sigmoid(z) - y) / len(y)
    return loss, grad


class MLP:
    """Multi-layer perceptron for binary classification.
    hidden=[] gives logistic regression — your original one-neuron model."""

    def __init__(self, n_in, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = [n_in, *hidden, 1]
        self.layers = []
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            self.layers.append(Dense(a, b, rng))
            if i < len(self.sizes) - 2:
                self.layers.append(ReLU())

    def logits(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out[:, 0]

    def predict_proba(self, X):
        return sigmoid(self.logits(X))

    def backward(self, grad_logits):
        grad = grad_logits[:, None]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]

    def loss_and_grads(self, X, y, l2=0.0):
        z = self.logits(X)
        loss, dz = bce_with_logits(z, y)
        self.backward(dz)
        weights = [layer.W for layer in self.layers if isinstance(layer, Dense)]
        loss += 0.5 * l2 * sum((W ** 2).sum() for W in weights)
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.dW += l2 * layer.W   # L2 penalty discourages huge weights (overfitting)
        return loss

    def get_weights(self):
        return [p.copy() for p in self.params()]

    def set_weights(self, weights):
        for p, w in zip(self.params(), weights):
            p[...] = w


class Adam:
    """Gradient descent with momentum and per-weight step sizes."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(model, X, y, X_val, y_val, epochs=30, batch_size=256, lr=1e-3, l2=1e-4,
          patience=5, seed=0, verbose=False):
    """Mini-batch training with early stopping on validation loss.
    Keeps the weights from the best validation epoch."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    best_loss, best_weights, bad_epochs = np.inf, model.get_weights(), 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(X))   # shuffling WITHIN the training set is fine
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            model.loss_and_grads(X[idx], y[idx], l2)
            opt.step(model.grads())
        train_loss = bce_with_logits(model.logits(X), y)[0]
        val_loss = bce_with_logits(model.logits(X_val), y_val)[0]
        history.append((train_loss, val_loss))
        if verbose:
            print(f"  epoch {epoch:2d}  train {train_loss:.5f}  val {val_loss:.5f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_weights, bad_epochs = val_loss, model.get_weights(), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.set_weights(best_weights)
    return history


def gradient_check(model, X, y, l2=0.0, eps=1e-5, samples_per_param=5, seed=0):
    """Compare backprop gradients with numerical ones: (L(w+e) - L(w-e)) / 2e.
    Returns the worst relative error. Below ~1e-6 means backprop is correct."""
    rng = np.random.default_rng(seed)
    model.loss_and_grads(X, y, l2)
    analytic = [g.copy() for g in model.grads()]
    worst = 0.0
    for p, g in zip(model.params(), analytic):
        for _ in range(samples_per_param):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + eps
            plus = model.loss_and_grads(X, y, l2)
            p[i] = old - eps
            minus = model.loss_and_grads(X, y, l2)
            p[i] = old
            numeric = (plus - minus) / (2 * eps)
            denom = max(abs(numeric) + abs(g[i]), 1e-7)  # ignore float noise when both are ~0
            worst = max(worst, abs(numeric - g[i]) / denom)
    return worst


class Standardizer:
    """Scale features using TRAINING statistics only (no peeking at the future)."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def save(path, model, scaler, meta):
    arrays = {f"p{i}": p for i, p in enumerate(model.params())}
    np.savez(path, mean=scaler.mean, std=scaler.std, sizes=np.array(model.sizes),
             meta=json.dumps(meta), **arrays)


def load(path):
    data = np.load(path, allow_pickle=False)
    sizes = data["sizes"].tolist()
    model = MLP(sizes[0], sizes[1:-1])
    model.set_weights([data[f"p{i}"] for i in range(len(model.params()))])
    scaler = Standardizer()
    scaler.mean, scaler.std = data["mean"], data["std"]
    return model, scaler, json.loads(str(data["meta"]))
