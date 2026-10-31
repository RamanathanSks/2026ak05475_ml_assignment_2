"""From-scratch ML models and metrics (NumPy only, no scikit-learn).

Shared by train_model.ipynb and app.py so saved .joblib files can be loaded anywhere.
"""
import numpy as np


class StandardScalerScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0

def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0

def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def mcc(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0

def rank_average(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1   # average rank for ties
        i = j + 1
    return ranks

def auc(y_true, scores):
    # Mann-Whitney U formulation of ROC-AUC
    ranks = rank_average(np.asarray(scores, dtype=float))
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=2000, C=1.0):
        self.lr, self.n_iter, self.C = lr, n_iter, C

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        lam = 1.0 / (self.C * n)
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y) / n + lam * self.w)
            self.b -= self.lr * np.mean(p - y)
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 2 * p * (1 - p)

    def _best_split(self, X, y):
        n, d = X.shape
        best = (None, None, self._gini(y))
        for j in range(d):
            order = np.argsort(X[:, j])
            xs, ys = X[order, j], y[order]
            cum_pos = np.cumsum(ys)
            total_pos = cum_pos[-1]
            for i in range(1, n):
                if xs[i] == xs[i - 1]:
                    continue
                nl, nr = i, n - i
                pl = cum_pos[i - 1] / nl
                pr = (total_pos - cum_pos[i - 1]) / nr
                g = (nl * 2 * pl * (1 - pl) + nr * 2 * pr * (1 - pr)) / n
                if g < best[2] - 1e-12:
                    best = (j, (xs[i] + xs[i - 1]) / 2.0, g)
        return best[0], best[1]

    def _build(self, X, y, depth):
        prob = np.mean(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or prob in (0.0, 1.0):
            return {"leaf": True, "prob": prob}
        j, thr = self._best_split(X, y)
        if j is None:
            return {"leaf": True, "prob": prob}
        mask = X[:, j] <= thr
        return {"leaf": False, "feat": j, "thr": thr, "prob": prob,
                "left": self._build(X[mask], y[mask], depth + 1),
                "right": self._build(X[~mask], y[~mask], depth + 1)}

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x):
        node = self.tree
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["prob"]

    def predict_proba(self, X):
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class KNNScratch:
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, X):
        # squared Euclidean distance matrix
        d2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(self.X ** 2, axis=1)[None, :]
              - 2 * X @ self.X.T)
        idx = np.argsort(d2, axis=1)[:, :self.k]
        return self.y[idx].mean(axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.array([0, 1])
        eps = self.var_smoothing * X.var(axis=0).max()
        self.mu = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes]) + eps
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        return self

    def _joint_log_likelihood(self, X):
        out = []
        for k in range(len(self.classes)):
            ll = -0.5 * np.sum(np.log(2 * np.pi * self.var[k]) + (X - self.mu[k]) ** 2 / self.var[k], axis=1)
            out.append(self.log_prior[k] + ll)
        return np.array(out).T

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return (p / p.sum(axis=1, keepdims=True))[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class SVMScratch:
    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_passes=10, max_iter=200):
        self.C, self.gamma, self.tol = C, gamma, tol
        self.max_passes, self.max_iter = max_passes, max_iter

    def _kernel(self, A, B):
        d2 = (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T)
        return np.exp(-self.gamma_ * np.maximum(d2, 0))

    def fit(self, X, y):
        n, d = X.shape
        self.gamma_ = 1.0 / (d * X.var()) if self.gamma == "scale" else self.gamma
        ys = np.where(y == 1, 1.0, -1.0)
        K = self._kernel(X, X)
        alpha = np.zeros(n)
        b = 0.0
        rng = np.random.RandomState(42)
        passes, it = 0, 0
        while passes < self.max_passes and it < self.max_iter:
            changed = 0
            for i in range(n):
                Ei = (alpha * ys) @ K[:, i] + b - ys[i]
                if (ys[i] * Ei < -self.tol and alpha[i] < self.C) or (ys[i] * Ei > self.tol and alpha[i] > 0):
                    j = rng.randint(n - 1)
                    j = j + 1 if j >= i else j
                    Ej = (alpha * ys) @ K[:, j] + b - ys[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if ys[i] != ys[j]:
                        L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] = np.clip(aj_old - ys[j] * (Ei - Ej) / eta, L, H)
                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue
                    alpha[i] = ai_old + ys[i] * ys[j] * (aj_old - alpha[j])
                    b1 = b - Ei - ys[i] * (alpha[i] - ai_old) * K[i, i] - ys[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - ys[i] * (alpha[i] - ai_old) * K[i, j] - ys[j] * (alpha[j] - aj_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            it += 1
        sv = alpha > 1e-8
        self.sv_X, self.sv_coef, self.b = X[sv], (alpha * ys)[sv], b
        # Platt scaling on training decision values
        f = self.decision_function(X)
        a, c = 1.0, 0.0
        for _ in range(3000):
            p = sigmoid(a * f + c)
            a -= 0.05 * np.mean((p - y) * f)
            c -= 0.05 * np.mean(p - y)
        self.platt = (a, c)
        return self

    def decision_function(self, X):
        return self._kernel(X, self.sv_X) @ self.sv_coef + self.b

    def predict_proba(self, X):
        a, c = self.platt
        return sigmoid(a * self.decision_function(X) + c)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class PipelineScratch:
    """Scaler + model bundled together, so saved files accept raw (unscaled) features."""

    def __init__(self, scaler, model):
        self.scaler, self.model = scaler, model

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        """Probability of class 1 (Benign), one value per row."""
        return self.model.predict_proba(self.scaler.transform(X))


def classification_report_df(y_true, y_pred, labels=("Malignant (0)", "Benign (1)")):
    import pandas as pd
    p0, r0, f0 = precision(1 - y_true, 1 - y_pred), recall(1 - y_true, 1 - y_pred), f1(1 - y_true, 1 - y_pred)
    p1, r1, f1_ = precision(y_true, y_pred), recall(y_true, y_pred), f1(y_true, y_pred)
    n0, n1 = int(np.sum(y_true == 0)), int(np.sum(y_true == 1))
    w = lambda a, b: (a * n0 + b * n1) / (n0 + n1)
    return pd.DataFrame({
        "precision": [p0, p1, (p0 + p1) / 2, w(p0, p1)],
        "recall": [r0, r1, (r0 + r1) / 2, w(r0, r1)],
        "f1-score": [f0, f1_, (f0 + f1_) / 2, w(f0, f1_)],
        "support": [n0, n1, n0 + n1, n0 + n1],
    }, index=list(labels) + ["macro avg", "weighted avg"])


def make_models():
    """The five classifiers (each wrapped with its own StandardScaler)."""
    return {
        "Logistic Regression": PipelineScratch(StandardScalerScratch(), LogisticRegressionScratch()),
        "Decision Tree": PipelineScratch(StandardScalerScratch(), DecisionTreeScratch(max_depth=5)),
        "KNN": PipelineScratch(StandardScalerScratch(), KNNScratch(k=7)),
        "Naive Bayes": PipelineScratch(StandardScalerScratch(), GaussianNBScratch()),
        "Support Vector Machine": PipelineScratch(StandardScalerScratch(), SVMScratch(C=1.0)),
    }


MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}
