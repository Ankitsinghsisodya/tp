# Roll: <roll_number>
# Name: <name>
# Set: 1
# Question: 2

import numpy as np
import urllib.request


def fetch_bc():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    X, y = [], []
    with urllib.request.urlopen(url) as f:
        for line in f:
            row = line.decode().strip().split(',')
            y.append(1 if row[1] == 'M' else 0)
            X.append(list(map(float, row[2:])))
    return np.array(X), np.array(y)


def split(X, y, ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n = int(len(y) * ratio)
    return X[idx[n:]], X[idx[:n]], y[idx[n:]], y[idx[:n]]


def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - m) / s, (Xte - m) / s


def pca_transform(Xtr, Xte, k=10):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    cov = Xc.T @ Xc / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, np.argsort(vals)[::-1][:k]]
    return (Xtr - mu) @ W, (Xte - mu) @ W


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class GaussianNB:
    def fit(self, X, y):
        self.cls_ = np.unique(y)
        self.pr_ = {c: (y == c).mean() for c in self.cls_}
        self.mu_ = {c: X[y == c].mean(0) for c in self.cls_}
        self.va_ = {c: X[y == c].var(0) + 1e-9 for c in self.cls_}

    def _log_prob(self, X, c):
        return (np.log(self.pr_[c])
                - 0.5 * np.sum(np.log(2 * np.pi * self.va_[c])
                               + (X - self.mu_[c]) ** 2 / self.va_[c], axis=1))

    def predict(self, X):
        lps = np.column_stack([self._log_prob(X, c) for c in self.cls_])
        return self.cls_[lps.argmax(1)]

    def score1(self, X):
        return sigmoid(self._log_prob(X, 1) - self._log_prob(X, 0))


class LDA:
    def fit(self, X, y):
        n0, n1 = (y == 0).sum(), (y == 1).sum()
        mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
        Sw = ((X[y == 0] - mu0).T @ (X[y == 0] - mu0)
              + (X[y == 1] - mu1).T @ (X[y == 1] - mu1))
        Sw /= len(y) - 2
        self.w_ = np.linalg.pinv(Sw) @ (mu1 - mu0)
        self.b_ = -0.5 * (mu0 + mu1) @ self.w_ + np.log(n1 / n0)

    def decision(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)


class LogisticSGD:
    def __init__(self, lr=0.1, epochs=100, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam

    def fit(self, X, y):
        n, d = X.shape
        self.w_, self.b_ = np.zeros(d), 0.0
        rng = np.random.default_rng(42)
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                p = sigmoid(self.w_ @ X[i] + self.b_)
                g = p - y[i]
                self.w_ -= self.lr * (g * X[i] + self.lam * self.w_)
                self.b_ -= self.lr * g

    def prob(self, X):
        return sigmoid(X @ self.w_ + self.b_)

    def predict(self, X):
        return (self.prob(X) >= 0.5).astype(int)


class LinearSVM:
    # hinge loss minimization via subgradient SGD
    def __init__(self, C=1.0, epochs=50, lr=0.01):
        self.C = C
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        n, d = X.shape
        self.w_, self.b_ = np.zeros(d), 0.0
        ys = 2 * y - 1
        rng = np.random.default_rng(7)
        for ep in range(self.epochs):
            lr_t = self.lr / (1 + 0.01 * ep)
            for i in rng.permutation(n):
                if ys[i] * (self.w_ @ X[i] + self.b_) < 1:
                    self.w_ = (1 - lr_t) * self.w_ + lr_t * self.C * ys[i] * X[i]
                    self.b_ += lr_t * self.C * ys[i]
                else:
                    self.w_ *= (1 - lr_t)

    def decision(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)


class KernelSVM:
    # RBF kernel SVM approximated via random Fourier features
    def __init__(self, gamma=0.1, D=300, C=1.0, epochs=50, lr=0.01):
        self.gamma = gamma
        self.D = D
        self.C = C
        self.epochs = epochs
        self.lr = lr

    def _rff(self, X):
        return np.sqrt(2 / self.D) * np.cos(X @ self.omega_.T + self.phi_)

    def fit(self, X, y):
        rng = np.random.default_rng(3)
        self.omega_ = rng.normal(0, np.sqrt(self.gamma), (self.D, X.shape[1]))
        self.phi_ = rng.uniform(0, 2 * np.pi, self.D)
        self._svm = LinearSVM(C=self.C, epochs=self.epochs, lr=self.lr)
        self._svm.fit(self._rff(X), y)

    def decision(self, X):
        return self._svm.decision(self._rff(X))

    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)


def roc_auc(yt, scores):
    # wilcoxon-mann-whitney form: P(score_pos > score_neg)
    order = np.argsort(scores)[::-1]
    yt = yt[order]
    P, N = yt.sum(), (1 - yt).sum()
    if P == 0 or N == 0:
        return 0.0
    tp, auc = 0, 0.0
    for label in yt:
        if label:
            tp += 1
        else:
            auc += tp
    return auc / (P * N)


def acc(yt, yp):
    return (yt == yp).mean()


def report(name, yt, yp, scores):
    print(f"  {name:<14} acc={acc(yt, yp):.4f}  auc={roc_auc(yt, scores):.4f}")


def main():
    print("Loading Breast Cancer Wisconsin dataset...")
    X, y = fetch_bc()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr_n, Xte_n = standardize(Xtr, Xte)
    Xtr_p, Xte_p = pca_transform(Xtr_n, Xte_n, k=10)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features, "
          f"{y.sum()} malignant / {(y==0).sum()} benign")

    print("\n--- Full features (30 dims) ---")
    nb = GaussianNB(); nb.fit(Xtr_n, ytr)
    report("Naive Bayes",   yte, nb.predict(Xte_n),    nb.score1(Xte_n))

    lda = LDA(); lda.fit(Xtr_n, ytr)
    report("LDA",           yte, lda.predict(Xte_n),   sigmoid(lda.decision(Xte_n)))

    lr = LogisticSGD(); lr.fit(Xtr_n, ytr)
    report("Logistic Reg",  yte, lr.predict(Xte_n),    lr.prob(Xte_n))

    svm = LinearSVM(); svm.fit(Xtr_n, ytr)
    report("Linear SVM",    yte, svm.predict(Xte_n),   svm.decision(Xte_n))

    ksvm = KernelSVM(gamma=0.5, D=300); ksvm.fit(Xtr_n, ytr)
    report("RBF SVM",       yte, ksvm.predict(Xte_n),  ksvm.decision(Xte_n))

    print("\n--- After PCA (10 components) ---")
    nb2 = GaussianNB(); nb2.fit(Xtr_p, ytr)
    report("Naive Bayes",   yte, nb2.predict(Xte_p),   nb2.score1(Xte_p))

    lda2 = LDA(); lda2.fit(Xtr_p, ytr)
    report("LDA",           yte, lda2.predict(Xte_p),  sigmoid(lda2.decision(Xte_p)))

    lr2 = LogisticSGD(); lr2.fit(Xtr_p, ytr)
    report("Logistic Reg",  yte, lr2.predict(Xte_p),   lr2.prob(Xte_p))

    svm2 = LinearSVM(); svm2.fit(Xtr_p, ytr)
    report("Linear SVM",    yte, svm2.predict(Xte_p),  svm2.decision(Xte_p))

    ksvm2 = KernelSVM(gamma=0.5, D=300); ksvm2.fit(Xtr_p, ytr)
    report("RBF SVM",       yte, ksvm2.predict(Xte_p), ksvm2.decision(Xte_p))


if __name__ == "__main__":
    main()
