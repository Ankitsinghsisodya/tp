# Roll: <roll_number>
# Name: <name>
# Set: 1
# Question: 1

import numpy as np
import urllib.request


def fetch_wine():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    rows = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            row = list(map(float, line.decode().strip().split(',')))
            rows.append(row)
    data = np.array(rows)
    return data[:, 1:], (data[:, 0].astype(int) - 1)


def split(X, y, ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n = int(len(y) * ratio)
    return X[idx[n:]], X[idx[:n]], y[idx[n:]], y[idx[:n]]


def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - m) / s, (Xte - m) / s


class LDA:
    # generative model - assumes shared covariance across classes
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        self.means_ = np.array([X[y == c].mean(0) for c in self.classes_])
        Sw = np.zeros((d, d))
        for c, mu in zip(self.classes_, self.means_):
            Xc = X[y == c] - mu
            Sw += Xc.T @ Xc
        self.Sw_inv_ = np.linalg.pinv(Sw / (n - len(self.classes_)))

    def scores(self, X):
        s = []
        for i in range(len(self.classes_)):
            diff = X - self.means_[i]
            score = -0.5 * (diff @ self.Sw_inv_ * diff).sum(1) + np.log(self.priors_[i])
            s.append(score)
        return np.array(s).T

    def predict(self, X):
        return self.classes_[self.scores(X).argmax(1)]


class QDA:
    # generative model - each class gets its own covariance matrix
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = [(y == c).mean() for c in self.classes_]
        self.params_ = []
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(0)
            cov = (Xc - mu).T @ (Xc - mu) / (len(Xc) - 1)
            cov_inv = np.linalg.pinv(cov)
            _, ld = np.linalg.slogdet(cov)
            self.params_.append((mu, cov_inv, ld))

    def predict(self, X):
        scores = []
        for i, (mu, cov_inv, ld) in enumerate(self.params_):
            diff = X - mu
            s = -0.5 * (diff @ cov_inv * diff).sum(1) - 0.5 * ld + np.log(self.priors_[i])
            scores.append(s)
        return self.classes_[np.array(scores).T.argmax(1)]


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class LogisticSGD:
    # discriminative model - one-vs-rest with L2 regularization
    def __init__(self, lr=0.05, epochs=80, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam

    def fit(self, X, y):
        n, d = X.shape
        self.classes_ = np.unique(y)
        self.W_ = np.zeros((len(self.classes_), d))
        self.b_ = np.zeros(len(self.classes_))
        rng = np.random.default_rng(1)
        for k, c in enumerate(self.classes_):
            yb = (y == c).astype(float)
            w, b = np.zeros(d), 0.0
            for _ in range(self.epochs):
                for i in rng.permutation(n):
                    p = sigmoid(w @ X[i] + b)
                    g = p - yb[i]
                    w -= self.lr * (g * X[i] + self.lam * w)
                    b -= self.lr * g
            self.W_[k] = w
            self.b_[k] = b

    def predict(self, X):
        probs = sigmoid(X @ self.W_.T + self.b_)
        return self.classes_[probs.argmax(1)]


def acc(yt, yp):
    return (yt == yp).mean()


def main():
    print("Loading Wine dataset...")
    X, y = fetch_wine()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr_n, Xte_n = standardize(Xtr, Xte)

    lda = LDA()
    lda.fit(Xtr, ytr)
    print(f"LDA accuracy : {acc(yte, lda.predict(Xte)):.4f}  (generative, shared covariance)")

    qda = QDA()
    qda.fit(Xtr, ytr)
    print(f"QDA accuracy : {acc(yte, qda.predict(Xte)):.4f}  (generative, per-class covariance)")

    lgr = LogisticSGD(lr=0.05, epochs=80, lam=0.01)
    lgr.fit(Xtr_n, ytr)
    print(f"LR  accuracy : {acc(yte, lgr.predict(Xte_n)):.4f}  (discriminative, L2 + SGD)")

    print()
    print("LDA and QDA are generative - they model P(x|y) and use Bayes rule for P(y|x).")
    print("Logistic regression is discriminative - it learns P(y|x) directly.")
    print("Bayes risk = E[loss under optimal decision] = 1 - max_c P(y=c|x)")


if __name__ == "__main__":
    main()
