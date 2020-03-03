import numpy as np
from scipy import stats
from copy import deepcopy


class NN():
    def __init__(self, M, H, N, sx=2, s0=10, s=1):
        # テキストではs=0.1にしてたけど勾配が大きくなりすぎる
        self.M = M
        self.H = H
        self.N = N
        self.sx = sx
        self.s0 = s0
        self.s = s
        self.d = (M + N) * H

    def activate(self, x):
        # 活性化関数
        return 1 / (1 + np.exp(-x))
        # return max(x, 0)

    def R(self, x, a, b):
        return np.array([
            sum([
                a[h][j] / self.activate((b[h] * x).sum())
                for h in range(self.H)
            ]) for j in range(self.N)
        ])

    def f(self, a, b, X, Y):
        XY = [(X[i], Y[i]) for i, _ in enumerate(X)]
        return ((a**2).sum() + (b**2).sum()) / (2 * self.s0**2) + sum([(
            (y - self.R(x, a, b))**2).sum() for x, y in XY]) / (2 * self.s**2)

    def df(self, a, b, X, Y):
        # Hのa,bに関する数値微分
        eps = 0.0001
        dfa = np.zeros_like(a)
        dfb = np.zeros_like(b)
        for h in range(self.H):
            for n in range(self.N):
                da = np.zeros_like(a)
                da[h][n] = eps
                dfa[h][n] = (self.f(a + da, b, X, Y) -
                             self.f(a - da, b, X, Y)) / (2 * eps)
        for h in range(self.H):
            for m in range(self.M):
                db = np.zeros_like(b)
                db[h][m] = eps
                dfb[h][n] = (self.f(a, b + db, X, Y) -
                             self.f(a, b - db, X, Y)) / (2 * eps)
        return dfa, dfb

    def p(self, x, y, a, b):
        # 確率モデル
        px = stats.multivariate_normal(np.zeros(self.M),
                                       self.sx * np.identity(self.M)).pdf(x)
        py = stats.multivariate_normal(self.R(x, a, b),
                                       self.s * np.identity(self.N)).pdf(y)
        return px * py

    def sampling(self, X, Y, size=500, burn=100, eps=0.001):
        # ランジュバンモンテカルロで事後分布からのサンプリング
        sample = []
        a = np.random.normal(0, 1, (self.H, self.N))
        b = np.random.normal(0, 1, (self.H, self.M))
        for _ in range(size):
            ga = np.random.normal(0, (2 * eps)**0.5, (self.H, self.N))
            gb = np.random.normal(0, (2 * eps)**0.5, (self.H, self.M))
            dfa, dfb = self.df(a, b, X, Y)
            a = a - eps * dfa + ga
            b = b - eps * dfb + gb
            sample.append((a, b))
        return sample[burn:]

    def generate(self, a, b):
        # パラメータを決めたときのX,Yのサンプリング
        x = np.random.multivariate_normal(np.zeros(self.M),
                                          self.sx * np.identity(self.M))
        y = np.random.multivariate_normal(self.R(x, a, b),
                                          self.s * np.identity(self.N))
        return (x, y)
