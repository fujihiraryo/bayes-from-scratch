import numpy as np
from scipy import stats


class NN():
    def __init__(self, M, H, N, sx=2, s0=10, s=0.1):
        self.M = M
        self.H = H
        self.N = N
        self.sx = sx
        self.s0 = s0
        self.s = s
        self.d = (M + N) * H

    def R(self, x, a, b):
        return np.array([
            sum([
                a[h][j] / (1 + np.exp(-(b[h] * x).sum()))
                for h in range(self.H)
            ]) for j in range(self.N)
        ])

    def p(self, x, y, a, b):
        return stats.multivariate_normal(
            np.zeros(self.M),
            self.sx * np.identity(self.M)).pdf(x) * stats.multivariate_normal(
                self.R(x, a, b), self.s * np.identity(self.N)).pdf(y)

    def sampling(self, X, Y, size=100, burn=10):
        XY = [(X[i], Y[i]) for i, _ in enumerate(X)]
        sample = []
        accept, reject = 0, 0
        a0 = np.random.normal(0, 1, (self.H, self.N))
        b0 = np.random.normal(0, 1, (self.H, self.M))
        while accept < size:
            a1 = np.array(
                [[np.random.normal(a0[h][n], 0.05) for n in range(self.N)]
                 for h in range(self.H)])
            b1 = np.array(
                [[np.random.normal(b0[h][m], 0.05) for m in range(self.M)]
                 for h in range(self.H)])
            u = np.random.random()
            P = np.exp(((a0**2 - a1**2).sum() +
                        (b0**2 - b1**2).sum()) / (2 * self.s0**2) +
                       sum([((y - self.R(x, a0, b0))**2 -
                             (y - self.R(x, a1, b1))**2).sum()
                            for x, y in XY]) / (2 * self.s**2))
            P = min(1, P)
            if u < P:
                a0, b0 = a1, b1
                accept += 1
                sample.append((a0, b0))
            else:
                reject += 1
                if reject == 10**6:
                    print(f'accept={accept}, reject={reject}')
                    raise Exception('MCMC無理やったわすまん')
        return sample[burn:]

    def generate(self, a, b):
        x = np.random.multivariate_normal(np.zeros(self.M),
                                          self.sx * np.identity(self.M))
        y = np.random.multivariate_normal(self.R(x, a, b),
                                          self.s * np.identity(self.N))
        return (x, y)


if __name__ == '__main__':
    M, H, N = 3, 4, 2
    a = np.random.normal(0, 1, (H, N))
    b = np.random.normal(0, 1, (H, M))
    x = np.zeros(M)
    nn = NN(M, H, N)
    print(nn.R(x, a, b))
    print(nn.p(x, nn.R(x, a, b), a, b))
    sample = nn.sampling([], [])
    a, b = sample[0]
    print(a.shape, b.shape)
    print(a, b)
