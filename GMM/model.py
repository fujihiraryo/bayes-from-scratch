import numpy as np
from scipy.special import psi
from scipy import stats


class GMM():
    def __init__(self, K, N):
        self.K = K
        self.N = N
        self.phi = np.ones(K)
        self.mu = np.random.normal(0, 0.1, (K, N))
        self.tau = np.ones(K)
        self.d = (K - 1) + K * N

    def gauss(self, x, mu, tau):
        return stats.multivariate_normal(mu, (tau)**(-0.5)).pdf(x)

    def p(self, x, a, b):
        return sum([
            a[k] * self.gauss(x, b[k], np.identity(self.N))
            for k in range(self.K)
        ])

    def update(self, X):
        Y = np.zeros(self.K)
        XY = np.zeros((self.K, self.N))
        for x in X:
            L = np.array([
                psi(self.phi[k]) - self.tau[k]**(-1) -
                ((x - self.mu[k])**2).sum() / 2 for k in range(self.K)
            ])
            y = np.exp(L) / np.exp(L).sum()
            Y += y
            XY += np.array([x * y[k] for k in range(self.K)])
        self.phi = self.phi + Y
        self.mu = np.array([
            (self.tau[k] * self.mu[k] + XY[k]) / (self.tau[k] + Y[k])
            for k in range(self.K)
        ])
        self.tau = self.tau + Y

    def predict(self, x):
        return np.sum([
            self.phi[k] * self.gauss(x, self.mu[k], self.tau[k] /
                                     (1 + self.tau[k])) for k in range(self.K)
        ]) / np.sum([self.phi[k] for k in range(self.K)])

    def posterior_sampling(self):
        a = np.random.dirichlet(self.phi)
        b = np.array([
            np.random.multivariate_normal(self.mu[k],
                                          np.identity(self.N) / self.tau[k])
            for k in range(self.K)
        ])
        return a, b

    def data_sampling(self, a, b):
        k = np.random.choice(self.K, p=a)
        return np.random.multivariate_normal(b[k], np.identity(self.N))
