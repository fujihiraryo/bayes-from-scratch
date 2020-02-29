import numpy as np
import matplotlib.pyplot as plt
import model

# 次元設定
N = 2

# 真の分布からのサンプリング
n = 1000
K0 = 2
a0 = np.array([.5, .5])
b0 = np.array([[-2, -2], [2, 2]])
X = [
    np.random.multivariate_normal(b0[np.random.choice(K0, p=a0)],
                                  np.identity(N)) for i in range(2 * n)
]
X_t, X_v = X[:n], X[n:]

# 変分ベイズによる学習
iter = 10
for K in range(1, 10):
    gmm = model.GMM(K, N)
    for _ in range(10):
        gmm.update(X_t)
    T = np.mean([-np.log(gmm.predict(x)) for x in X_t])
    G = np.mean([-np.log(gmm.predict(x)) for x in X_v])
    print(round(T, 3), round(G, 3))
