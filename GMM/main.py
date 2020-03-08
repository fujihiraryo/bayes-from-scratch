import numpy as np
import pandas as pd

import model
import criteria

N = 2  # データの次元
iter = 10  # ハイパラの更新回数
T = 100  # 事後分布からのサンプリング数
n = 100  # データ数
K0 = 2  # 真のコンポーネント数
ex = 10  # 実験回数

# 真のモデル
a0 = np.array([.5, .5])
b0 = np.array([[-2, -2], [2, 2]])
true = model.GMM(N, K0)

result = {'G': [], 'AIC': [], 'WAIC': [], 'CV': []}
for t in range(ex):
    # 真の分布からのサンプリング
    X = [true.data_sampling(a0, b0) for i in range(n)]
    X_t, X_v = X[:n], X[n:]

    # 変分ベイズによる学習
    for K in range(1, 10):
        gmm = model.GMM(K, N)
        for _ in range(iter):
            gmm.update(X_t)

    # 各種規準の計算
    params = [gmm.poserior_sampling() for _ in range(T)]
    crt = criteria.Criteria(gmm, true, X_t, X_v, params, (a0, b0))
    L = crt.L
    Ln = crt.Ln
    G = crt.G
    AIC = crt.AIC
    WAIC = crt.WAIC
    CV = crt.CV
    result['G'].append(G - L)
    result['AIC'].append(AIC - Ln)
    result['WAIC'].append(WAIC - Ln)
    result['CV'].append(CV - Ln)

# 結果をcsvに保存
pd.DataFrame(result).to_csv('GMM/result.csv', index=False)
