import numpy as np
import pandas as pd

import model
import criteria

M = 3  # 入力次元
N = 3  # 出力次元
H0 = 1  # 真の隠れユニット数
H = 3  # モデルの隠れユニット数
n = 200  # サンプル数
T = 100  # 実験の回数
size = 100  # MCMCの遷移回数
burn = 20  # サンプルの最初何個捨てるか

# a0 = np.random.normal(0, 1, (H0, N))
# b0 = np.random.normal(0, 1, (H0, M))
a0 = np.ones((H0, N))
b0 = np.ones((H0, M))

# 学習とモデルの評価
result = {'G': [], 'AIC': [], 'WAIC': [], 'DIC1': [], 'DIC2': [], 'CV': []}
for t in range(T):
    # データ生成
    nn0 = model.NN(M, H0, N)
    XY = [nn0.generate(a0, b0) for _ in range(2 * n)]
    XYt, XYv = XY[:n], XY[n:]
    Xt = [xy[0] for xy in XYt]
    Xv = [xy[0] for xy in XYv]
    Yt = [xy[1] for xy in XYt]
    Yv = [xy[1] for xy in XYv]

    # MCMC
    nn = model.NN(M, H, N)
    params = nn.sampling(Xt, Yt, size=size, burn=burn)

    # 各種規準の計算
    crt = criteria.Criteria(nn, nn0, XYt, XYv, params, (a0, b0))
    L = crt.L
    Ln = crt.Ln
    G = crt.G
    AIC = crt.AIC
    WAIC = crt.WAIC
    DIC1 = crt.DIC1
    DIC2 = crt.DIC2
    CV = crt.CV
    print(
        f'loss={G-L:.3g}, AIC={AIC-Ln:.3g},WAIC={WAIC-Ln:.3g}, DIC1={DIC1-Ln:.3g}, DIC2={DIC2-Ln:.3g}, CV={CV-Ln:.3g}'
    )
    result['G'].append(G - L)
    result['AIC'].append(AIC - Ln)
    result['WAIC'].append(WAIC - Ln)
    result['DIC1'].append(DIC1 - Ln)
    result['DIC2'].append(DIC2 - Ln)
    result['CV'].append(CV - Ln)

# 結果をcsvに保存
pd.DataFrame(result).to_csv('NN/result.csv', index=False)
