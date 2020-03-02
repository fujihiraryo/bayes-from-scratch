import numpy as np
import model
import criteria

M = 3  # 入力次元
N = 2  # 出力次元
H0 = 1  # 真の隠れユニット数
H = 3  # モデルの隠れユニット数
n = 100  # サンプル数
T = 5  # 実験の回数
size = 400  # MCMCの遷移回数
burn = 100  # サンプルの最初何個捨てるか

# a0 = np.random.normal(0, 1, (H0, N))
# b0 = np.random.normal(0, 1, (H0, M))
a0 = np.ones((H0, N))
b0 = np.ones((H0, M))

# 学習とモデルの評価
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
    T = crt.T
    AIC = crt.AIC
    WAIC = crt.WAIC
    DIC1 = crt.DIC1
    DIC2 = crt.DIC2
    print(
        f'G={G:.3g}, T={T:.3g}, AIC={AIC:.3g},WAIC={WAIC:.3g}, DIC1={DIC1:.3g}, DIC2={DIC2:.3g}'
    )
