import model
import numpy as np

# 設定
M, H0, N = 3, 1, 2
H = 3
n = 100
T = 100
a0 = np.random.normal(0, 1, (H0, N))
b0 = np.random.normal(0, 1, (H0, M))


# 予測分布
def p(x, y):
    return np.mean([nn.p(x, y, a, b) for a, b in params])


# 予測関数
def R(x):
    return np.array(
        [np.mean([nn.R(x, a, b)[j] for a, b in params]) for j in range(N)])


# 独立した実験をT回行う
for t in range(T):
    # データ生成
    nn0 = model.NN(M, H0, N)
    XY = [nn0.generate(a0, b0) for _ in range(2 * n)]
    XYt, XYv = XY[:n], XY[n:]
    Xt = [xy[0] for xy in XYt]
    Xv = [xy[0] for xy in XYv]
    Yt = [xy[1] for xy in XYt]
    Yv = [xy[1] for xy in XYv]

    # 損失の計算
    nn = model.NN(M, H, N)
    params = nn.sampling(Xt, Yt)
    L = np.mean([-np.log(nn0.p(x, y, a0, b0)) for x, y in XYv])
    T = np.mean([-np.log(p(x, y)) for x, y in XYt])
    G = np.mean([-np.log(p(x, y)) for x, y in XYv])
    AIC = T + nn.d / n
    V = np.sum([
        np.mean([np.log(nn.p(x, y, a, b))**2 for a, b in params]) -
        np.mean([np.log(nn.p(x, y, a, b)) for a, b in params])**2
        for x, y in XYt
    ])
    WAIC = T + V / n
    am = np.mean([a for a, b in params], axis=0)
    bm = np.mean([b for a, b in params], axis=0)
    Deff1 = 2 * np.mean([
        np.log(nn.p(x, y, am, bm)) -
        np.mean([np.log(nn.p(x, y, a, b)) for a, b in params]) for x, y in XYt
    ])
    Deff2 = 2 * (np.mean([
        np.sum([np.log(nn.p(x, y, a, b)) for x, y in XYt])**2
        for a, b in params
    ]) - np.mean([
        np.sum([np.log(nn.p(x, y, a, b)) for x, y in XYt]) for a, b in params
    ])**2)
    DIC1 = T + Deff1 / n
    DIC2 = T + Deff2 / n
    print(
        f'G={G-L:.3g}, AIC={AIC-L:.3g}, WAIC={WAIC-L:.3g}, DIC1={DIC1-L:.3g}, DIC2={DIC2-L:.3g}'
    )
