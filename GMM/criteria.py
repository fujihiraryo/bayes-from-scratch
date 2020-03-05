import numpy as np


class Criteria():
    def __init__(self, model, true, train, test, params, tparam):
        self.model = model
        self.true = true
        self.train = train
        self.test = test
        self.params = params
        self.tparam = tparam
        self.n = len(train)
        self.m = len(params)

        # 計算
        self.L = self.calcL()
        self.Ln = self.calcLn()
        self.T = self.calcT()
        self.G = self.calcG()
        self.AIC = self.calcAIC()
        self.V = self.calcV()
        self.WAIC = self.calcWAIC()
        self.CV = self.calcCV()

    def ptrue(self, x):
        a, b = self.tparam
        return self.true.p

    def calcL(self):
        res = 0
        for x in self.test:
            res += -np.log(self.ptrue(x))
        return res / self.n

    def calcLn(self):
        res = 0
        for x in self.train:
            res += -np.log(self.ptrue(x))
        return res / self.n

    def calcT(self):
        res = 0
        for x in self.train:
            res += -np.log(self.true.predict(x))
        return res / self.n

    def calcG(self):
        res = 0
        for x in self.test:
            res += -np.log(self.true.predict(x))
        return res / self.n

    def calcAIC(self):
        return self.T + self.model.d / self.n

    def calcV(self):
        res = 0
        for x in self.train:
            tmp0, tmp1 = 0, 0
            for a, b in self.params:
                tmp0 += np.log(self.model.p(x, a, b))**2
                tmp1 += np.log(self.model.p(x, a, b))
            tmp0 = tmp0 / self.m
            tmp1 = (tmp1 / self.m)**2
            res += tmp0 - tmp1
        return res

    def calcWAIC(self):
        return self.T + self.V / self.n

    def calcCV(self):
        res = 0
        for x in self.train:
            tmp = 0
            for a, b in self.params:
                tmp += self.model.p(x, a, b)**(-1)
            res += np.log(tmp / self.m)
        return res / self.n
