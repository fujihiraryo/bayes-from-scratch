import numpy as np


class Criteria():
    def __init__(self, model, true_model, train_data, test_data, params,
                 true_param):
        self.model = model
        self.true_model = true_model
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.true_param = true_param
        self.n = len(train_data)
        self.m = len(params)

        # 計算
        self.L = self.calcL()
        self.Ln = self.calcLn()
        self.T = self.calcT()
        self.G = self.calcG()
        self.AIC = self.calcAIC()
        self.V = self.calcV()
        self.WAIC = self.calcWAIC()
        self.Deff1 = self.calcDeff1()
        self.Deff2 = self.calcDeff2()
        self.DIC1 = self.calcDIC1()
        self.DIC2 = self.calcDIC2()

    def predict(self, x, y):
        # 予測分布
        res = 0
        for a, b in self.params:
            res += self.model.p(x, y, a, b)
        return res / self.m

    def ptrue(self, x, y):
        # 真の分布
        a, b = self.true_param
        return self.true_model.p(x, y, a, b)

    def calcL(self):
        res = 0
        for x, y in self.test_data:
            res += -np.log(self.ptrue(x, y))
        return res / self.n

    def calcLn(self):
        res = 0
        for x, y in self.train_data:
            res += -np.log(self.ptrue(x, y))
        return res / self.n

    def calcT(self):
        res = 0
        for x, y in self.train_data:
            res += -np.log(self.predict(x, y))
        return res / self.n

    def calcG(self):
        res = 0
        for x, y in self.test_data:
            res += -np.log(self.predict(x, y))
        return res / self.n

    def calcAIC(self):
        return self.T + self.model.d / self.n

    def calcV(self):
        res = 0
        for x, y in self.train_data:
            tmp0, tmp1 = 0, 0
            for a, b in self.params:
                tmp0 += np.log(self.model.p(x, y, a, b))**2
                tmp1 += np.log(self.model.p(x, y, a, b))
            tmp0 = tmp0 / self.m
            tmp1 = (tmp1 / self.m)**2
            res += tmp0 - tmp1
        return res

    def calcWAIC(self):
        return self.T + self.V / self.n

    def calcDeff1(self):
        am = np.zeros((self.model.H, self.model.N))
        bm = np.zeros((self.model.H, self.model.M))
        for a, b in self.params:
            am = am + a
            bm = bm + b
        am = am / self.m
        bm = bm / self.m
        res = 0
        for x, y in self.train_data:
            res += np.log(self.model.p(x, y, am, bm))
            tmp = 0
            for a, b in self.params:
                tmp += np.log(self.model.p(x, y, a, b))
            tmp = tmp / self.m
            res -= tmp
        return 2 * res / self.n

    def calcDeff2(self):
        res0, res1 = 0, 0
        for a, b in self.params:
            tmp = 0
            for x, y in self.train_data:
                tmp += np.log(self.model.p(x, y, a, b))
            res0 += tmp**2
            res1 += tmp
        return 2 * (res0 / self.m - (res1 / self.m)**2)

    def calcDIC1(self):
        return self.T + self.Deff1 / self.n

    def calcDIC2(self):
        return self.T + self.Deff2 / self.n