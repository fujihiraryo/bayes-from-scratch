import numpy as np
import pandas as pd

df = pd.read_csv('NN/result.csv')
G = df['G']
AIC = df['AIC']
WAIC = df['WAIC']
DIC1 = df['DIC1']
DIC2 = df['DIC2']
CV = df['CV']

# 平均と分散
print(f'G mean={G.mean():.3f} std={G.std():.3f}')
print(f'AIC mean={AIC.mean():.3f} std={AIC.std():.3f}')
print(f'WAIC mean={WAIC.mean():.3f} std={WAIC.std():.3f}')
print(f'DIC1 mean={DIC1.mean():.3f} std={DIC1.std():.3f}')
print(f'DIC2 mean={DIC2.mean():.3f} std={DIC2.std():.3f}')
print(f'CV mean={CV.mean():.3f} std={CV.std():.3f}')

# それぞれの指標が汎化誤差とどれだけずれてるか
AIC = ((AIC - G)**2).sum()
WAIC = ((WAIC - G)**2).sum()
DIC1 = ((DIC1 - G)**2).sum()
DIC2 = ((DIC2 - G)**2).sum()
CV = ((CV - G)**2).sum()
print(f'AIC {AIC:.3f}')
print(f'WAIC {WAIC:.3f}')
print(f'DIC1 {DIC1:.3f}')
print(f'DIC2 {DIC2:.3f}')
print(f'CV {CV:.3f}')
