import pandas as pd
import matplotlib.pyplot as plt

# ヒストグラムを書く
bins = 50
range = (-0.1, 1)

df = pd.read_csv('NN/result.csv')
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
axes[0, 0].hist(df['G'], bins=bins, range=range)
axes[0, 1].hist(df['AIC'], bins=bins, range=range)
axes[0, 2].hist(df['WAIC'], bins=bins, range=range)
axes[1, 0].hist(df['DIC1'], bins=bins, range=range)
axes[1, 1].hist(df['DIC2'], bins=bins, range=range)
axes[0, 0].set_title('G')
axes[0, 1].set_title('AIC')
axes[0, 2].set_title('WAIC')
axes[1, 0].set_title('DIC1')
axes[1, 1].set_title('DIC2')
plt.savefig('NN/result.png')
