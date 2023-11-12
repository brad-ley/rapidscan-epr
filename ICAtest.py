from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as P
import pandas as pd
from twoComponentSVDsim import lorentzian

dat = pd.read_csv('/Users/Brad/Desktop/testICA.csv', index_col=False)
v = dat.loc[:, ['Unnamed' not in ii for ii in dat.columns]].to_numpy()

B = np.linspace(-25, 25, 1000)
t = np.linspace(0, 500, 500)
t1_light = 50
t2_light = 50
pct_change = 80
g1 = (90 * np.ones(len(t)) - pct_change * np.heaviside(t - t1_light, 0.5) * np.exp(-(t - t1_light) / 100))
g2 = (10 * np.ones(len(t)) + pct_change * np.heaviside(t - t2_light, 0.5) * np.exp(-(t - t2_light) / 100))
l1 = lorentzian(B, 10, 9)
l2 = lorentzian(B, 10, 8)
v = np.array([l1 * g1[ii] / 100 + l2 * g2[ii] / 100 for ii, _ in enumerate(t)]).T
v2 = l1 - l2

n = 4
ica = FastICA(n_components=n)
S_ = ica.fit_transform(v)  # estimated independent sources

fig, ax = plt.subplots()
ax.imshow(v, aspect='auto')
fig, a = plt.subplots()
# ax.imshow(S_, aspect='auto')
for i in range(n):
    a.plot(S_[:, i]/np.max(S_[:, i]) + i * n)
a.plot(v2/np.max(v2))
plt.show()
