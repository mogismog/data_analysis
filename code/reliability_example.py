#!/usr/bin/env python

import numpy as np
from research.reliability import reliability_diagram as rd
from research.reliability import relia_roc as rr
import matplotlib.pyplot as plt

fcst = np.random.ranf(5000)*100.
verif = np.rint(np.random.ranf(5000))

fig = plt.figure(figsize=(9.5,9.5))

rd(fig,fcst,verif,bootstrap=False)
plt.show()
plt.close()

fig = plt.figure(figsize=(13.5,6.5))
rr(fig,fcst,verif,bootstrap=False)
plt.show()
plt.close()
