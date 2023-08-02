import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

n,p=10,0.5
x=np.arange(0,10,0.001)
plt.plot(x,binom.pmf(x,n,p))
plt.show()
