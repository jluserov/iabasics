import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mu=5.0
sigma=2.0
values=np.random.normal(mu,sigma,10000)
plt.hist(values,50)
plt.show()
