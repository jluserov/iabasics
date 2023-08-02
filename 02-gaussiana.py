import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x=np.arange(-3,3,0.01)
plt.plot(x,norm.pdf(x))
plt.show()
