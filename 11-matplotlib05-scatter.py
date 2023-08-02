import numpy as np
import scipy.stats as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

"""
plt.xkcd()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30,10])

data=np.ones(100)
data[70:]-=np.arange(30)

plt.annotate('ANOTACIÓN',xy=(70,1),arrowprops=dict(arrowstyle='->'),xytext=(15,-10))
plt.plot(data)

plt.xlabel('time')
plt.ylabel('my overall health')

plt.show()
"""
plt.rcdefaults()

x=np.random.randn(500)
y=np.random.randn(500)
plt.scatter(x,y)
plt.show()
