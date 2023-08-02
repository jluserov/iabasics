import numpy as np
import scipy.stats as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

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

plt.rcdefaults()

values=[12,55,4,32,14]
colors=['r','g','b','c','m']
explode=[0,0,0.2,0,0]
labels=['I','I','I','I','I']
plt.pie(values,colors=colors,labels=labels,explode=explode)
plt.title("Localizaciones")
plt.show()
