import numpy as np
#import statistics as stats
import scipy.stats as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
from pylab import *

def de_mean(x):
    xmean=mean(x)
    return(xi-xmean for xi in x)

def covariance(x,y):
    n=len(x)
    return np.dot(de_mean(x),de_mean(y))/(n-1)

pagespeeds=np.random.normal(3.0,1.0,1000)
purchaseamount=np.random.normal(50.0,10.0,1000)

plt.scatter(pagespeeds,purchaseamount)
covariance(pagespeeds,purchaseamount)
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

plt.rcdefaults()

x=np.random.randn(500)
y=np.random.randn(500)
plt.scatter(x,y)
plt.show()

incomes=np.random.normal(27000,15000,10000)
plt.hist(incomes,50)
plt.show()

uniformskewed=np.random.rand(100)*100-40
high_outliers=np.random.rand(10)*50+100
low_outliers=np.random.rand(10)*50-100
data=np.concatenate((uniformskewed,high_outliers,low_outliers))
plt.boxplot(data)
plt.show()
"""
