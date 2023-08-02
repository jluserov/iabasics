import numpy as np
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt

def predict(x):
    return slope*x+intercept

pagespeeds=np.random.normal(3.0,1.0,1000)
purchaseamount=100-(pagespeeds+np.random.normal(0,0.1,1000))*3
scatter(pagespeeds,purchaseamount)

slope,intercept,r_value,p_value,std_err=stats.linregress(pagespeeds,purchaseamount)
print(r_value)
fitline=predict(pagespeeds)

plt.scatter(pagespeeds,purchaseamount)
plt.plot(pagespeeds,fitline,c='r')
plt.show()



