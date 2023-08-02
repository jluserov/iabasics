from pylab import *
import numpy as np
import matplotlib.pyplot as plt
#hay que instalar antes scikit-learn
from sklearn.metrics import r2_score


np.random.seed(2)
pagespeeds=np.random.normal(3.0,1.0,1000)
#purchaseamount=np.random.normal(50.0,10.0,1000)
purchaseamount=100-(pagespeeds+np.random.normal(0,0.7,1000))*3
scatter(pagespeeds,purchaseamount)

x=np.array(pagespeeds)
y=np.array(purchaseamount)
p4=np.poly1d(np.polyfit(x,y,4))
xp=np.linspace(0,7,100)
r2=r2_score(y,p4(x))
print(r2)
plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

