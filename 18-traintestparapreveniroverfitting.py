import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
pagespeeds=np.random.normal(3.0,1.0,100)
purchaseamount=np.random.normal(50.0,30.0,100)
scatter(pagespeeds,purchaseamount)

trainX=pagespeeds[:80]
testX=pagespeeds[80:]
trainY=purchaseamount[:80]
testY=purchaseamount[80:]

x=np.array(trainX)
y=np.array(trainY)
p4=np.poly1d(np.polyfit(x,y,8))

xp=np.linspace(0,7,100)
axes=plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

testx = np.array(testX)
testy = np.array(testY)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.show()

r2 = r2_score(testy, p4(testx))
r2 = r2_score(np.array(trainY), p4(np.array(trainX)))
print(r2)
