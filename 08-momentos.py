import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

vals=np.random.normal(0,0.5,10000)
print(np.percentile(vals,50))
print(np.percentile(vals,90))
print(np.percentile(vals,20))

print("Media:"+str(np.mean(vals)))
print("Varianza:"+str(np.var(vals)))
print("Skew:"+str(sp.skew(vals))) #asimetría
#print(np.skew(vals))
print("Kurtosis:"+str(sp.kurtosis(vals)))

plt.hist(vals,50)
plt.show()
