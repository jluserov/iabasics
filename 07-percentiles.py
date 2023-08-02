import numpy as np
import matplotlib.pyplot as plt

vals=np.random.normal(0,0.5,10000)
print(np.percentile(vals,50))
print(np.percentile(vals,90))
print(np.percentile(vals,20))
plt.hist(vals,50)
plt.show()


