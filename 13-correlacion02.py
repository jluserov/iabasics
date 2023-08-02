import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
# Creamos un array de 50 números aleatorios entre el 0 y el 10
var1 = np.random.randint(0, 10, 50)
# Creamos una correlación positiva sumandole al primer array un ruido blanco
# En sí, estamos añadiendo un termino de erro que se distribuye de forma normal.
var2 = var1 + np.random.normal(0, 0.1, 50)
# Calculamos el nivel de correlación, si no hubieramos añadido ese error, hubiera sido una correlación perfecta.
print(np.corrcoef(var1, var2))
coefcorr=np.corrcoef(var1, var2)[0,1]
plt.scatter(var1,var2)
plt.show()
