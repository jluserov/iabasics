import numpy as np
import scipy.stats as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

x=np.arange(-3,3,0.001)

#ajustar los ejes
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0,1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
#añadir grid
axes.grid()
#dos gráficos en el mismo
#plt.plot(x,norm.pdf(x))
#colores
plt.plot(x,norm.pdf(x),'b-')
#plt.plot(x,norm.pdf(x,1.0,0.5))
plt.plot(x,norm.pdf(x,1.0,0.5),'r:')
#etiquetas
plt.xlabel('Greebles')
plt.ylabel('Probability')
plt.legend(['Sneetches','Gacks'],loc=4)

#guardar en un archivo
plt.savefig('grafico1.png',format='png')
plt.show()

