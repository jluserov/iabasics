import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from random import seed, randint
from numpy import array
from fractions import Fraction

N = 10000
X = np.random.random(size=[N,2])
#aquí es donde multiplica
Y = X[:,0] * X[:,1]
X_train = X[:8*N//10,:]
Y_train = Y[:8*N//10]
X_test = X[8*N//10:,:]
Y_test = Y[8*N//10:]

N_in = 2
N_hid = 3
N_out = 1

model = Sequential()
model.add(Dense(N_hid, input_dim=N_in, activation='sigmoid'))
model.add(Dense(N_out, activation='sigmoid'))

model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test),epochs=100, batch_size=100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

N = 1000
X = np.random.random(size=2*N).reshape(N,2)
Y = X[:,0]*X[:,1]
y_pred = model.predict(X)
plt.plot(Y,y_pred,'o',markersize=2)
plt.plot([0,1],[0,1],'r-',linewidth=3)
plt.xlabel('Actual Product', fontsize=25)
plt.ylabel('Network Prediction', fontsize=25)
plt.show()
