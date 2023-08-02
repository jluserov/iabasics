import numpy as np
import tensorflow as tf
from random import randrange

trainingInput = [[i, i + randrange(5000)] for i in range(1, 5000)]
trainingOutput = [(input [0] + input [1]) for input  in trainingInput ]

testInput = [[5, 5], [1, 9], [2, 5], [6, 3], [1, 4]]
testOutput = [10, 10, 7, 9, 5]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(2,)))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss=tf.keras.losses.mae, metrics=['mae'])

model.fit(trainingInput, trainingOutput, batch_size=5, epochs=50)

test_loss, test_acc = model.evaluate(testInput, testOutput)
print("Test Accuracy : ", test_acc)
a = np.array([[3, 3000], [4, 5], [1,10], [2,10],[5,9], [4,10], [1,15]])
print(model.predict(a))
