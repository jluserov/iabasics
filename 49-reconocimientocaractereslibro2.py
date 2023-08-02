from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()

info = mnist_builder.info
print(info)

mnist_train = mnist_builder.as_dataset(split="train")
fig = tfds.show_examples(info, mnist_train)

#flatten_image = partial(flatten_image, label=True)
#for image, label in mnist_train.map(flatten_image).take(1):
#    plt.imshow(image.numpy().reshape(28,28).astype(np.float32),cmap=plt.get_cmap("gray"))
#    print("Label: %d" % label.numpy())

def flatten_image(x, label=True):
    if label:
        return (tf.divide(tf.dtypes.cast(tf.reshape(x["image"],(1,28*28)), tf.float32), 256.0) , x["label"])
    else:
        return (tf.divide(tf.dtypes.cast(tf.reshape(x["image"],(1,28*28)), tf.float32), 256.0))

for image, label in mnist_train.map(flatten_image).take(1):
    plt.imshow(image.numpy().astype(np.float32), cmap=plt.get_cmap("gray"))
    print("Label: %d" % label.numpy())
