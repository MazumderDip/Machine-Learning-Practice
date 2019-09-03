from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
len(train_labels)
print(train_labels)
print(test_images.shape)
len(test_labels)

"""plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()"""

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
model.save('fashion_mnist.model')
new_model = tf.keras.models.load_model('fashion_mnist.model')
print('Test accuracy:', test_acc)
predictions = new_model.predict(test_images)

for i in range(25):
    print("Test Actual:" + class_names[test_labels[i]])
    print("Perdicted:" + class_names[np.argmax(predictions[i])])
    plt.imshow(test_images[i])
    plt.xlabel(
        "Test Actual:" + class_names[test_labels[i]] + "\n" + "PrEdicted:" + class_names[np.argmax(predictions[i])])
plt.show()
