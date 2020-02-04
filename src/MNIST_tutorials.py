#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:41:15 2020
MNIST Fasion 
@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)), keras.layers.Dense(128,activation='relu'), keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.title('Model without noise')
plt.show()


mu, sigma = np.mean(train_images), np.std(train_images)

train_images_twenty = train_images + np.random.normal(0.20*mu, sigma, train_images.shape)
train_images_thirty = train_images + np.random.normal(0.30*mu, sigma, train_images.shape)
train_images_fifty = train_images + np.random.normal(0.50*mu, sigma, train_images.shape)

model.fit(train_images_twenty, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy for 20% gaussian noise:', test_acc)

model.fit(train_images_thirty, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy for 30% noise:', test_acc)

model.fit(train_images_fifty, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy for 50% noise:', test_acc)

test_images_twenty = test_images + np.random.normal(0.20*mu, sigma, test_images.shape)
test_images_thirty = test_images + np.random.normal(0.30*mu, sigma, test_images.shape)
test_images_fifty = test_images + np.random.normal(0.50*mu, sigma, test_images.shape)

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images_twenty,  test_labels, verbose=2)
print('\nTest accuracy for 20% gaussian noise in test data:', test_acc)

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images_thirty,  test_labels, verbose=2)
print('\nTest accuracy for 30% gaussian noise in test data:', test_acc)

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images_fifty,  test_labels, verbose=2)
print('\nTest accuracy for 50% gaussian noise in test data:', test_acc)

predictions = model.predict(test_images_fifty)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.title('Model with 50% noise')
plt.show()