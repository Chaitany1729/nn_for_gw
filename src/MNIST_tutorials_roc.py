#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:41:15 2020
MNIST Fasion 
@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from roc import roc_curve


def buildModel():
    model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)), keras.layers.Dense(128,activation='relu'), keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def addNoise(percent, train_data):
    mu, sigma = np.mean(train_data), np.std(train_data)
    return train_data + np.random.normal(percent*mu/100, sigma, train_data.shape)
    

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

model = buildModel()

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
predicted = model.predict(test_images)

model.fit(train_images, train_labels, epochs=10)
test_images_twenty = addNoise(20, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_twenty,  test_labels, verbose=2)
predicted_twenty_noise = model.predict(test_images_twenty)

model.fit(train_images, train_labels, epochs=10)
test_images_thirty = addNoise(30, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_thirty,  test_labels, verbose=2)
predicted_thirty_noise = model.predict(test_images_thirty)

model.fit(train_images, train_labels, epochs=10)
test_images_fifty = addNoise(50, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_fifty,  test_labels, verbose=2)
predicted_fifty_noise = model.predict(test_images_fifty)


plt.figure(figsize=(10,10))
plt.title('ROC Curve for Class Shirt')
plt.ylabel('TPR (sensitivity)')
plt.xlabel('FPR (1 - specificity)')
x = np.arange(0,1,0.0001)
no_discrimination = x
plt.plot(x, no_discrimination, '--' )
roc_curve(test_labels,predicted[:,6],6)
roc_curve(test_labels,predicted_twenty_noise[:,6],6)
roc_curve(test_labels,predicted_thirty_noise[:,6],6)
roc_curve(test_labels,predicted_fifty_noise[:,6],6)
plt.legend(('Line of No-discrimination', 'Without Noise', '20% Noise', '30% Noise', '50% Noise'))

