#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:41:15 2020
MNIST Fasion 
@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import tensorflow as tf
from tensorflow import keras 
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def plot(fpr,tpr,roc_auc,n_classes):
    lw = 2
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
def buildModel():
    model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)), keras.layers.Dense(128,activation='relu'), keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def addNoise(percent, train_data):
    mu, sigma = np.mean(train_data), np.std(train_data)
    return train_data + np.random.normal(percent*mu/100, sigma, train_data.shape)
    
def plotRoc(test_labels,predicted_labels,n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    true_label = np.zeros( predicted_labels.shape)
    for i in range(test_labels.size):
        true_label[i][test_labels[i]] = 1
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_label[:,i], predicted_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(),predicted_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])


    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plot(fpr,tpr,roc_auc,n_classes)

    


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
plotRoc(test_labels,predicted,10)


model.fit(train_images, train_labels, epochs=10)
test_images_twenty = addNoise(20, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_twenty,  test_labels, verbose=2)
predicted_twenty_noise = model.predict(test_images_twenty)
plotRoc(test_labels,predicted_twenty_noise,10)

model.fit(train_images, train_labels, epochs=10)
test_images_thirty = addNoise(30, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_thirty,  test_labels, verbose=2)
predicted_thirty_noise = model.predict(test_images_thirty)
plotRoc(test_labels,predicted_thirty_noise,10)

model.fit(train_images, train_labels, epochs=10)
test_images_fifty = addNoise(50, train_data=test_images)
test_loss, test_acc = model.evaluate(test_images_fifty,  test_labels, verbose=2)
predicted_fifty_noise = model.predict(test_images_fifty)
plotRoc(test_labels,predicted_fifty_noise,10)
