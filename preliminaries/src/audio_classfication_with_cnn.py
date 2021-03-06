#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:10:13 2020

@author: chaitanya
"""
import pandas as pd
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


fulldatasetpath = '/home/chaitanya/Academics/Sem2/sop/data/UrbanSound8K/audio/'

metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')

features = []

for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold'+str(row["fold"]) + '/', str(row["slice_file_name"]))
    
    class_label = row["class"]
    data = extract_features(file_name)
    
    features.append([data, class_label])
    
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])


print('Finished feature extraction from ', len(featuresdf), ' files')

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.20292, random_state = 42)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

num_rows = 40
num_columns = 174
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)

x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape = (num_rows,num_columns,num_channels),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))


model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(num_labels,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 72
num_batch_size = 256

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


y_score = model.predict(x_test)


import matplotlib.pyplot as plt
from scipy import interpolate

P = np.count_nonzero(y_test[:,1])
N = y_test[:,1].size - P
    
threshold = np.array(range(0,1000))
threshold = threshold /1000
    
    
tpr = np.zeros(threshold.size)
fpr = np.zeros(threshold.size)
    
for i in range(threshold.size):
        for j in range(y_score[:,1].size):
            if(y_score[j][1] > threshold[i] and y_test[j][1]==1):
                tpr[i] = tpr[i] + 1
            if(y_score[j][1] > threshold[i] and y_test[j][1] == 0):
                fpr[i] = fpr[i] + 1
                
tpr = tpr / P
fpr = fpr / N

    
roc = interpolate.interp1d(fpr,tpr,fill_value='extrapolate')
x = np.arange(0,1,0.0001)
y = roc(x)
plt.plot(x,y)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC for class: car_horn')