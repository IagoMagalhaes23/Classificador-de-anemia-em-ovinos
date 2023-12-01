'''
    Autores: 
    Data: 09/11/2023
    Descrição:
        -
'''

import os
import random
import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import CarNet, efficientnetb0, inceptionV3, vgg16
from utils import functions as fct

caminho = 'dataset/'
size = 224

data = fct.readFiles(caminho)

random.shuffle(data)

df = pd.DataFrame(data, columns=['image', 'label'])

df = df.reset_index(drop = True)

X_train, X_test, y_train, y_test = train_test_split(df["image"], df["label"], test_size=0.1, shuffle=False)

X_train.reset_index()
X_test.reset_index()
y_train.reset_index()
y_test.reset_index()

print('----------------------------- SHAPES DF -----------------------------')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('---------------------------------------------------------------------')

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

X_train = fct.compose_dataset(X_train, size)
X_test = fct.compose_dataset(X_test, size)

datagen.fit(X_train)
datagen.fit(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('----------------------------- DATA AGUMENTATION -----------------------------')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('---------------------------------------------------------------------')

vgg = vgg16.vgg16()
history = vgg.fit(X_train, y_train, batch_size=8, validation_data=(X_test, y_test), epochs=1000)

inception = inceptionV3.inceptionV3()
history = inception.fit(X_train, y_train, batch_size=8, validation_data=(X_test, y_test), epochs=1000)

efficient = efficientnetb0.efficientNetB0()
history = efficient.fit(X_train, y_train, batch_size=8, validation_data=(X_test, y_test), epochs=1000)

carnet = CarNet.carnet()
history = carnet.fit(X_train, y_train, batch_size=8, validation_data=(X_test, y_test), epochs=1000)