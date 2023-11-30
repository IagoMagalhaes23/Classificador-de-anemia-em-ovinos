'''
    Autores: 
    Data: 09/11/2023
    Descrição:
        -
'''

import os
import random
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import vgg16
from utils import functions as fct

caminho_train = 'dataset/train/Mucosa'
caminho_test = 'dataset/test/Mucosa'
caminho_valid = 'dataset/valid'

data_train = fct.readFiles(caminho_train)
data_test= fct.readFiles(caminho_test)

random.shuffle(data_train)
random.shuffle(data_test)

df_train = pd.DataFrame(data_train, columns=['image', 'x', 'y', 'w', 'h'])
df_test = pd.DataFrame(data_test, columns=['image', 'x', 'y', 'w', 'h'])

df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

print('----------------------------- SHAPES DF -----------------------------')
print(df_train.shape)
print(df_test.shape)
print('---------------------------------------------------------------------')

X_train, y_train = fct.compose_dataset(df_train)
X_test, y_test = fct.compose_dataset(df_test)

print('----------------------------- SHAPES LABEL -----------------------------')
print('Treino shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
print('Teste shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))
print('------------------------------------------------------------------------')
