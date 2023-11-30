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

# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
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






from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow import keras

# Suponha que você tenha uma camada de saída convolucional chamada "saida_conv"
# Substitua isso com a camada de saída real do seu modelo

# Saída da camada convolucional
saida_conv = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)

# Adiciona camadas densas para prever as coordenadas da bounding box
saida_flatten = Flatten()(saida_conv)
camada_total = Dense(128, activation='relu')(saida_flatten)
camada_total = Dense(64, activation='relu')(camada_total)

# Camadas para prever as coordenadas (x, y, largura, altura)
saida_x = Dense(1, activation='sigmoid', name='saida_x')(camada_total)
saida_y = Dense(1, activation='sigmoid', name='saida_y')(camada_total)
saida_largura = Dense(1, activation='sigmoid', name='saida_largura')(camada_total)
saida_altura = Dense(1, activation='sigmoid', name='saida_altura')(camada_total)

# Criação do modelo
modelo = Model(inputs=entrada, outputs=[saida_x, saida_y, saida_largura, saida_altura])

# Compilação do modelo (usando uma função de perda adequada, como mean squared error)
modelo.compile(optimizer='adam', loss={'saida_x': 'mean_squared_error',
                                       'saida_y': 'mean_squared_error',
                                       'saida_largura': 'mean_squared_error',
                                       'saida_altura': 'mean_squared_error'})

# Treinamento do modelo (substitua os dados de treinamento e os rótulos com seus próprios dados)
modelo.fit(x=X_train, y={y_train[['x', 'y', 'w', 'h']]},
            epochs=5, batch_size=4)