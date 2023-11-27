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

# from models import vgg16
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

# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range = 0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=False,
#     vertical_flip=False
# )

# datagen.fit(X_train)
# datagen.fit(X_test)

# print('----------------------------- SHAPES LABEL -----------------------------')
# print('Treino shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
# print('Teste shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))
# print('------------------------------------------------------------------------')

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_valid = to_categorical(y_valid)

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

# Carregar a VGGNet pré-treinada (excluindo a camada totalmente conectada no topo)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar os pesos da VGGNet para que eles não sejam atualizados durante o treinamento
for layer in vgg_model.layers:
    layer.trainable = False

# Adicionar camadas de detecção no topo da VGGNet
x = vgg_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

# Adapte o número de saídas à sua tarefa de detecção
# No exemplo abaixo, assumimos uma tarefa de detecção com 4 classes
num_classes = 4
detecao = layers.Dense(num_classes, activation='softmax')(x)

# Criar o modelo final
modelo_final = Model(inputs=vgg_model.input, outputs=detecao)

# Compilar o modelo e prepará-lo para o treinamento
modelo_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Exibir a arquitetura do modelo
modelo_final.summary()

modelo_final.fit(X_train, y_train,
           epochs=5, batch_size=4, validation_split=0.2)