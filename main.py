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

# from sklearn.model_selection import train_test_split

# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from models import CarNet, efficientnetb0, inceptionV3, vgg16, vitB16
from utils import functions as fct

caminho = 'dataset/'

data = fct.readFiles(caminho)

random.shuffle(data)

df = pd.DataFrame(data, columns=['image', 'label'])

df = df.reset_index(drop = True)

# X_train, X_test, y_train, y_test = train_test_split(df.drop(df.columns["label"], axis=1), df["label"], test_size=0.1, shuffle=False)

print('----------------------------- SHAPES DF -----------------------------')
print(df.shape)
print('---------------------------------------------------------------------')

# X_train, y_train = fct.compose_dataset(df_train)
# X_test, y_test = fct.compose_dataset(df_test)

# print('----------------------------- SHAPES LABEL -----------------------------')
# print('Treino shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
# print('Teste shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))
# print('------------------------------------------------------------------------')
