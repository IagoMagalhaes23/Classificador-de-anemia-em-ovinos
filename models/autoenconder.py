'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 26/11/2023
    Descrição:
        - Implementa a rede DeepEncoder
'''

import tensorflow as tf
from tensorflow.keras import layers, models

def deepEncoder(input_shape, encoding_dim):
    # Encoder
    input_layer = tf.keras.Input(shape=input_shape)
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(decoded)

    # Criar modelo
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)

    # Compilar o modelo
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder