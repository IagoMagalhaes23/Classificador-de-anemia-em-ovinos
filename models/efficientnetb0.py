'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 26/11/2023
    Descrição:
        - Implementa a rede EfficientNetB0
'''

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

def efficientNetB0():
    NUM_CLASSES = 1
    IMG_SIZE = 224
    size = (IMG_SIZE, IMG_SIZE)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model