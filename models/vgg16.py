'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 26/11/2023
    Descrição:
        - Implementa a rede VGGNet16
'''

from tensorflow import keras

def vgg16():
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)

    base_model.trainable = False

    print(base_model.summary())

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    return model