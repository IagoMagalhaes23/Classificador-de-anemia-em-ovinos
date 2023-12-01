'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 30/11/2023
    Descrição:
        - Implementa a rede InceptionV3
        - !wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
'''

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model

def inceptionV3():
    local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = None)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    pre_trained_model.summary()

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2, activation='sigmoid')(x)           

    model = Model(pre_trained_model.input, x) 

    model.summary()

    model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
    
    return model