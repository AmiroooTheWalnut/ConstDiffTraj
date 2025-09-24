'''

'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

def generate_ts(timesteps, num):
    return np.random.randint(0, timesteps, size=num)
    # return np.random.randint(timesteps-3, timesteps, size=num)

def forward_noise(timesteps, x, t):
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(size=x.shape)*0.8  # noise mask
    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

class PairwiseSubtractionLayer(tf.keras.layers.Layer):
    def __init__(self, B,lenForbidden,maxLengthSize, **kwargs):
        super(PairwiseSubtractionLayer, self).__init__(**kwargs)
        self.B = tf.convert_to_tensor(B, dtype=tf.float32)  # Store B as a constant tensor
        self.lenForbidden=lenForbidden
        self.maxLengthSize=maxLengthSize

    def custom_activation(self,x):
        a = tf.sigmoid(10*x)
        b = tf.sigmoid(-10*x)
        return a * b

    def call(self, A):
        # Reshape for broadcasting
        A_expanded = tf.expand_dims(A, axis=3)  # Shape (batch, 2, 1)
        B_expanded = tf.expand_dims(self.B, axis=0)  # Shape (1, 2, 5)

        # Subtraction with broadcasting
        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        C1 = self.custom_activation(C)

        splittedChannels = tf.split(C1, [1, 1], axis=2)

        min_channels = tf.keras.layers.minimum([splittedChannels[0], splittedChannels[1]])

        max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, self.lenForbidden),
                                                strides=1, padding="valid")
        reshapedPooled = tf.keras.layers.Reshape((1, self.lenForbidden, self.maxLengthSize))(min_channels)
        poolValue2 = max_pool_2d(reshapedPooled)

        poolValue3 = tf.keras.layers.Reshape((-1,))(poolValue2)

        poolValue4=tf.reduce_min(poolValue3, axis=1, keepdims=True)

        return poolValue4

    def get_config(self):
        config = super().get_config()
        config.update({
            "forbMatrix": self.B.numpy().tolist(),
            "forbLength": self.lenForbidden,
            "maxLengthSize": self.maxLengthSize
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['forbMatrix'], config['forbLength'], config['maxLengthSize'])

def make_model(forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2):
    # x_input = layers.Input(shape=(maxLengthSize, temporalFeatureSize), name='x_input')
    #
    # x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')
    # x_forb_strips = PairwiseSubtractionLayer(forbiddenSerialMap,lenForbidden,maxLengthSize)(x_input)
    # x_forb_dense = layers.Dense(maxLengthSize * 128)(x_forb_strips)
    # x_forb_start = layers.Reshape((maxLengthSize, 128))(x_forb_dense)
    #
    # # x_forb_start = layers.Conv1D(128, kernel_size=7, padding='same')(x_forb_start)
    # # x_forb_start = layers.Activation('sigmoid')(x_forb_start)
    # # x_forb_start = layers.Conv1D(128, kernel_size=5, padding='same')(x_forb_start)
    # # x_forb_start = layers.Activation('sigmoid')(x_forb_start)
    # # x_forb_start = layers.Conv1D(128, kernel_size=3, padding='same')(x_forb_start)
    # # x_forb_start = layers.Activation('sigmoid')(x_forb_start)
    #
    # # x_forb_final = layers.Dense(128)(x_forb_start)
    #
    # # x_im = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    # # x_im = layers.Activation('relu')(x_im)
    #
    # time_parameter = layers.Dense(maxLengthSize * 128)(x_ts)
    # im_parameter = layers.Dense(maxLengthSize * 32)(x_input)
    # reshapedTime = layers.Reshape((maxLengthSize, 128))(time_parameter)
    # reshapedIm = layers.Reshape((maxLengthSize, 128))(im_parameter)
    #
    # timed_img = layers.Multiply()([reshapedTime,reshapedIm])
    #
    # x_parameter = layers.Conv1D(128, kernel_size=7, padding='same')(timed_img)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    # x_parameter = layers.Conv1D(128, kernel_size=5, padding='same')(x_parameter)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    # x_parameter = layers.Conv1D(128, kernel_size=3, padding='same')(x_parameter)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    #
    # x_parameter = layers.Dense(128)(x_parameter)
    #
    # # flatImage = layers.Flatten(data_format='channels_last')(x_parameter)
    #
    # x_direct = layers.Dense(128)(x_input)
    #
    # concatenatedLayer = layers.Add()([x_parameter,reshapedTime,x_direct])
    #
    # # concatFinal=layers.Concatenate()([concatenatedLayer, x_forb_start])
    #
    # #finalMult = layers.Multiply()([concatenatedLayer, x_forb_start])
    #
    # # finalDense=layers.Dense(128)(concatFinal)
    # # finalDense = layers.Activation('sigmoid')(finalDense)
    #
    # x = layers.Conv1D(2, kernel_size=1, padding='same')(concatenatedLayer)




    x_input = layers.Input(shape=(maxLengthSize, temporalFeatureSize), name='input_tensor')

    x_ts = x_ts_input = layers.Input(shape=(1,), name='input_scalar')

    # x_im = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    # x_im = layers.Activation('relu')(x_im)

    x_forb_strips = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)(x_input)
    x_forb_dense = layers.Dense(maxLengthSize * 32)(x_forb_strips)
    x_forb_start = layers.Reshape((maxLengthSize, 32))(x_forb_dense)

    time_parameter = layers.Dense(maxLengthSize * 128)(x_ts)
    im_parameter = layers.Dense(maxLengthSize * 32)(x_input)
    reshapedTime = layers.Reshape((maxLengthSize, 128))(time_parameter)
    reshapedIm = layers.Reshape((maxLengthSize, 128))(im_parameter)

    timed_img = layers.Multiply()([reshapedTime, reshapedIm])

    x_parameter = layers.Conv1D(128, kernel_size=7, padding='same')(timed_img)
    x_parameter = layers.Activation('sigmoid')(x_parameter)
    x_parameter = layers.Conv1D(128, kernel_size=5, padding='same')(x_parameter)
    x_parameter = layers.Activation('sigmoid')(x_parameter)
    x_parameter = layers.Conv1D(128, kernel_size=3, padding='same')(x_parameter)
    x_parameter = layers.Activation('sigmoid')(x_parameter)

    x_parameter = layers.Dense(128)(x_parameter)

    # flatImage = layers.Flatten(data_format='channels_last')(x_parameter)

    x_direct = layers.Dense(128)(x_input)

    concatenatedLayer = layers.Add()([x_parameter, reshapedTime, x_direct])

    concatFinal = layers.Concatenate()([concatenatedLayer, x_forb_start])

    finalDense = layers.Dense(256)(concatFinal)

    x = layers.Conv1D(2, kernel_size=1, padding='same', name="output_layer")(finalDense)

    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model