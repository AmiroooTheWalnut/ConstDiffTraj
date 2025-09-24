'''
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ x_ts_input          │ (None, 1)         │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ x_input             │ (None, 4, 2)      │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 512)       │      1,024 │ x_ts_input[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 4, 128)    │        384 │ x_input[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ reshape (Reshape)   │ (None, 4, 128)    │          0 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ reshape_1 (Reshape) │ (None, 4, 128)    │          0 │ dense_1[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multiply (Multiply) │ (None, 4, 128)    │          0 │ reshape[0][0],    │
│                     │                   │            │ reshape_1[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d (Conv1D)     │ (None, 4, 128)    │    114,816 │ multiply[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (None, 4, 128)    │          0 │ conv1d[0][0]      │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_1 (Conv1D)   │ (None, 4, 128)    │     82,048 │ activation[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_1        │ (None, 4, 128)    │          0 │ conv1d_1[0][0]    │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_2 (Conv1D)   │ (None, 4, 128)    │     49,280 │ activation_1[0][… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_2        │ (None, 4, 128)    │          0 │ conv1d_2[0][0]    │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (Dense)     │ (None, 4, 128)    │     16,512 │ activation_2[0][… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (Dense)     │ (None, 4, 128)    │        384 │ x_input[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (Add)           │ (None, 4, 128)    │          0 │ dense_2[0][0],    │
│                     │                   │            │ reshape[0][0],    │
│                     │                   │            │ dense_3[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_3 (Conv1D)   │ (None, 4, 2)      │        258 │ add[0][0]         │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from keras.saving import register_keras_serializable

def generate_ts(timesteps, num):
    return np.random.randint(0, timesteps, size=num)
    # return np.random.randint(timesteps-3, timesteps, size=num)

def forward_noise(timesteps, x, t):
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

@register_keras_serializable()
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

def make_model(forbiddenSerialMap=None, lenForbidden=10,maxLengthSize = 10, temporalFeatureSize=2):
    x = x_input = layers.Input(shape=(maxLengthSize, temporalFeatureSize), name='x_input')

    x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')

    # x_im = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    # x_im = layers.Activation('relu')(x_im)

    time_parameter = layers.Dense(maxLengthSize * 256)(x_ts)
    im_parameter = layers.Dense(maxLengthSize * 64)(x_input)
    reshapedTime = layers.Reshape((maxLengthSize, 256))(time_parameter)
    reshapedIm = layers.Reshape((maxLengthSize, 256))(im_parameter)

    timed_img = layers.Multiply()([reshapedTime,reshapedIm])

    x_parameter = layers.Conv1D(256, kernel_size=7, padding='same')(timed_img)
    x_parameter = layers.Activation('sigmoid')(x_parameter)
    x_parameter = layers.Conv1D(256, kernel_size=5, padding='same')(x_parameter)
    x_parameter = layers.Activation('sigmoid')(x_parameter)
    x_parameter = layers.Conv1D(256, kernel_size=3, padding='same')(x_parameter)
    x_parameter = layers.Activation('sigmoid')(x_parameter)

    x_parameter = layers.Dense(256)(x_parameter)

    # flatImage = layers.Flatten(data_format='channels_last')(x_parameter)

    x_direct = layers.Dense(256)(x_input)

    concatenatedLayer = layers.Add()([x_parameter,reshapedTime,x_direct])

    x = layers.Conv1D(2, kernel_size=1, padding='same')(concatenatedLayer)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model



    # x_input = layers.Input(shape=(maxLengthSize, temporalFeatureSize), name='input_tensor')
    #
    # x_ts = x_ts_input = layers.Input(shape=(1,), name='input_scalar')
    #
    # # x_im = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    # # x_im = layers.Activation('relu')(x_im)
    #
    # x_forb_strips = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)(x_input)
    # x_forb_dense = layers.Dense(maxLengthSize * 64)(x_forb_strips)
    # x_forb_start = layers.Reshape((maxLengthSize, 64))(x_forb_dense)
    #
    # time_parameter = layers.Dense(maxLengthSize * 256)(x_ts)
    # im_parameter = layers.Dense(maxLengthSize * 64)(x_input)
    # reshapedTime = layers.Reshape((maxLengthSize, 256))(time_parameter)
    # reshapedIm = layers.Reshape((maxLengthSize, 256))(im_parameter)
    #
    # timed_img = layers.Multiply()([reshapedTime, reshapedIm])
    #
    # x_parameter = layers.Conv1D(256, kernel_size=7, padding='same')(timed_img)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    # x_parameter = layers.Conv1D(256, kernel_size=5, padding='same')(x_parameter)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    # x_parameter = layers.Conv1D(256, kernel_size=3, padding='same')(x_parameter)
    # x_parameter = layers.Activation('sigmoid')(x_parameter)
    #
    # x_parameter = layers.Dense(256)(x_parameter)
    #
    # # flatImage = layers.Flatten(data_format='channels_last')(x_parameter)
    #
    # x_direct = layers.Dense(256)(x_input)
    #
    # concatenatedLayer = layers.Add()([x_parameter, reshapedTime, x_direct])
    #
    # concatFinal = layers.Concatenate()([concatenatedLayer, x_forb_start])
    #
    # finalDense = layers.Dense(512)(concatFinal)
    #
    # x = layers.Conv1D(2, kernel_size=1, padding='same', name="output_layer")(finalDense)
    #
    # model = tf.keras.models.Model([x_input, x_ts_input], x)
    # return model


class CustomModel(tf.keras.Model):
    def __init__(self, base_model, loss_fn, forbiddens):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = loss_fn
        self.forbiddens = tf.convert_to_tensor(forbiddens, dtype=tf.float32)
        self.mae = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, data):
        (x, time), y = data  # Unpack input (assuming x is the image, time is extra input)

        with tf.GradientTape() as tape:
            y_pred = self.base_model([x, time], training=True)
            loss = self.custom_loss(y, y_pred, time)  # Pass time to loss function

        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return np.array(loss)

    def train_on_batch(self, x, y):
        return self.train_step((x, y))  # Make train_on_batch use the same logic

    def custom_loss(self, y_true, y_pred, time):
        mae_loss = self.mae(y_true, y_pred)
        A_expanded = tf.expand_dims(y_pred, axis=3)
        B_expanded = tf.expand_dims(self.forbiddens, axis=0)
        C = A_expanded - B_expanded
        C1 = self.custom_activation(C)
        penalty = 1 / (0.1 + tf.math.reduce_max(C1,axis=[1,2,3]) * (16^4-time^4+1)*(0.01))
        return mae_loss + 0.01*tf.math.reduce_sum(penalty)
        # return mae_loss

    def custom_activation(self, x):
        a = tf.sigmoid(10 * x)
        b = tf.sigmoid(-10 * x)
        return a * b

    def predict_step(self, data):
        return self.base_model(data, training=False)  # Time might not be used directly unless needed

    #def get_config(self):
    #    config = super().get_config()
    #    config.update({
    #        "forbMatrix": self.forbiddens.numpy().tolist(),
    #        "forbLength": self.lenForbidden,
    #        "maxLengthSize": self.maxLengthSize
    #    })
    #    return config

    @classmethod
    def from_config(cls, config):
        return cls(config['forbMatrix'], config['forbLength'], config['maxLengthSize'])



class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, forbiddens,lenForbidden,maxLengthSize, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize
        self.forbiddens = tf.convert_to_tensor(forbiddens, dtype=tf.float32)

    def custom_activation(self, x):
        a = tf.sigmoid(10 * x)
        b = tf.sigmoid(-10 * x)
        return a * b

    def call(self, y_true, y_pred, time):
        mae_loss = self.mae(y_true, y_pred)
        A_expanded = tf.expand_dims(y_pred, axis=3)
        B_expanded = tf.expand_dims(self.forbiddens, axis=0)
        C = A_expanded - B_expanded
        C1 = self.custom_activation(C)
        penalty = 1/(0.1+tf.math.reduce_sum(C1,axis=[1,2,3]) * (16-time+1)*(0.01))
        return mae_loss + 0.01*penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            "forbMatrix": self.forbiddens.numpy().tolist(),
            "forbLength": self.lenForbidden,
            "maxLengthSize": self.maxLengthSize
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['forbMatrix'], config['forbLength'], config['maxLengthSize'])

