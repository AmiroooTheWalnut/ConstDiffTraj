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
│ dense (Dense)       │ (None, 256)       │        512 │ x_ts_input[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 4, 64)     │        192 │ x_input[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ reshape (Reshape)   │ (None, 4, 64)     │          0 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ reshape_1 (Reshape) │ (None, 4, 64)     │          0 │ dense_1[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multiply (Multiply) │ (None, 4, 64)     │          0 │ reshape[0][0],    │
│                     │                   │            │ reshape_1[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d (Conv1D)     │ (None, 4, 128)    │     41,088 │ multiply[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (None, 4, 128)    │          0 │ conv1d[0][0]      │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (Dense)     │ (None, 4, 64)     │      8,256 │ activation[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (Dense)     │ (None, 4, 64)     │        192 │ x_input[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (Add)           │ (None, 4, 64)     │          0 │ dense_2[0][0],    │
│                     │                   │            │ reshape[0][0],    │
│                     │                   │            │ dense_3[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_1 (Conv1D)   │ (None, 4, 2)      │        130 │ add[0][0]         │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
'''


import tensorflow as tf
from tensorflow.keras import layers
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

    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

def make_model(forbiddenSerialMap=None, lenForbidden=10,maxLengthSize = 10, temporalFeatureSize=2):
    x = x_input = layers.Input(shape=(maxLengthSize, temporalFeatureSize), name='x_input')

    x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')

    # x_im = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    # x_im = layers.Activation('relu')(x_im)

    time_parameter = layers.Dense(maxLengthSize * 64)(x_ts)
    im_parameter = layers.Dense(maxLengthSize * 16)(x_input)
    reshapedTime = layers.Reshape((maxLengthSize, 64))(time_parameter)
    reshapedIm = layers.Reshape((maxLengthSize, 64))(im_parameter)

    timed_img = layers.Multiply()([reshapedTime,reshapedIm])

    x_parameter = layers.Conv1D(128, kernel_size=5, padding='same')(timed_img)
    x_parameter = layers.Activation('relu')(x_parameter)

    x_parameter = layers.Dense(64)(x_parameter)

    # flatImage = layers.Flatten(data_format='channels_last')(x_parameter)

    x_direct = layers.Dense(64)(x_input)

    concatenatedLayer = layers.Add()([x_parameter,reshapedTime,x_direct])

    x = layers.Conv1D(2, kernel_size=1, padding='same')(concatenatedLayer)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model