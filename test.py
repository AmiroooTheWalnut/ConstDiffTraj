import tensorflow as tf
from tensorflow import keras
import numpy as np


class PairwiseSubtractionLayer(tf.keras.layers.Layer):
    def __init__(self, B, **kwargs):
        super(PairwiseSubtractionLayer, self).__init__(**kwargs)
        self.B = tf.convert_to_tensor(B, dtype=tf.float32)  # Store B as a constant tensor

    def custom_activation(self,x):
        a = tf.sigmoid(x)
        b = tf.sigmoid(-x)
        return a * b

    def call(self, A):
        # Reshape for broadcasting
        A_expanded = tf.expand_dims(A, axis=2)  # Shape (batch, 2, 1)
        B_expanded = tf.expand_dims(tf.transpose(self.B), axis=0)  # Shape (1, 2, 5)

        # Subtraction with broadcasting
        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,
                                                strides=1, padding="valid")
        C1 = self.custom_activation(C)

        splittedChannels = tf.split(C1, [1, 1], axis=1)

        min_channels = tf.keras.layers.minimum([splittedChannels[0], splittedChannels[1]])

        max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, 5),
                                                strides=1, padding="valid")
        x = tf.expand_dims(min_channels, axis=-1)
        poolValue2 = max_pool_2d(x)

        poolValue3 = tf.keras.layers.Reshape((-1,))(poolValue2)

        return poolValue3


# Example fixed matrix B
B_fixed = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
], dtype=tf.float32)

# Example usage
A_input = tf.keras.Input(shape=(2,))  # Shape (batch, 2)

# Custom layer with fixed B
C_output = PairwiseSubtractionLayer(B_fixed)(A_input)

# Build the model
model = tf.keras.Model(inputs=A_input, outputs=C_output)

# Test with sample data
# A_sample = tf.random.normal((4, 2))  # Batch size 4

A_sample = tf.constant([[9, 10],
                 [3, 4],
                 [4, 6]], dtype=tf.float32)  # Shape (3, 2)

result = model(A_sample)

print("Output shape:", result.shape)  # Should be (4, 2, 5)
print("!!!")