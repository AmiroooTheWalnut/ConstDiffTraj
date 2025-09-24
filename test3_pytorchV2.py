import numpy as np
import torch

B_fixed = torch.from_numpy(np.array([
    [0.1, 0.3],
    [0.5, 0.2],
    [0.3, 0.4],
    [0.7, 0.9],
    [0.6, 0.5]
], dtype=np.float32))

# Test with sample data
# A_sample = tf.random.normal((4, 2))  # Batch size 4

A_sample = torch.from_numpy(np.array([[[0.5, 0.5],
                 [0.3, 0.4],
                 [0.8, 0.6]],[[0.8, 0.9],
                 [0.5, 0.4],
                 [0.9, 0.6]]], dtype=np.float32))  # Shape (3, 2)

def custom_activation(x):
    a=torch.sigmoid(10*x)
    b=torch.sigmoid(-10*x)
    return a*b


A_expanded = torch.unsqueeze(A_sample, 3)  # Shape (batch, 2, 1)
B_expanded = torch.unsqueeze(B_fixed, 0)
        #B_expanded = torch.unsqueeze(torch.transpose(self.B, 0, 1), 0)  # Shape (1, 2, 5)
        # A_expanded = tf.expand_dims(A, axis=3)  # Shape (batch, 2, 1)
        # B_expanded = tf.expand_dims(self.B, axis=0)  # Shape (1, 2, 5)

        # Subtraction with broadcasting
C = A_expanded - B_expanded  # Shape (batch, 2, 5)

C1 = custom_activation(C)

splittedChannels = torch.split(C1,[1,1],dim=2)

min_channels = torch.minimum(splittedChannels[0],splittedChannels[1])
max_pool_2d = torch.nn.MaxPool2d((1, 5), stride=1)
        #max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, self.lenForbidden),
        #                                        strides=1, padding="valid")
        # reshapedPooled = torch.reshape(min_channels,(-1,1, self.lenForbidden, self.maxLengthSize))
poolValue2 = max_pool_2d(min_channels)

poolValue3 = torch.squeeze(poolValue2)

poolValue4 = torch.min(poolValue3, dim=1, keepdim=True).values

