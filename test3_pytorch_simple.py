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
                 [0.8, 0.6]],[[0.7, 0.9],
                 [0.5, 0.2],
                 [0.6, 0.5]]], dtype=np.float32))  # Shape (3, 2)

# Reshape for broadcasting
A_expanded = torch.unsqueeze(A_sample, 3)  # Shape (batch, 2, 1)
B_expanded = torch.unsqueeze(torch.transpose(B_fixed,0,1), 0)  # Shape (1, 2, 5)

# Subtraction with broadcasting
C = A_expanded - B_expanded

def custom_activation(x):
    a=torch.sigmoid(10*x)
    b=torch.sigmoid(-10*x)
    return a*b

max_pool_1d = torch.nn.MaxPool1d(2,stride=1)
# C1 = custom_activation(C)

# splittedChannels = torch.split(C1,[1,1],dim=2)

# mins=torch.min(C,3).values
# mins = torch.min(torch.abs(C),3).values
mins = torch.min(torch.abs(torch.sum(C,2,keepdim=True)),3).values

# min_channels=torch.minimum(splittedChannels[0],splittedChannels[1])
#
# max_pool_2d = torch.nn.MaxPool2d((1, 5),stride=1)
#
# reshapedPooled=torch.reshape(min_channels,(2,1,5,3))
# poolValue2 = max_pool_2d(min_channels)
#
# # poolValue3 = torch.reshape(poolValue2,(-1,))
# poolValue3 = torch.squeeze(poolValue2)
#
# poolValue4=torch.min(poolValue3, dim=1)

# print("Output shape:", poolValue4)

# temp = torch.nn.CosineSimilarity()
# temp(torch.reshape(A_sample,(A_sample.shape[0],2,A_sample.shape[1])),B_fixed.repeat((2,1,1)).reshape(2,2,5))


print("!!!")

