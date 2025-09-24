import numpy as np
import torch

B_fixed = torch.from_numpy(np.array([
    [0.1, 0.3],
    [0.5, 0.2],
    [0.3, 0.4],
    [0.7, 0.9],
    [0.6, 0.5]
], dtype=np.float32))

mask = torch.from_numpy(np.array([[
        [1,1],
        [1,1],
        [1,1],
        [0,0]],
    [
        [1,1],
        [1,1],
        [1,1],
        [1,1]]
],dtype=np.float32))

# Test with sample data
# A_sample = tf.random.normal((4, 2))  # Batch size 4

A_sample = torch.from_numpy(np.array([[[0.5, 0.5],
                 [0.3, 0.4],
                 [0.8, 0.6],
                 [10.0, 10.0]],[[0.7, 0.9],
                 [0.5, 0.2],
                 [0.6, 0.5],
                 [0.5, 0.5]]], dtype=np.float32))  # Shape (3, 2)

# Reshape for broadcasting
A_expanded = torch.unsqueeze(A_sample, 3)  # Shape (batch, 2, 1)
B_expanded = torch.unsqueeze(torch.transpose(B_fixed,0,1), 0)  # Shape (1, 2, 5)

# Subtraction with broadcasting
C = A_expanded - B_expanded

def custom_activation(x):
    a = torch.sigmoid(x)
    b = torch.sigmoid(-x)
    d = torch.sign(x)

    d[d == 0] = 1

    #OR
    e = torch.sign(x) + torch.tanh(x)


    return a * b * d

C = A_expanded - B_expanded  # Shape (batch, 2, 5)

C = C*mask.unsqueeze(dim=3).repeat(1,1,1,5)

# C1 = torch.tanh(C)
# C1 = custom_activation(C)

finIndex = torch.argmin(torch.max(torch.abs(C),dim=2,keepdim=True).values, dim=3)

finIndex = finIndex.repeat(repeats=(1,1,2))

valsMin = torch.gather(C,dim=3,index=finIndex.unsqueeze(dim=3))

finFin = custom_activation(valsMin.squeeze())
finFin2 = valsMin.squeeze()
print("!!!")

