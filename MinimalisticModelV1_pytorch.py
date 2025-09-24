import torch
import torch.nn as nn
import numpy as np

# # Custom Activation Layer
# class Activation(nn.Module):
#     def forward(self, x1):
#         return nn.ReLU(x1)  # Element-wise addition

# Custom Add2 Layer
class Add2(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2  # Element-wise addition

# Custom Add3 Layer
class Add3(nn.Module):
    def forward(self, x1, x2, x3):
        return x1 + x2 +x3  # Element-wise addition

# Custom Multiply Layer
class Multiply(nn.Module):
    def forward(self, x1, x2):
        return x1 * x2  # Element-wise multiplication

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2):
        super(SimpleNN, self).__init__()
        self.maxLengthSize = maxLengthSize
        self.temporalFeatureSize = temporalFeatureSize
        self.trajInput = nn.Linear(temporalFeatureSize, 64)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, 64)
        self.timeInput = nn.Linear(1, 4*64)
        self.mult1 = Multiply()
        self.conv1 = nn.Conv1d(64,128, 5,padding=2)
        self.activation = nn.ReLU()
        # self.activation = Activation()
        self.lastDense = nn.Linear(128, 64)
        self.lastAdd = Add3()
        self.convLast = nn.Conv1d(64, 2, 1)

    def forward(self, traj, time):
        trajPath = self.trajInput(traj)
        trajPathDirect = self.trajInputDirect(traj)
        timePath = self.timeInput(time)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, 64))
        mergePath = self.mult1(trajPath,timePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.conv1(mergePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.activation(mergePath)
        mergePath = self.lastDense(mergePath)
        finalPath = self.lastAdd(timePath,mergePath,trajPathDirect)
        finalPath = finalPath.permute(0, 2, 1)
        finalPath = self.convLast(finalPath)
        finalPath = finalPath.permute(0, 2, 1)
        return finalPath

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