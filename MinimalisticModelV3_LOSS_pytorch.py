import torch
import torch.nn as nn
import numpy as np


class CustomLoss(nn.Module):
    def __init__(self, forbiddens,lenForbidden,maxLengthSize):
        super(CustomLoss, self).__init__()
        self.forbiddens = forbiddens
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize

    def custom_activation(self,x):
        a = torch.sigmoid(10*x)
        b = torch.sigmoid(-10*x)
        return a * b

    def forward(self, y_true, y_pred, time):
        mae_loss = torch.abs(y_true- y_pred)
        A_expanded = torch.unsqueeze(y_pred, 3)
        B_expanded = torch.unsqueeze(self.forbiddens, 0)
        C = A_expanded - B_expanded
        C1 = self.custom_activation(C)
        penalty = 1 / (0.1 + torch.amax(C1, dim=(1, 2, 3), keepdim=False) * (16 - torch.squeeze(time) + 1) * (0.01))
        return torch.sum(mae_loss) + 0.01 * torch.sum(penalty)

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

# Custom dist Layer
class PairwiseSubtractionLayer(nn.Module):
    def __init__(self, B,lenForbidden,maxLengthSize):
        super(PairwiseSubtractionLayer, self).__init__()
        self.B = B
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize

    def custom_activation(self,x):
        a = torch.sigmoid(10*x)
        b = torch.sigmoid(-10*x)
        return a * b

    def forward(self, A):
        # Reshape for broadcasting
        A_expanded = torch.unsqueeze(A, 3)  # Shape (batch, 2, 1)
        B_expanded = torch.unsqueeze(self.B, 0)
        #B_expanded = torch.unsqueeze(torch.transpose(self.B, 0, 1), 0)  # Shape (1, 2, 5)
        # A_expanded = tf.expand_dims(A, axis=3)  # Shape (batch, 2, 1)
        # B_expanded = tf.expand_dims(self.B, axis=0)  # Shape (1, 2, 5)

        # Subtraction with broadcasting
        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        C1 = self.custom_activation(C)

        splittedChannels = torch.split(C1,[1,1],dim=2)

        min_channels = torch.minimum(splittedChannels[0],splittedChannels[1])
        max_pool_2d = torch.nn.MaxPool2d((1, self.lenForbidden), stride=1)
        #max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, self.lenForbidden),
        #                                        strides=1, padding="valid")
        # reshapedPooled = torch.reshape(min_channels,(-1,1, self.lenForbidden, self.maxLengthSize))
        poolValue2 = max_pool_2d(min_channels)

        poolValue3 = torch.squeeze(poolValue2)

        poolValue4 = torch.min(poolValue3, dim=1, keepdim=True).values
        return poolValue4

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
        self.trajInput = nn.Linear(temporalFeatureSize, 256)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, 256)
        self.timeInput = nn.Linear(1, 4 * 256)
        self.mult1 = Multiply()
        self.conv1 = nn.Conv1d(256, 256, 7, padding=2)
        self.activation1 = nn.Sigmoid()
        self.conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.activation2 = nn.Sigmoid()
        self.conv3 = nn.Conv1d(256, 256, 3, padding=2)
        self.activation3 = nn.Sigmoid()
        # self.activation = Activation()
        self.lastDense = nn.Linear(256, 256)
        self.lastAdd = Add3()
        self.convLast = nn.Conv1d(256, 2, 1)

    def forward(self, traj, time):
        trajPath = self.trajInput(traj)
        trajPathDirect = self.trajInputDirect(traj)
        timePath = self.timeInput(time)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, 256))
        mergePath = self.mult1(trajPath, timePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.conv1(mergePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.activation1(mergePath)

        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.conv2(mergePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.activation2(mergePath)

        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.conv3(mergePath)
        mergePath = mergePath.permute(0, 2, 1)
        mergePath = self.activation3(mergePath)

        mergePath = self.lastDense(mergePath)
        finalPath = self.lastAdd(timePath, mergePath, trajPathDirect)
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