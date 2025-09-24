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
        self.multPrePsl = Multiply()
        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.pslDense = nn.Linear(1, 512)
        self.trajInput = nn.Linear(temporalFeatureSize, 64)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, 64)
        self.timeInput = nn.Linear(1, maxLengthSize * 64)
        self.mult1 = Multiply()
        self.conv1d1 = nn.Conv1d(maxLengthSize, 128, 7, padding=3)
        self.activation = nn.Sigmoid()
        self.conv2d1 = nn.Conv1d(128, 128, 5, padding=2)
        self.conv3d1 = nn.Conv1d(128, 128, 3, padding=1)

        #self.auxDense = nn.Linear(510, 512)

        self.conv1d2 = nn.Conv1d(maxLengthSize, 128, 7, padding=6, dilation=2)
        self.conv2d2 = nn.Conv1d(128, 128, 5, padding=4, dilation=2)
        self.conv3d2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)

        self.conv1d3 = nn.Conv1d(maxLengthSize, 128, 7, padding=9, dilation=3)
        self.conv2d3 = nn.Conv1d(128, 128, 5, padding=6, dilation=3)
        self.conv3d3 = nn.Conv1d(128, 128, 3, padding=3, dilation=3)

        # self.activation = Activation()
        self.lastDenseInPath = nn.Linear(2048, 64)
        self.denseAfterCat3 = nn.Linear(64*3, 64)
        self.lastAdd = Add3()
        self.lastDense = nn.Linear(128 + 64, 512)
        self.convLast = nn.Conv1d(512, 2, 1)

    def forward(self, traj, time):
        plsPath = self.psl(traj)
        plsPath = self.multPrePsl(plsPath,time)
        plsPathD = self.pslDense(plsPath)
        plsPathDRS = torch.reshape(plsPathD, (-1, self.maxLengthSize, (int)(512 / self.maxLengthSize)))

        trajPath = self.trajInput(traj)
        trajPathDirect = self.trajInputDirect(traj)
        timePath = self.timeInput(time)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, 64))
        #mergePathRoot = self.mult1(trajPath, timePath)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv1d1(trajPath)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv2d1(mergePath1)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv3d1(mergePath1)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)




        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv1d2(trajPath)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv2d2(mergePath2)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv3d2(mergePath2)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)




        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv1d3(trajPath)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv2d3(mergePath3)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv3d3(mergePath3)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)


        mergePath1 = torch.reshape(mergePath1, (-1, self.maxLengthSize, 128 * 16))
        mergePath2 = torch.reshape(mergePath2, (-1, self.maxLengthSize, 128 * 16))
        mergePath3 = torch.reshape(mergePath3, (-1, self.maxLengthSize, 128 * 16))


        mergePath1 = self.lastDenseInPath(mergePath1)
        mergePath2 = self.lastDenseInPath(mergePath2)
        mergePath3 = self.lastDenseInPath(mergePath3)

        mergePath3Total = torch.cat((mergePath1, mergePath2, mergePath3), 2)
        mergePath3Total = self.denseAfterCat3(mergePath3Total)
        mergePath3Total = self.mult1(mergePath3Total, timePath)
        mergePath3Total = self.activation(mergePath3Total)

        #finalPath = self.lastAdd(timePath, mergePath3Total, trajPathDirect)
        finalPath = torch.cat((timePath, mergePath3Total, trajPathDirect), 2)
        finalPath = self.denseAfterCat3(finalPath)
        finalPath = self.activation(finalPath)

        finalPath = torch.cat((finalPath, plsPathDRS), 2)
        finalPath = self.lastDense(finalPath)
        finalPath = self.activation(finalPath)

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