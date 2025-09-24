import torch
import torch.nn as nn
import numpy as np


class CustomLoss(nn.Module):
    def __init__(self, forbiddens,lenForbidden,maxLengthSize,timesteps):
        super(CustomLoss, self).__init__()
        self.forbiddens = forbiddens
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize
        self.timesteps=timesteps

    def custom_activation(self,x):
        a = torch.sigmoid(6*x)
        b = torch.sigmoid(-6*x)
        return a * b

    def forward(self, y_true, y_pred, time):
        mae_loss = torch.abs(y_true - y_pred)
        A_expanded = torch.unsqueeze(y_pred, 3)
        B_expanded = torch.unsqueeze(self.forbiddens, 0)
        C = A_expanded - B_expanded
        C1 = self.custom_activation(C)
        processedTime = torch.relu(((torch.squeeze(time) + 1)/self.timesteps)-0.8)*1
        penalty = processedTime / (0.1 + torch.mean(C1, dim=(1, 2, 3), keepdim=False))
        temp=torch.amax(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        temp2 = torch.mean(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        #return torch.max((torch.max(torch.max(mae_loss, dim=1).values, dim=1).values) ) + 0.0 * torch.sum(penalty)
        test=mae_loss.cpu().detach().numpy()
        detachedTime=torch.squeeze(time).cpu().detach().numpy()
        #return torch.mean((torch.mean(torch.mean(mae_loss, dim=1), dim=1))) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + torch.mean(mae_loss) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + 0.0 * torch.mean(penalty)
        return torch.mean(mae_loss) + 0.001 * torch.mean(penalty)

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
        a = torch.sigmoid(6*x)
        b = torch.sigmoid(-6*x)
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

        self.timeForbMixedLinear = nn.Linear(2, 64)
        self.multTrajCondForbTime = Multiply()

        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.pslDense = nn.Linear(1+2, self.maxLengthSize*10)
        #self.trajCondConcat=
        self.trajInput = nn.Linear(temporalFeatureSize, 64)
        #self.conditionInput = nn.Linear(2, 16)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, 64)
        self.timeInput = nn.Linear(1+2, maxLengthSize * 64)
        self.mult1 = Multiply()
        self.conv1d1 = nn.Conv1d(maxLengthSize+1, 64, 11, padding=5)
        self.activation = nn.SiLU()
        self.activationFinal = nn.Tanh()
        self.conv2d1 = nn.Conv1d(64, 64, 9, padding=4)
        self.conv3d1 = nn.Conv1d(64, 64, 7, padding=3)

        #self.auxDense = nn.Linear(510, 512)

        self.conv1d2 = nn.Conv1d(maxLengthSize+1, 64, 11, padding=10, dilation=2)
        self.conv2d2 = nn.Conv1d(64, 64, 9, padding=8, dilation=2)
        self.conv3d2 = nn.Conv1d(64, 64, 7, padding=6, dilation=2)

        self.conv1d3 = nn.Conv1d(maxLengthSize+1, 64, 11, padding=15, dilation=3)
        self.conv2d3 = nn.Conv1d(64, 64, 9, padding=12, dilation=3)
        self.conv3d3 = nn.Conv1d(64, 64, 7, padding=9, dilation=3)

        # self.activation = Activation()
        self.lastDenseInPathAdjust = nn.Linear(64, maxLengthSize)
        self.lastDenseInPath = nn.Linear(64, 64)
        self.denseAfterCat3 = nn.Linear(64*3, 64)
        self.lastAdd = Add3()
        self.lastDense = nn.Linear(64 + 10, 256)
        self.convLast = nn.Conv1d(256, 2, 1)

    def forward(self, traj, time, condLat, condLon):
        plsPath = self.psl(traj)

        plsPathUnsqueeze = torch.unsqueeze(plsPath,1)
        timeUnsqueeze = torch.unsqueeze(time,1)
        timeForbsMixed = torch.cat((plsPathUnsqueeze, timeUnsqueeze), dim=2)
        timeForbsMixedLinearOut = self.timeForbMixedLinear(timeForbsMixed)


        plsPath1 = self.multPrePsl(plsPath,time)
        mixedPlsPathCond = torch.cat((plsPath1, condLat, condLon), 1)
        plsPathD = self.pslDense(mixedPlsPathCond)
        plsPathDRS = torch.reshape(plsPathD, (-1, self.maxLengthSize, (int)(10)))


        condLatLon=torch.cat((torch.unsqueeze(condLat,1),torch.unsqueeze(condLon,1)),dim=2)
        mixedTrajCond=torch.cat((condLatLon,traj), 1)
        trajPath = self.trajInput(mixedTrajCond)

        trajPathAfterMultForbTime = self.multTrajCondForbTime(timeForbsMixedLinearOut,trajPath)

        trajPathDirect = self.trajInputDirect(traj)
        mixedTimeCond = torch.cat((time, condLat, condLon), 1)
        timePath = self.timeInput(mixedTimeCond)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, 64))
        #mergePathRoot = self.mult1(trajPath, timePath)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv1d1(trajPathAfterMultForbTime)
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
        mergePath2 = self.conv1d2(trajPathAfterMultForbTime)
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
        mergePath3 = self.conv1d3(trajPathAfterMultForbTime)
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

        mergePath1 = self.lastDenseInPathAdjust(mergePath1)
        mergePath2 = self.lastDenseInPathAdjust(mergePath2)
        mergePath3 = self.lastDenseInPathAdjust(mergePath3)

        mergePath1 = torch.reshape(mergePath1, (traj.shape[0], self.maxLengthSize, -1))
        mergePath2 = torch.reshape(mergePath2, (traj.shape[0], self.maxLengthSize, -1))
        mergePath3 = torch.reshape(mergePath3, (traj.shape[0], self.maxLengthSize, -1))


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
        finalPath = self.activationFinal(finalPath)

        finalPathC = torch.cat((finalPath, plsPathDRS), 2)
        finalPathC = self.lastDense(finalPathC)
        finalPathC = self.activationFinal(finalPathC)

        finalPathC = finalPathC.permute(0, 2, 1)
        finalPathC = self.convLast(finalPathC)
        finalPathC = finalPathC.permute(0, 2, 1)
        return finalPathC

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