import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from triton.language import dtype
import matplotlib.cm as cm
import math

oneTimeVisGenAB=True

class CustomLoss(nn.Module):
    def __init__(self, forbiddens,lenForbidden,maxLengthSize,timesteps):
        super(CustomLoss, self).__init__()
        self.forbiddens = forbiddens
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize
        self.timesteps=timesteps

    def custom_activation(self,x):
        a = torch.sigmoid(8*x)
        b = torch.sigmoid(-8*x)
        return a * b

    def forward(self, y_pred, y_true, time):
        mae_loss = torch.abs(y_true - y_pred)
        # A_expanded = torch.unsqueeze(y_pred, 3)
        # B_expanded = torch.unsqueeze(self.forbiddens, 0)
        # C = A_expanded - B_expanded
        # # C1 = self.custom_activation(C)
        # # processedTime = torch.relu(((torch.squeeze(time) + 1)/self.timesteps)-0.6)*1
        # processedTime = torch.squeeze(time)
        # processedTime = processedTime*processedTime
        # # penaltyPred = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))
        #
        # minsPred = torch.min(torch.abs(torch.sum(C, 2, keepdim=True)), 3).values
        #
        # penaltyPred = processedTime * (torch.sum(minsPred, dim=(1, 2), keepdim=False))
        #
        # A_expanded = torch.unsqueeze(y_true, 3)
        # # B_expanded = torch.unsqueeze(self.forbiddens, 0)
        # C = A_expanded - B_expanded
        # # C1 = self.custom_activation(C)
        # # processedTime = torch.relu(((torch.squeeze(time) + 1) / self.timesteps) - 0.8) * 1
        # # penaltyTrue = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))
        #
        # minsTrue = torch.min(torch.abs(torch.sum(C, 2, keepdim=True)), 3).values
        #
        # penaltyTrue = processedTime * (torch.sum(minsTrue, dim=(1, 2), keepdim=False))

        # temp=torch.amax(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # temp2 = torch.mean(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # #return torch.max((torch.max(torch.max(mae_loss, dim=1).values, dim=1).values) ) + 0.0 * torch.sum(penalty)
        # test=mae_loss.cpu().detach().numpy()
        # detachedTime=torch.squeeze(time).cpu().detach().numpy()

        #return torch.mean((torch.mean(torch.mean(mae_loss, dim=1), dim=1))) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + torch.mean(mae_loss) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + 0.0 * torch.mean(penalty)
        # return torch.mean(mae_loss) + 0.01 * torch.mean(penalty)
        # return torch.mean(mae_loss) + torch.mean(torch.abs(penaltyTrue-penaltyPred))
        return torch.mean(mae_loss)

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
        # tAdj = torch.unsqueeze(torch.unsqueeze(t,dim=2),dim=3)
        a = torch.sigmoid(10*x)
        b = torch.sigmoid(-10*x)
        # return a * b - torch.nn.ELU()(tAdj)
        # return a * b - (tAdj * x)
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

        # C1 = self.custom_activation(C)
        #
        # splittedChannels = torch.split(C1,[1,1],dim=2)
        #
        # min_channels = torch.minimum(splittedChannels[0],splittedChannels[1])
        # max_pool_2d = torch.nn.MaxPool2d((1, self.lenForbidden), stride=1)
        # #max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, self.lenForbidden),
        # #                                        strides=1, padding="valid")
        # # reshapedPooled = torch.reshape(min_channels,(-1,1, self.lenForbidden, self.maxLengthSize))
        # poolValue2 = max_pool_2d(min_channels)
        #
        # poolValue3 = torch.squeeze(poolValue2)
        #
        # poolValue4 = torch.min(poolValue3, dim=1, keepdim=True).values
        # return poolValue4

        # processedTime = torch.relu(((torch.squeeze(time) + 1) / self.timesteps) - 0.8) * 1
        # # processedTime = torch.squeeze(time)
        # processedTime = processedTime * processedTime

        mins = torch.min(torch.abs(torch.sum(C, 2, keepdim=True)), 3).values

        penalty = torch.unsqueeze(torch.sum(mins, dim=(1, 2), keepdim=False),1)
        return penalty

# Custom Multiply Layer
class Multiply(nn.Module):
    def forward(self, x1, x2):
        return x1 * x2  # Element-wise multiplication

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2):
        super(SimpleNN, self).__init__()
        self.size = 64
        self.maxLengthSize = maxLengthSize
        self.temporalFeatureSize = temporalFeatureSize
        self.multPrePsl = Multiply()

        self.timeForbMixedLinear = nn.Linear(2, self.size)
        self.multTrajCondForbTime = Multiply()
        self.normBeforeConv = nn.LayerNorm([maxLengthSize+1,self.size])
        # self.normInConv = nn.LayerNorm([self.size, self.size])
        self.normInConv = nn.BatchNorm1d(self.size)

        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.normDists = nn.LayerNorm([1])
        self.pslDense = nn.Linear(1+2, self.maxLengthSize*10)
        #self.trajCondConcat=
        self.trajInput = nn.Linear(temporalFeatureSize, self.size)
        #self.conditionInput = nn.Linear(2, 16)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, self.size)
        self.timeInput = nn.Linear(1+2, maxLengthSize * self.size)
        self.multAllConvTime = Multiply()
        self.conv1d1 = nn.Conv1d(maxLengthSize+1, self.size, 11, padding=5)
        self.activation = nn.ReLU()
        self.activationFinal = nn.Tanh()
        self.normFinal1 = nn.LayerNorm([maxLengthSize, self.size])
        self.normFinal2 = nn.LayerNorm([maxLengthSize, 256])

        self.conv2d1 = nn.Conv1d(self.size, self.size, 9, padding=4)
        self.conv3d1 = nn.Conv1d(self.size, self.size, 7, padding=3)

        #self.auxDense = nn.Linear(510, 512)

        self.conv1d2 = nn.Conv1d(maxLengthSize+1, self.size, 11, padding=10, dilation=2)
        self.conv2d2 = nn.Conv1d(self.size, self.size, 9, padding=8, dilation=2)
        self.conv3d2 = nn.Conv1d(self.size, self.size, 7, padding=6, dilation=2)

        self.conv1d3 = nn.Conv1d(maxLengthSize+1, self.size, 11, padding=15, dilation=3)
        self.conv2d3 = nn.Conv1d(self.size, self.size, 9, padding=12, dilation=3)
        self.conv3d3 = nn.Conv1d(self.size, self.size, 7, padding=9, dilation=3)

        # self.activation = Activation()
        self.lastDenseInPathAdjust = nn.Linear(self.size, maxLengthSize)
        self.lastDenseInPath = nn.Linear(self.size, self.size)
        self.denseAfterCat3 = nn.Linear(self.size*3, self.size)
        self.lastAdd = Add3()
        self.lastDense = nn.Linear(self.size + 10, 256)
        self.convLast = nn.Conv1d(256, 2, 1)

    def forward(self, traj, time, condLat, condLon):
        plsPath = self.psl(traj)
        plsPath = self.multPrePsl(plsPath, time)
        plsPathNorm = self.normDists(plsPath)

        plsPathUnsqueeze = torch.unsqueeze(plsPathNorm,1)
        timeUnsqueeze = torch.unsqueeze(time,1)
        timeForbsMixed = torch.cat((plsPathUnsqueeze, timeUnsqueeze), dim=2)
        timeForbsMixedLinearOut = self.timeForbMixedLinear(timeForbsMixed)



        mixedPlsPathCond = torch.cat((plsPathNorm, condLat, condLon), 1)
        plsPathD = self.pslDense(mixedPlsPathCond)
        plsPathDRS = torch.reshape(plsPathD, (-1, self.maxLengthSize, (int)(10)))


        condLatLon=torch.cat((torch.unsqueeze(condLat,1),torch.unsqueeze(condLon,1)),dim=2)
        mixedTrajCond=torch.cat((condLatLon,traj), 1)
        trajPath = self.trajInput(mixedTrajCond)

        trajPathAfterMultForbTime = self.multTrajCondForbTime(timeForbsMixedLinearOut,trajPath)
        trajPathAfterMultForbTimeNorm = self.normBeforeConv(trajPathAfterMultForbTime)

        trajPathDirect = self.trajInputDirect(traj)
        mixedTimeCond = torch.cat((time, condLat, condLon), 1)
        timePath = self.timeInput(mixedTimeCond)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, self.size))
        #mergePathRoot = self.mult1(trajPath, timePath)

        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv1d1(trajPathAfterMultForbTimeNorm)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.normInConv(mergePath1)
        mergePath1 = self.activation(mergePath1)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv2d1(mergePath1)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.normInConv(mergePath1)
        mergePath1 = self.activation(mergePath1)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv3d1(mergePath1)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.normInConv(mergePath1)
        mergePath1 = self.activation(mergePath1)





        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv1d2(trajPathAfterMultForbTimeNorm)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.normInConv(mergePath2)
        mergePath2 = self.activation(mergePath2)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv2d2(mergePath2)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.normInConv(mergePath2)
        mergePath2 = self.activation(mergePath2)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv3d2(mergePath2)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.normInConv(mergePath2)
        mergePath2 = self.activation(mergePath2)





        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv1d3(trajPathAfterMultForbTimeNorm)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.normInConv(mergePath3)
        mergePath3 = self.activation(mergePath3)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv2d3(mergePath3)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.normInConv(mergePath3)
        mergePath3 = self.activation(mergePath3)


        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv3d3(mergePath3)
        #mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.normInConv(mergePath3)
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
        mergePath3Total = self.multAllConvTime(mergePath3Total, timePath)
        mergePath3Total = self.activation(mergePath3Total)

        #finalPath = self.lastAdd(timePath, mergePath3Total, trajPathDirect)
        finalPath = torch.cat((timePath, mergePath3Total, trajPathDirect), 2)
        finalPath = self.denseAfterCat3(finalPath)
        finalPath = self.normFinal1(finalPath)
        finalPath = self.activationFinal(finalPath)

        finalPathC = torch.cat((finalPath, plsPathDRS), 2)
        finalPathC = self.lastDense(finalPathC)
        finalPathC = self.normFinal2(finalPathC)
        finalPathC = self.activationFinal(finalPathC)

        finalPathC = finalPathC.permute(0, 2, 1)
        finalPathC = self.convLast(finalPathC)
        finalPathC = finalPathC.permute(0, 2, 1)
        return finalPathC

def generate_ts(timesteps, num):
    orig = np.random.randint(0, timesteps, size=num)
    lastOnes = np.random.randint((int)(math.floor(timesteps*0.8)), timesteps, size=num)
    finalTs = np.concat((orig,lastOnes))
    finalTs = np.random.choice(finalTs,num)
    return finalTs
    # return np.arange(0,timesteps,1,dtype=int)
    # return np.random.randint(timesteps-1, timesteps, size=num)
    # return np.random.randint(timesteps-3, timesteps, size=num)

def forward_noise(timesteps, x, t):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    # noise = np.random.normal(loc=0.5,scale=0.33,size=x.shape)  # noise mask
    noise = np.random.uniform(low=0,high=1,size=x.shape)

    # for i in range(1, noise.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([noise[i, h - 1, 0], noise[i, h, 0]], [noise[i, h - 1, 1], noise[i, h, 1]], marker='', zorder=2, alpha=0.5,color='c')


    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    img_a = x * (1 - np.pow(a,2)) + noise * np.pow(a,3)
    img_b = x * (1 - np.pow(b,2)) + noise * np.pow(b,3)

    # img_a = x + noise * np.pow(a,2)
    # img_b = x + noise * np.pow(b,2)

    # idx=237
    # for h in range(1, 12):
    #     plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]], marker='', zorder=2, alpha=0.5,color='r')
    # for h in range(1, 12):
    #     plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]], marker='', zorder=2, alpha=0.5,color='g')
    # plt.show()

    # for idx in range(x.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
    #                  zorder=2, alpha=0.5, color='r')
    # # # plt.show()
    # for idx in range(x.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]], marker='',
    #                  zorder=2, alpha=0.5, color='g')
    # plt.show()

    if oneTimeVisGenAB==True:
        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, timesteps)
        for idx in range(x.shape[0]):
            color = cmap(t[idx])
            for h in range(1, 12):
                plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]], marker='',
                         zorder=2, alpha=0.5, color=color)
        plt.show()
        for idx in range(x.shape[0]):
            color = cmap(t[idx])
            for h in range(1, 12):
                plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]], marker='',
                         zorder=2, alpha=0.5, color=color)
        plt.show()
        oneTimeVisGenAB=False

    return img_a, img_b