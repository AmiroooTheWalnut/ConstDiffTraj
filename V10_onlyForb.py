import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from triton.language import dtype
import matplotlib.cm as cm
import math

oneTimeVisGenAB=True

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

class UNet1D(nn.Module):
    def __init__(self, input_channels=2, base_channels=64, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, base_channels, kernel_size=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.res_skips = []

        ch = base_channels
        for i in range(num_blocks):
            self.downs.append(nn.Sequential(
                ResidualBlock1D(ch),
                nn.Conv1d(ch, ch * 2, kernel_size=2, stride=2),  # Downsample
            ))
            ch *= 2

        for i in range(num_blocks):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=2, stride=2),  # Upsample
                ResidualBlock1D(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Conv1d(base_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # Input: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips = skips[::-1]
        for i, up in enumerate(self.ups):
            x = up(x)
            # Crop if shape mismatch due to odd-length sequences
            if x.shape[-1] > skips[i].shape[-1]:
                x = x[..., :skips[i].shape[-1]]
            elif x.shape[-1] < skips[i].shape[-1]:
                skips[i] = skips[i][..., :x.shape[-1]]
            x = x + skips[i]  # Residual skip connection

        x = self.output_proj(x)
        return x.permute(0, 2, 1)  # Back to (B, T, F)

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
        A_expanded = torch.unsqueeze(y_pred, 3)
        B_expanded = torch.unsqueeze(self.forbiddens, 0)
        C = A_expanded - B_expanded
        # C1 = self.custom_activation(C)
        processedTime = torch.relu(((torch.squeeze(time) + 1)/self.timesteps)-0.6)*1
        # processedTime = torch.squeeze(time)
        processedTime = processedTime*processedTime
        # penaltyPred = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))

        minsPred = torch.min(torch.abs(torch.sum(C, 2, keepdim=True)), 3).values

        penaltyPred = processedTime * (torch.mean(minsPred, dim=(1, 2), keepdim=False))

        A_expanded = torch.unsqueeze(y_true, 3)
        # B_expanded = torch.unsqueeze(self.forbiddens, 0)
        C = A_expanded - B_expanded
        # C1 = self.custom_activation(C)
        # processedTime = torch.relu(((torch.squeeze(time) + 1) / self.timesteps) - 0.8) * 1
        # penaltyTrue = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))

        minsTrue = torch.min(torch.sum(torch.abs(C), 2, keepdim=True), 3).values

        penaltyTrue = processedTime * (torch.mean(minsTrue, dim=(1, 2), keepdim=False))

        # temp=torch.amax(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # temp2 = torch.mean(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # #return torch.max((torch.max(torch.max(mae_loss, dim=1).values, dim=1).values) ) + 0.0 * torch.sum(penalty)
        # test=mae_loss.cpu().detach().numpy()
        # detachedTime=torch.squeeze(time).cpu().detach().numpy()

        #return torch.mean((torch.mean(torch.mean(mae_loss, dim=1), dim=1))) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + torch.mean(mae_loss) + 0.001 * torch.sum(penalty)
        #return torch.mean(torch.pow(mae_loss,2)) + 0.0 * torch.mean(penalty)
        # return torch.mean(mae_loss) + 0.01 * torch.mean(penalty)
        return torch.mean(mae_loss) + torch.mean(torch.abs(penaltyTrue-penaltyPred))
        # return torch.mean(mae_loss)

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
        a = torch.sigmoid(6 * x)
        b = torch.sigmoid(-6 * x)
        d = torch.sign(x)

        d[d == 0] = 1
        return a * b * d

    def forward(self, A):
        # Reshape for broadcasting
        A_expanded = torch.unsqueeze(A, 3)  # Shape (batch, 2, 1)
        B_expanded = torch.unsqueeze(self.B, 0)
        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        # C1 = torch.tanh(C)
        # C1 = custom_activation(C)

        finIndex = torch.argmin(torch.max(torch.abs(C), dim=2, keepdim=True).values, dim=3)

        finIndex = finIndex.repeat(repeats=(1, 1, 2))

        valsMin = torch.gather(C, dim=3, index=finIndex.unsqueeze(dim=3))

        # finFin = self.custom_activation(valsMin.squeeze())
        finFin = valsMin.squeeze()
        return finFin

# Custom dist Layer
class PairwiseSubtractionSumLayer(nn.Module):
    def __init__(self, B,lenForbidden,maxLengthSize):
        super(PairwiseSubtractionSumLayer, self).__init__()
        self.B = B
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize

    def custom_activation(self,x):
        # tAdj = torch.unsqueeze(torch.unsqueeze(t,dim=2),dim=3)
        a = torch.sigmoid(6*x)
        b = torch.sigmoid(-6*x)
        # return a * b - torch.nn.ELU()(tAdj)
        # return a * b - (tAdj * x)
        return a * b

    def forward(self, A):
        # Reshape for broadcasting
        A_expanded = torch.unsqueeze(A, 3)  # Shape (batch, 2, 1)
        B_expanded = torch.unsqueeze(self.B, 0)

        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        C1 = self.custom_activation(C)

        # return torch.min(C1, dim=3).values

        splittedChannels = torch.split(C1, [1, 1], dim=2)

        min_channels = torch.minimum(splittedChannels[0], splittedChannels[1])
        max_pool_2d = torch.nn.MaxPool2d((1, self.lenForbidden), stride=1)

        poolValue2 = max_pool_2d(min_channels)

        poolValue3 = torch.squeeze(poolValue2)

        # poolValue4 = torch.min(poolValue3, dim=1, keepdim=True).values
        poolValue4 = torch.sum(poolValue3, dim=1, keepdim=True)

        return poolValue4

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

        self.timeDense = nn.Linear(1, self.size)
        self.forbDense = nn.Linear(2, self.size)
        self.timeForbMixedLinear = nn.Linear(2, self.size)
        self.multTrajCondForbTime = Multiply()

        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        pslSum = PairwiseSubtractionSumLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.pslSum = pslSum
        self.pslDense = nn.Linear(1 + 2, self.maxLengthSize * 10)
        # self.trajCondConcat=
        self.trajInput = nn.Linear(temporalFeatureSize, self.size)
        # self.conditionInput = nn.Linear(2, 16)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, self.size)
        self.timeInput = nn.Linear(1 + 2, maxLengthSize * self.size)
        self.mult1 = Multiply()

        # self.ConvDirect = ConvLayers(maxLengthSize,self.size,1)
        self.ConvDists = ConvLayers(maxLengthSize,self.size,0)

        self.allInfoDense = nn.Linear(self.size*2, self.size)

        # self.activation = nn.SiLU()
        # self.activationFinal = nn.Tanh()


        # self.denseAfterCat3 = nn.Linear(self.size * 3, self.size)
        self.denseAfterCat4Traj = nn.Linear(self.size * 4, self.size)
        self.denseAfterCat4Forb = nn.Linear(self.size * 4, self.size)
        self.denseAfterCat5 = nn.Linear(self.size * 4, self.size)
        self.lastAdd = Add3()
        self.lastDense = nn.Linear(self.size + 10, 32)
        self.convLast = nn.Conv1d(32, 2, 1)

    def forward(self, traj, time, condLat, condLon):
        plsPath = self.psl(traj)
        plsPathSum = self.pslSum(traj)

        # plsPathUnsqueeze = torch.unsqueeze(plsPath, 1)
        # timeUnsqueeze = torch.unsqueeze(time, 1)
        # timeForbsMixed = torch.cat((plsPath, timeUnsqueeze), dim=2)
        forbDensed = self.forbDense(plsPath)
        # timeDensed = self.timeDense(timeUnsqueeze)
        # timeDensed = self.activationFinal(timeDensed)


        plsPath1 = self.multPrePsl(plsPathSum, time)
        mixedPlsPathCond = torch.cat((plsPath1, condLat, condLon), 1)
        plsPathD = self.pslDense(mixedPlsPathCond)
        plsPathDRS = torch.reshape(plsPathD, (-1, self.maxLengthSize, (int)(10)))

        condLatLon = torch.cat((torch.unsqueeze(condLat, 1), torch.unsqueeze(condLon, 1)), dim=2)
        mixedTrajCond = torch.cat((condLatLon, traj), 1)
        trajPath = self.trajInput(mixedTrajCond)

        # trajPathAfterMultForbTime = self.multTrajCondForbTime(timeForbsMixedLinearOut, trajPath)
        # trajPathAfterCatForbTime = torch.cat((forbDensed,trajPath),dim=1)

        trajPathDirect = self.trajInputDirect(traj)
        mixedTimeCond = torch.cat((time, condLat, condLon), 1)
        timePath = self.timeInput(mixedTimeCond)
        timePath = torch.reshape(timePath, (-1, self.maxLengthSize, self.size))
        # mergePathRoot = self.mult1(trajPath, timePath)

        # convedTrajs = self.ConvDirect(trajPath)
        convedDists = self.ConvDists(forbDensed)

        # mergePathTraj = self.denseAfterCat4Traj(convedTrajs)
        mergePathForb = self.denseAfterCat4Forb(convedDists)

        # allInfo = torch.cat((mergePathTraj,mergePathForb,timePath),dim=2)
        allInfo = torch.cat((mergePathForb, timePath), dim=2)
        allInfoDrnsed = self.allInfoDense(allInfo)
        # allInfoDrnsed = self.activationFinal(allInfoDrnsed)

        # finalPath = self.lastAdd(timePath, mergePath3Total, trajPathDirect)
        # finalPath = torch.cat((timePath, allInfoDrnsed, trajPathDirect,mergePathTraj,mergePathForb), 2)
        finalPath = torch.cat((timePath, allInfoDrnsed, trajPathDirect, mergePathForb), 2)
        finalPath = self.denseAfterCat5(finalPath)
        # finalPath = self.activationFinal(finalPath)

        finalPathC = torch.cat((finalPath, plsPathDRS), 2)
        finalPathC = self.lastDense(finalPathC)
        # finalPathC = self.activationFinal(finalPathC)

        finalPathC = finalPathC.permute(0, 2, 1)
        finalPathC = self.convLast(finalPathC)
        finalPathC = finalPathC.permute(0, 2, 1)
        return finalPathC

def generate_ts(timesteps, num):
    orig = np.random.randint(0, timesteps, size=num)
    lastOnes = np.random.randint((int)(math.floor(timesteps*0.5)), timesteps, size=(int)(num/2))
    lastOnes2 = np.random.randint((int)(math.floor(timesteps * 0.95)), timesteps, size=(int)(num / 2))
    finalTs = np.concat((orig,lastOnes,lastOnes2))
    finalTs = np.random.choice(finalTs,num)
    return finalTs
    # return np.arange(0,timesteps,1,dtype=int)
    # return np.random.randint(timesteps-1, timesteps, size=num)
    # return np.random.randint(timesteps-3, timesteps, size=num)

def forward_noise(timesteps, x, t, isVizualize=False):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(loc=0.5,scale=0.5,size=x.shape)  # noise mask
    # noise = np.random.normal(loc=0.5,scale=0.33,size=x.shape)  # noise mask
    # noise = np.random.uniform(low=0,high=1,size=x.shape)

    # for i in range(1, noise.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([noise[i, h - 1, 0], noise[i, h, 0]], [noise[i, h - 1, 1], noise[i, h, 1]], marker='', zorder=2, alpha=0.5,color='c')


    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    img_a = x * (1 - np.pow(a,3)) + noise * np.pow(a,3)
    img_b = x * (1 - np.pow(b,3)) + noise * np.pow(b,3)

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

    if isVizualize==True:
        if oneTimeVisGenAB == True:
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, timesteps)
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, 12):
                    plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, 12):
                    plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            oneTimeVisGenAB = False

    return img_a, img_b

class ConvLayers(nn.Module):
    def __init__(self, maxLengthSize,size,lenOffest):
        super(ConvLayers,self).__init__()
        self.size=size
        self.maxLengthSize=maxLengthSize
        self.conv1d1 = nn.Conv1d(maxLengthSize +lenOffest, self.size, 11, padding=5)
        self.activation = nn.SiLU()
        self.activationFinal = nn.Tanh()
        self.conv2d1 = nn.Conv1d(self.size, self.size, 9, padding=4)
        self.conv3d1 = nn.Conv1d(self.size, self.size, 7, padding=3)

        # self.auxDense = nn.Linear(510, 512)

        self.conv1d2 = nn.Conv1d(maxLengthSize +lenOffest, self.size, 11, padding=10, dilation=2)
        self.conv2d2 = nn.Conv1d(self.size, self.size, 9, padding=8, dilation=2)
        self.conv3d2 = nn.Conv1d(self.size, self.size, 7, padding=6, dilation=2)

        self.conv1d3 = nn.Conv1d(maxLengthSize +lenOffest, self.size, 11, padding=15, dilation=3)
        self.conv2d3 = nn.Conv1d(self.size, self.size, 9, padding=12, dilation=3)
        self.conv3d3 = nn.Conv1d(self.size, self.size, 7, padding=9, dilation=3)

        self.inputProcess = nn.Linear(self.size,64)

        self.e11 = nn.Conv1d(maxLengthSize +lenOffest, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv1d(64, self.size, kernel_size=3, padding=1)

        self.lastDenseInPathAdjustTCN = nn.Linear(self.size, maxLengthSize)
        self.lastDenseInPathAdjustUNET = nn.Linear(64, maxLengthSize)
        self.lastDenseInPath = nn.Linear(self.size, self.size)

    def forward(self,input):
        inputFixed = self.inputProcess(input)

        xe11 = self.activation(self.e11(inputFixed))
        xe12 = self.activation(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.activation(self.e21(xp1))
        xe22 = self.activation(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.activation(self.e31(xp2))
        xe32 = self.activation(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.activation(self.e41(xp3))
        xe42 = self.activation(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.activation(self.e51(xp4))
        xe52 = self.activation(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.activation(self.d11(xu11))
        xd12 = self.activation(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.activation(self.d21(xu22))
        xd22 = self.activation(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.activation(self.d31(xu33))
        xd32 = self.activation(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.activation(self.d41(xu44))
        xd42 = self.activation(self.d42(xd41))

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv1d1(input)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv2d1(mergePath1)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.conv3d1(mergePath1)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath1 = self.activation(mergePath1)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv1d2(input)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv2d2(mergePath2)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.conv3d2(mergePath2)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath2 = self.activation(mergePath2)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv1d3(input)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv2d3(mergePath3)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)

        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.conv3d3(mergePath3)
        # mergePath = mergePath.permute(0, 2, 1)
        mergePath3 = self.activation(mergePath3)

        mergePath1 = self.lastDenseInPathAdjustTCN(mergePath1)
        mergePath2 = self.lastDenseInPathAdjustTCN(mergePath2)
        mergePath3 = self.lastDenseInPathAdjustTCN(mergePath3)
        xd42 = self.lastDenseInPathAdjustUNET(xd42)

        mergePath1 = torch.reshape(mergePath1, (input.shape[0], self.maxLengthSize, -1))
        mergePath2 = torch.reshape(mergePath2, (input.shape[0], self.maxLengthSize, -1))
        mergePath3 = torch.reshape(mergePath3, (input.shape[0], self.maxLengthSize, -1))
        xd42 = torch.reshape(xd42, (input.shape[0], self.maxLengthSize, -1))

        mergePath1 = self.lastDenseInPath(mergePath1)
        mergePath2 = self.lastDenseInPath(mergePath2)
        mergePath3 = self.lastDenseInPath(mergePath3)
        xd42 = self.lastDenseInPath(xd42)

        mergePath3Total = torch.cat((mergePath1, mergePath2, mergePath3, xd42), 2)

        return mergePath3Total