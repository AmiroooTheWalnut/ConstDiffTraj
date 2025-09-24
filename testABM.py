import DataGeneratorFcn
import MinimalisticModelV2_pytorch as MinimalisticModel
from pathlib import Path
import tensorflow as tf
import numpy as np
import math
from tqdm.auto import trange
from datetime import datetime
import torch

numTrajectories=2000
trajectoryLength=4
[data,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)

numDays=20
numAgents=50000
agentDailyTraj=np.zeros((numAgents,4,2))
start=datetime.now()
for d in range(numDays):
    for a in range(numAgents):
        index=np.random.randint(0,numTrajectories-1)
        agentDailyTraj[a,:,:]=data[index,:,:]
end=datetime.now()
delta_time = end - start
delta_milliseconds = delta_time.total_seconds() * 1000
print('Mobility sampling time: '+str(delta_milliseconds))


# modelVersion="V3"
#
# timesteps = 16
# BATCH_SIZE = 64
#
# model = MinimalisticModel.make_model(serializedSelected2DMatrix, lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)
# my_file = Path('model'+modelVersion+'.keras')
# if my_file.is_file():
#     try:
#         loadedModel = tf.keras.models.load_model('model' + modelVersion + '.keras')
#         if "LOSS" in modelVersion:
#             model.base_model=loadedModel
#         else:
#             model = loadedModel
#     except Exception as e:
#         print("FAILED TO LOAD WEGHTS!")
#         print(e)
#
# def predict(trajectoryLength,numTraj=10):
#     x = np.random.normal(size=(numTraj, trajectoryLength, 2))*0.8
#     #fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps/2))
#     for i in trange(timesteps):
#         #colVal=i%(int)(timesteps/2)
#         #rowVal=math.floor(i/(int)(timesteps/2))
#         t = i
#         x = model.predict([x, np.full((numTraj), t)], verbose=0)
#         #visualiza(numTraj, x, axis=axs[rowVal,colVal])
#         #axs[rowVal,colVal].title.set_text("Time: "+str(i))
#     #plt.show()
#     return x
#
# numPredicts = numAgents
#
# agentDailyTraj=np.zeros((numAgents,4,2))
# start=datetime.now()
# for d in range(numDays):
#     agentDailyTraj = predict(trajectoryLength, numTraj=numPredicts)
#     #for a in range(numAgents):
#     #    agentDailyTraj=pred
#     #agentDailyTraj = pred
# end=datetime.now()
# delta_time = end - start
# delta_milliseconds = delta_time.total_seconds() * 1000
# print('Mobility diffusion time: '+str(delta_milliseconds))


modelVersion="V2"
isRunOnCPU=False

timesteps = 16
BATCH_SIZE = 64

model = MinimalisticModel.SimpleNN(maxLengthSize=trajectoryLength)
my_file = Path('model'+modelVersion+'.pytorch')
if my_file.is_file():
    try:
        model.load_state_dict(torch.load('model' + modelVersion + '.pytorch', weights_only=True))
    except:
        print("FAILED TO LOAD WEGHTS!")
if isRunOnCPU==False:
    model.cuda()

def predict(trajectoryLength,numTraj=10):
    x = np.random.normal(size=(numTraj, trajectoryLength, 2))
    x = torch.from_numpy(x).to(torch.float32)
    # fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps/2))
    with torch.no_grad():
        for i in trange(timesteps):
            # colVal=i%(int)(timesteps/2)
            # rowVal=math.floor(i/(int)(timesteps/2))
            t = i
            x_ts = np.full((numTraj), t)
            x_ts = torch.from_numpy(x_ts).to(torch.float32)
            x_ts = x_ts.unsqueeze(1)
            if isRunOnCPU == False:
                x = model(x.cuda(),x_ts.cuda())
            else:
                x = model(x, x_ts)
            # visualiza(numTraj, x, axis=axs[rowVal,colVal])
            # axs[rowVal,colVal].title.set_text("Time: "+str(i))
    # plt.show()
    return x

numPredicts = numAgents

agentDailyTraj=np.zeros((numAgents,4,2))
start=datetime.now()
for d in range(numDays):
    agentDailyTraj = predict(trajectoryLength, numTraj=numPredicts)
    #for a in range(numAgents):
    #    agentDailyTraj=pred
    #agentDailyTraj = pred
end=datetime.now()
delta_time = end - start
delta_milliseconds = delta_time.total_seconds() * 1000
print('Mobility diffusion time: '+str(delta_milliseconds))

print("!!!")