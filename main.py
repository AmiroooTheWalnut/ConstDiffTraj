import DataGeneratorFcn
import MinimalisticModelV3_LOSS as MinimalisticModel
import tensorflow as tf
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
import math
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from LossVisualizer import LossVisualizer


import tf2onnx
import onnx

print(tf.config.list_physical_devices('GPU'))

modelVersion="V3_LOSS"
continueTrain=True
numOptIterates=500
isSkipTrain=False

def visualiza(numInstances, input, axis=None):
    cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
    cmap = cm.get_cmap(cmap_name, numInstances)
    for i in range(numInstances):
        for h in range(1, trajectoryLength):
            if input[i, h, 0] > -1:
                color = cmap(i)
                if axis==None:
                    plt.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='o')
                else:
                    axis.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='o')
    if axis == None:
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                    extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                    origin='lower')
    else:
        axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                   origin='lower')
    # plt.grid(True)
    if axis == None:
        plt.show()

numTrajectories=2000
trajectoryLength=4

[data,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
# DataGeneratorFcn.generateSyntheticDataVariableLength(numTrajectories=100,longestTrajectory=80,numGrid=50,seed=3,visualize=True)

time=datetime.now()
np.random.seed(time.minute+time.hour+time.microsecond)

np.random.shuffle(data)

timesteps = 32
BATCH_SIZE = 128

model = MinimalisticModel.make_model(serializedSelected2DMatrix, lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)
model.summary()

plot_model(model, to_file="model"+modelVersion+".png", show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(model, to_file="model"+modelVersion+".png",
           expand_nested=True, show_shapes=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
if "LOSS" in modelVersion:
    loss_func = MinimalisticModel.CustomLoss
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=optimizer)
    model = MinimalisticModel.CustomModel(model, loss_func, serializedSelected2DMatrix)
    model.compile(optimizer='adam')
else:
    loss_func = tf.keras.losses.MeanAbsoluteError()
    model.compile(loss=loss_func, optimizer=optimizer)


my_file = Path('model'+modelVersion+'.keras')
if my_file.is_file() and continueTrain==True:
    try:
        loadedModel = tf.keras.models.load_model('model' + modelVersion + '.keras')
        if "LOSS" in modelVersion:
            model.base_model=loadedModel
        else:
            model = loadedModel
    except Exception as e:
        print("FAILED TO LOAD WEGHTS!")
        print(e)

def train_one(x_img):
    x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img))
    x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts)
    # for layer in model.layers:
    #     print(layer.name)
    # intermediate_model = keras.Model(inputs=model.input, outputs=model.get_layer('pairwise_subtraction_layer_1').output)
    # intermediate_output = intermediate_model.predict([x_a, x_ts])
    # plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
    #            extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
    #            origin='lower')
    # for t in range(x_a.shape[0]):
    #     plt.plot(x_a[t,:,0],x_a[t,:,1])
    # plt.show()
    loss = model.train_on_batch(x=[x_a, x_ts], y=x_b)
    # visualiza(x_a.shape[0], x_a)
    # visualiza(x_b.shape[0], x_b)
    return loss

def train(X_train, BATCH_SIZE, numIterates=30, isVisualizeLoss=False):
    bar = trange(numIterates)
    total = 100
    if isVisualizeLoss==True:
        lv=LossVisualizer(numIterates)
    for i in bar:
        for j in range(total):
            x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
            loss = train_one(x_img)
            #if not isinstance(loss, np.ndarray):
            #    loss = np.array(list(loss.items())[0][1])
            pg = (j / total) * 100
            if j % 5 == 0:
                bar.set_description(f'loss: {loss:.5f}, p: {pg:.2f}%')
        if isVisualizeLoss == True:
            lv.values[i]=loss
    if isVisualizeLoss == True:
        lv.visualize(saveFileName='Loss_tensorflow_' + modelVersion + '.png',titleText="Tensorflow loss value over iterations. Model "+modelVersion+".")

if isSkipTrain==False:
    train(data,BATCH_SIZE,numIterates=numOptIterates,isVisualizeLoss=True)

def predict(trajectoryLength,numTraj=10):
    x = np.random.normal(size=(numTraj, trajectoryLength, 2))*0.9
    #fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps/2))
    #for i in trange(timesteps):
    for i in range(timesteps):
        #colVal=i%(int)(timesteps/2)
        #rowVal=math.floor(i/(int)(timesteps/2))
        t = i
        x = model.predict([x, np.full((numTraj), t)], verbose=0)
        #visualiza(numTraj, x, axis=axs[rowVal,colVal])
        #axs[rowVal,colVal].title.set_text("Time: "+str(i))
    #plt.show()
    return x

numPredicts = 8000
for i in range(1):
    start = datetime.now()
    pred = predict(trajectoryLength,numTraj=numPredicts)
    end=datetime.now()
    delta_time = end - start
    delta_milliseconds = delta_time.total_seconds() * 1000
    print(str(delta_milliseconds))

# def predict_step():
#     xs = []
#     x = np.random.normal(size=(8, IMG_SIZE, IMG_SIZE, 3))
#
#     for i in trange(timesteps):
#         t = i
#         x = model.predict([x, np.full((8),  t)], verbose=0)
#         if i % 2 == 0:
#             xs.append(x[0])
#
#     plt.figure(figsize=(20, 2))
#     for i in range(len(xs)):
#         plt.subplot(1, len(xs), i+1)
#         plt.imshow(cvtImg(xs[i]))
#         plt.title(f'{i}')
#         plt.axis('off')

visualiza(numPredicts, pred)

onnx_model_path = 'model' + modelVersion + '.onnx'

if "LOSS" in modelVersion:
    model.base_model.save('model' + modelVersion + '.keras')
    tf2onnx.convert.from_keras(
        model.base_model,
        input_signature=(tf.TensorSpec([None, 4, 2], tf.float32, name="input_tensor"),
                         tf.TensorSpec([None], tf.float32, name="input_scalar")),
        output_path=onnx_model_path
    )
else:
    model.save('model'+modelVersion+'.keras')
    tf2onnx.convert.from_keras(
        model,
        input_signature=(tf.TensorSpec([None, 4, 2], tf.float32, name="input_tensor"),
                         tf.TensorSpec([None], tf.float32, name="input_scalar")),
        output_path=onnx_model_path
    )


print("!!!")