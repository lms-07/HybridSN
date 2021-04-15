import os
import spectral
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tifffile
import datetime

import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization,concatenate
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

### Drw Model figure---Model Visualization
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback,EarlyStopping

from operator import truediv

from plotly.offline import init_notebook_mode

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, roc_auc_score


### Note start time for model running.
starttime = datetime.datetime.now()


## GLOBAL VARIABLES
# dataset1 = 'IP'
# dataset2 = 'SA'
# dataset3 = 'PU'
# dataset4 = ‘HU13'
# dataset5 = 'KSC'
dataset = 'KSC'
test_ratio = 0.8
windowSize = 25


### Create file path for savinerated file during model runtime.
save_path=os.path.join(os.getcwd(),"Result/"+"Stand"+starttime.strftime("%y-%m-%d-%H.%M")+str("_")+dataset)
save_DataPath=os.path.join(os.getcwd(),"data/")
data_path = os.path.join(os.getcwd(),'data') #os.getcwd()
os.makedirs(save_path)

data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))
labels = sio.loadmat(os.path.join(data_path,'KSC_gt.mat'))

       

print(data)
print(labels)


#Load dataset
def loadData(name):
    data_path = os.path.join(os.getcwd(),'data') #os.getcwd()
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'HU13':
        # dict_keys(['__header__', '__version__', '__globals__', 'Houston'])
        #dict_values([b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Jul 17 16:45:01 2019', '1.0', [], array()])
        #data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))
        #labels = sio.loadmat(os.path.join(data_path,'Houston_gt.mat'))
        data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['Houston']
        labels = sio.loadmat(os.path.join(data_path,'Houston_gt.mat'))['Houston_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path,'KSC_gt.mat'))['KSC_gt']

    return data, labels

  

# Use tifffile pkg read the hyperspectral img.
# Load .tiff data set and converted to .mat data
def loadTifDataTomat(name):
    if name=='HU13':
        totalTif=tifffile.imread(os.path.join(data_path,'2013_IEEE_GRSS_DF_Contest_CASI.tif'))
        trainTif=tifffile.imread(os.path.join(data_path,'train_roi.tif'))
        valTif=tifffile.imread(os.path.join(data_path,'val_roi.tif'))

        print(totalTif.shape,trainTif.shape,valTif.shape)
        #spectral.imshow(totalTif)
        #spectral.imshow(trainTif)
        sio.savemat(os.path.join(save_DataPath,"totalTifHouston13.mat"),{'totalTifHouston13':totalTif})
        sio.savemat(os.path.join(save_DataPath,"trainTifHouston13.mat"),{'trainTifHouston13':trainTif})
        sio.savemat(os.path.join(save_DataPath,"valTifHouston13.mat"),{'valTifHouston13':valTif})

  

def loadTifMat(name):
    if name=='HU13':
        data=sio.loadmat(os.path.join(data_path, 'totalTifHouston13.mat'))['totalTifHouston13']
        train=sio.loadmat(os.path.join(data_path, 'trainTifHouston13.mat'))['trainTifHouston13']
        val=sio.loadmat(os.path.join(data_path, 'valTifHouston13.mat'))['valTifHouston13']
        return data,train,val

  

### Using PCA for removing the spectral redundancy(冗余) 
### Reduce the spectral dimension, from high-dimensional to low-dimensional.
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

  

### Padding zeros
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

  

### Create data cube,3D-patch.
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

  

# Dataset split.
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

  

X, y = loadData(dataset)

X.shape, y.shape

K = X.shape[2]

  

K = 32 if dataset == 'IP' else 16
X,pca = applyPCA(X,numComponents=K)
X.shape,pca

  

X, y = createImageCubes(X, y, windowSize=windowSize)

X.shape, y.shape


# 3:7 Split  Train:Test st 3:7 from total data
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape


# Before mode training splite to get validation
# Will not resample the validation set after each epoch
# 2:1 Split  Train:Valid split 2:1 from total Train 
Xtrain, Xvalid, ytrain, yvalid = splitTrainTestSet(Xtrain, ytrain, 0.3333)

Xtrain.shape, Xvalid.shape, ytrain.shape, yvalid.shape


# Due to the scarce samples, split from Xtest as 7:1 for Xtest and XValid,respectly.
Xvalid,Xtest,yvalid,ytest= splitTrainTestSet(Xtest, ytest, 0.875)
Xvalid.shape,Xtest.shape,yvalid.shape,ytest.shape

       

# Model and Training

  

Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
Xtrain.shape

  

ytrain = np_utils.to_categorical(ytrain)
ytrain.shape


  

# Reshape validition
Xvalid = Xvalid.reshape(-1, windowSize, windowSize, K, 1)
Xvalid.shape

  

# For validation
yvalid = np_utils.to_categorical(yvalid)
yvalid.shape

  

S = windowSize
L = K
# IP SA:16
# HU13:15
# PU:9
if (dataset == 'IP' or dataset == 'SA'):
    output_units = 16
elif dataset=='PU':
    output_units=9
elif dataset=='HU13':
    output_units=15
elif dataset=='KSC':
    output_units=13

  

## input layer
input_layer = Input((S, S, L, 1))
input_shape=input_layer.shape

#input_layer1=Reshape((input_shape[1],input_shape[2],input_shape[3]*input_shape[4]))(input_layer)
## 2D convolutional layers

conv_3d_layer11=Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(input_layer)
conv_3d_layer21=Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_3d_layer11)
conv_3d_layer311=Conv3D(filters=32,kernel_size=(3,3,3),activation='relu')(conv_3d_layer21)

conv3d_shape_31 = conv_3d_layer311.shape
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv3d_shape_31))
### conv3D-->conv2D
conv_3d_layer312 = Reshape((conv3d_shape_31[1], conv3d_shape_31[2], conv3d_shape_31[3]*conv3d_shape_31[4]))(conv_3d_layer311)
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv_3d_layer312.shape))
conv_2d_layer41 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_3d_layer312)

## 3D convolutional layers
### filters---卷积核数；kernel_size---卷积核大小
conv_3d_layer12 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_3d_layer22 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_3d_layer12)
conv_3d_layer321 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_3d_layer22)

conv3d_shape_32 = conv_3d_layer321.shape
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv3d_shape_32))
### conv3D-->conv2D
conv_3d_layer322 = Reshape((conv3d_shape_32[1], conv3d_shape_32[2], conv3d_shape_32[3]*conv3d_shape_32[4]))(conv_3d_layer321)
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv_3d_layer322.shape))
conv_2d_layer42 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_3d_layer322)


### Flatte层:将张量扁平化，即输入一维化，不影响张量大小.
### 常在Conv层和Dense层之间过渡.
flatten_layer1 = Flatten()(conv_2d_layer41)
flatten_layer2 = Flatten()(conv_2d_layer42)

flatten_merge_layer=concatenate([flatten_layer1,flatten_layer2])

## fully connected layers
### Dense层：全连接层.
### Dropout层：Dense层之后，防止过拟合，提高模型泛化性能.
dense_layer1 = Dense(units=256, activation='relu')(flatten_merge_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

       

## Average performance

## input layer
input_layer = Input((S, S, L, 1))
input_shape=input_layer.shape

input_layer1=Reshape((input_shape[1],input_shape[2],input_shape[3]*input_shape[4]))(input_layer)
## 2D convolutional layers

conv_2d_layer1=Conv2D(filters=2,kernel_size=(3,3),activation='relu')(input_layer1)
conv_2d_layer2=Conv2D(filters=16,kernel_size=(3,3),activation='relu')(conv_2d_layer1)
conv_2d_layer3=Conv2D(filters=64,kernel_size=(3,3),activation='relu')(conv_2d_layer2)

## 3D convolutional layers
### filters---卷积核数；kernel_size---卷积核大小
conv_3d_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_3d_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_3d_layer1)
conv_3d_layer3_1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_3d_layer2)

conv3d_shape = conv_3d_layer3_1.shape
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv3d_shape))
### conv3D-->conv2D
conv_3d_layer3_2 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_3d_layer3_1)
print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv_3d_layer3_2.shape))

conv_2d_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_3d_layer3_2)


### Flatte层:将张量扁平化，即输入一维化，不影响张量大小.
### 常在Conv层和Dense层之间过渡.
flatten_layer1 = Flatten()(conv_2d_layer3)
flatten_layer2 = Flatten()(conv_2d_layer4)

flatten_merge_layer=concatenate([flatten_layer1,flatten_layer2])

## fully connected layers
### Dense层：全连接层.
### Dropout层：Dense层之后，防止过拟合，提高模型泛化性能.
dense_layer1 = Dense(units=256, activation='relu')(flatten_merge_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

  

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

  

model.summary()

### Model Visualization
plot_model(model,to_file=os.path.join(save_path,'ModelVisual.png'),show_shapes=True)

  

# compiling the model
adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

  

# checkpoint
filepath = "best-model.hdf5"
# ve_best_only mode
checkpoint = ModelCheckpoint(os.path.join(save_path,filepath), monitor='accuracy', verbose=1, save_best_only=True, mode='max')
earlyStopping=EarlyStopping(monitor='accuracy',patience=15,mode=max)
#earlyStopping=EarlyStopping(monitor='val_accuracy',patience=30,mode='max')
#earlyStopping=EarlyStopping(monitor='val_loss',patience=15,mode='min')
#callbacks_list = [checkpoint,MyCallback(),EarlyStopping(monitor='roc_auc', patience=20, verbose=2, mode='max')]
#callbacks_list = [checkpoint,earlyStopping]
callbacks_list = [checkpoint]

  

starttimeFIT = datetime.datetime.now()
###About 60 epochs to reach  acceptable accuracy.
#history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=200, callbacks=callbacks_list)
#history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=300,
#callbacks=callbacks_list,validation_data=(Xvalid,yvalid),validation_steps=10,validation_batch_size=64)
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, callbacks=callbacks_list,validation_data=(Xvalid,yvalid))
endtimeFIT = datetime.datetime.now()
print("Model training time:{}".format(int((endtimeFIT - starttimeFIT).seconds)))

  

plt.figure(figsize=(7,7))
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
#plt.legend(['Training'], loc='upper right')
plt.legend(['Training','Validation'], loc='upper right')
plt.savefig(os.path.join(save_path,'loss_curve.pdf'))
plt.show()

  

plt.figure(figsize=(7,7))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
#plt.legend(['Training'],loc='lower right')
plt.legend(['Training','Validation'],loc='lower right')
plt.savefig(os.path.join(save_path,'acc_curve.pdf'))
plt.show()

       

# Test

  

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

  

def reports (X_test,y_test,name):
    #start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    #end = time.time()
    #print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']

    elif name=='HU13':
        target_names = ['Grass_healthy','Grass_stressed','Grass_synthetic','Tree','Soil','Water','Residential',
                        'Commercial','Road','Highway','Railway','Parking_lot1','Parking_lot2','Tennis_court','Running_track']

    elif name=="KSC":
        target_names = ['Scrub','Willow swamp','Cabbage palm hammock','Cabbage palm/oak hammock','Slash pine','Oak/broadleaf hammock',
                        'Hardwood swamp','Graminoid marsh','Spartine marsh','Cattail marsh','Salt marsh','Mud flats','Water']


    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]
    Test_accuracy = score[1]*100

    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

  

# load best weights
model.load_weights(os.path.join(save_path,filepath))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

  

Y_pred_train=model.predict(Xtrain)
Y_pred_train=np.argmax(Y_pred_train,axis=1)

classificationTrain=classification_report(np.argmax(ytrain,axis=1),Y_pred_train)
print(classificationTrain)

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtrain,ytrain,dataset)
classification = str(classification)
confusion = str(confusion)

file_TrainName =dataset+"classification_train_report.txt"
with open(os.path.join(save_path,file_TrainName), 'w') as x_file:
    x_file.write('{} Train loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Train accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

  

Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
Xtest.shape

  

ytest = np_utils.to_categorical(ytest)
ytest.shape

  

Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)

  

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
classification = str(classification)
confusion = str(confusion)
file_name =dataset+"classification_report.txt"

with open(os.path.join(save_path,file_name), 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

  

def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch

  

# load the original image
X, y = loadData(dataset)

  

height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K

  

X,pca = applyPCA(X, numComponents=numComponents)

  

X = padWithZeros(X, PATCH_SIZE//2)

  

# calculate the predicted image
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if target == 0 :
            continue
        else :
            image_patch=Patch(X,i,j)
            X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction+1

  

ground_truth = spectral.imshow(classes = y,figsize =(7,7))

  

predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))

  

spectral.save_rgb(os.path.join(save_path,str(dataset)+"_predictions.jpg"), outputs.astype(int), colors=spectral.spy_colors)

  

spectral.save_rgb(os.path.join(save_path,str(dataset)+"_ground_truth.jpg"), y, colors=spectral.spy_colors)

  

#Running Time.
endtime = datetime.datetime.now()
totalRunTime=(endtime - starttime).seconds/60
print("Total runtime for Model on Dataset {}: {}".format(dataset,totalRunTime))