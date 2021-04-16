import os
import numpy as np
import scipy.io as sio
import tifffile

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


#Load dataset
def loadData(name,data_path):
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
def loadTifDataTomat(data_path,save_DataPath,name):
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



def loadTifMat(data_path,name):
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