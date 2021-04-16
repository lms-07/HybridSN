import os
import spectral
import numpy as np
from DataLoadAndOperate import loadData,applyPCA,padWithZeros

def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

# Visualization for the ground truth img and the predicted img.
def Visualization(dataset,model,windowSize,K,save_path,data_path):
    #load the original image
    X, y = loadData(dataset,data_path)

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
                image_patch=Patch(X,i,j,PATCH_SIZE)
                X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction+1

    ground_truth = spectral.imshow(classes = y,figsize =(7,7))
    predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))

    spectral.save_rgb(os.path.join(save_path,str(dataset)+"_predictions.jpg"), outputs.astype(int), colors=spectral.spy_colors)
    spectral.save_rgb(os.path.join(save_path,str(dataset)+"_ground_truth.jpg"), y, colors=spectral.spy_colors)
