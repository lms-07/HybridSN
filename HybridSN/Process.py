import os
import spectral
import datetime
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from sklearn.metrics import classification_report

from DataLoadAndOperate import loadData,applyPCA,padWithZeros,createImageCubes,splitTrainTestSet
from Model import getHybridSNModel
from Testing import reports
from Visualization import Visualization

### Note start time for model running.
starttime = datetime.datetime.now()


## GLOBAL VARIABLES
# dataset1 = 'IP'
# dataset2 = 'SA'
# dataset3 = 'PU'
# dataset4 = ‘HU13'
# dataset5 = 'KSC'
dataset = 'KSC'
test_ratio = 0.7
windowSize = 19


### Create file path for savinerated file during model runtime.
save_path=os.path.join(os.path.dirname(os.getcwd()),"Result/"+"Stand"+starttime.strftime("%y-%m-%d-%H.%M")+str("_")+dataset)
save_DataPath=os.path.join(os.getcwd(),"data/")
data_path = os.path.join(os.path.dirname(os.getcwd()),'data') #os.getcwd() Up-level directory
os.makedirs(save_path)

if __name__ == '__main__':
    print(os.getcwd())
    print(os.path.dirname(os.getcwd()))
    # Load data and preprocessing.
    X, y = loadData(dataset,data_path)
    print(X.shape, y.shape)

    #K = 30 if dataset == 'IP' else 15
    K=30
    X,pca = applyPCA(X,numComponents=K)
    print(X.shape,pca)

    X, y = createImageCubes(X, y, windowSize=windowSize)
    print(X.shape, y.shape)

    # 3:7 Split Train:Test set 3:7 from total data.
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
    print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

    '''
    # Before mode training splite to get validation
    # Will not resample the validation set after each epoch
    # 2:1 Split  Train:Valid split 2:1 from total Train 
    Xtrain, Xvalid, ytrain, yvalid = splitTrainTestSet(Xtrain, ytrain, 0.3333)
    
    Xtrain.shape, Xvalid.shape, ytrain.shape, yvalid.shape
    '''

    # Due to the scarce samples, split from Xtest as 7:1 for Xtest and XValid,respectly.
    #Xvalid,Xtest,yvalid,ytest= splitTrainTestSet(Xtest, ytest, 0.875)
    #Xvalid.shape,Xtest.shape,yvalid.shape,ytest.shape


    # Model and Training
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    print(Xtrain.shape)
    ytrain = np_utils.to_categorical(ytrain)
    print(ytrain.shape)

    # Reshape validition
    #Xvalid = Xvalid.reshape(-1, windowSize, windowSize, K, 1)
    #Xvalid.shape
    # For validation
    #yvalid = np_utils.to_categorical(yvalid)
    #yvalid.shape

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

    model=getHybridSNModel(S,L,output_units)
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
    #earlyStopping=EarlyStopping(monitor='accuracy',patience=15,mode=max)
    callbacks_list = [checkpoint]

    # Recording the training time.
    starttimeFIT = datetime.datetime.now()
    ###About 60 epochs to reach an acceptable accuracy.
    history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, callbacks=callbacks_list)
    endtimeFIT = datetime.datetime.now()
    print("Model training time:{}".format(int((endtimeFIT - starttimeFIT).seconds)))

    # Loss figure.
    plt.figure(figsize=(7,7))
    plt.grid()
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training'], loc='upper right')
    #plt.legend(['Training','Validation'], loc='upper right')
    plt.savefig(os.path.join(save_path,'loss_curve.pdf'))
    plt.show()

    # Accuracy figure.
    plt.figure(figsize=(7,7))
    plt.ylim(0,1.1)
    plt.grid()
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.plot(history.history['val_acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training'],loc='lower right')
    #plt.legend(['Training','Validation'],loc='lower right')
    plt.savefig(os.path.join(save_path,'acc_curve.pdf'))
    plt.show()

    # Test
    # Load best weights.
    model.load_weights(os.path.join(save_path,filepath))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Model testing in Xtrain and reporting.
    Y_pred_train=model.predict(Xtrain)
    Y_pred_train=np.argmax(Y_pred_train,axis=1)

    classificationTrain=classification_report(np.argmax(ytrain,axis=1),Y_pred_train)
    print(classificationTrain)

    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtrain,ytrain,dataset,model)
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
    print(Xtest.shape)

    ytest = np_utils.to_categorical(ytest)
    ytest.shape
    print(ytest.shape)

    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)

    classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
    print(classification)

    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset,model)
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

    Visualization(dataset,model,windowSize,K,save_path,data_path)

    #Running Time.
    endtime = datetime.datetime.now()
    totalRunTime=(endtime - starttime).seconds/60
    print("Total runtime for Model on Dataset {}: {}".format(dataset,totalRunTime))












