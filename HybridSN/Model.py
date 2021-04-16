from keras.models import Model
from keras.layers import Dropout, Input
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization,concatenate

def getHybridSNModel(S,L,output_units):
    ## input layer
    input_layer = Input((S, S, L, 1))
    ## convolutional layers
    ### filters---卷积核数；kernel_size---卷积核大小

    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer3 = Conv3D(filters=16, kernel_size=(3, 3, 7), activation='relu')(conv_layer1)



    ### AttributeError: 'KerasTensor' object has no attribute '_keras_shape'
    ### Try to use shape instead
    #print(conv_layer3._keras_shape)
    #conv3d_shape = conv_layer3._keras_shape

    conv3d_shape = conv_layer3.shape
    print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv3d_shape))
    ### conv3D-->conv2D
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    print("After three times convD,and before reshaping,\nKerasTensorShape:{}".format(conv_layer3.shape))

    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    ### Flatte层:将张量扁平化，即输入一维化，不影响张量大小.
    ### 常在Conv层和Dense层之间过渡.
    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    ### Dense层：全连接层.
    ### Dropout层：Dense层之后，防止过拟合，提高模型泛化性能.
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)


    model = Model(inputs=input_layer, outputs=output_layer)

    return model

