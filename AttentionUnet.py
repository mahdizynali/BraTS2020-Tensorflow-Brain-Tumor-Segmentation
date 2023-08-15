from tensorflow.keras.layers import Input,Add, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Activation, Attention, BatchNormalization
from tensorflow.keras.models import Model

class attUnet:
    '''Attention Unet Model'''

    def __init__(self, inp, kernel, dropout):
        self.kernel = kernel
        self.dropout = dropout
        self.input_layers = Input(shape=inp)
        self.generateModel()
        
    def generateModel(self) -> Model:
        
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(self.input_layers)
        conv1 = Dropout(self.dropout)(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv1)
        conv1 = BatchNormalization()(conv1)
        mxpool = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool)
        conv = Dropout(self.dropout)(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv)
        conv = BatchNormalization()(conv)
        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool1)
        conv2 = Dropout(self.dropout)(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv2)
        conv2 = BatchNormalization()(conv2)
        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool2)
        conv3 = Dropout(self.dropout)(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv3)
        conv3 = BatchNormalization()(conv3)
        mxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool3)
        conv5 = Dropout(self.dropout)(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv5)
        drop5 = Dropout(self.dropout)(conv5)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(
            UpSampling2D(size=(2, 2))(drop5))
        merge7 = concatenate([conv3, up7], axis=3)
        att7 = Attention(use_scale=False)([conv3, up7])
        merge7 = concatenate([att7, merge7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        att8 = Attention(use_scale=False)([conv2, up8])
        merge8 = concatenate([att8, merge8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv, up9], axis=3)
        att9 = Attention(use_scale=False)([conv, up9])
        merge9 = concatenate([att9, merge9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv9)

        up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(
            UpSampling2D(size=(2, 2))(conv9))
        merge = concatenate([conv1, up], axis=3)
        att10 = Attention(use_scale=False)([conv1, up])
        merge = concatenate([att10, merge], axis=3)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv)

        conv10 = Conv2D(4, (1, 1), activation='softmax')(conv)
        
        return Model(inputs = self.input_layers, outputs = conv10)


#========================================================== 
class simpleUnet():
    '''simple unet model'''

    def __init__(self, inp, kernel, dropout):
        self.kernel = kernel
        self.dropout = dropout
        self.input_layer = Input(shape=inp)
        self.generateModel()

    def generateModel(self) -> Model :
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(self.input_layer)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv1)
        
        mxpool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool)
        conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        
        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv2)
        
        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv3)
        
        
        mxpool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv5)
        drop5 = Dropout(self.dropout)(conv5)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(drop5))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv9)
        
        up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv9))
        merge = concatenate([conv1,up], axis = 3)
        conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge)
        conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        
        conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
        
        return Model(inputs = self.input_layer, outputs = conv10)


class simpleUnet2():
    '''simple unet model'''

    def __init__(self, inp, kernel, dropout):
        self.kernel = kernel
        self.dropout = dropout
        self.input_layer = Input(shape=inp)
        self.generateModel()

    def generateModel(self) -> Model :
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(self.input_layer)
        conv1 = Dropout(self.dropout)(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv1)
        conv1 = BatchNormalization()(conv1)
        
        mxpool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool)
        conv = Dropout(self.dropout)(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv)
        conv = BatchNormalization()(conv)
        
        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool1)
        conv2 = Dropout(self.dropout)(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv2)
        conv2 = BatchNormalization()(conv2)
        
        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool2)
        conv3 = Dropout(self.dropout)(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv3)
        conv3 = BatchNormalization()(conv3)
        
        mxpool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool4)
        drop5 = Dropout(self.dropout)(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv5)
        conv5 = BatchNormalization()(conv5)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(UpSampling2D(size=(2, 2))(drop5))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv9)
        
        up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=self.kernel)(UpSampling2D(size=(2, 2))(conv9))
        merge = concatenate([conv1, up], axis=3)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(merge)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv)
        
        conv10 = Conv2D(4, (1, 1), activation='softmax')(conv)
        
        return Model(inputs=self.input_layer, outputs=conv10)

#==========================================================

class testUnet():
    '''simple unet model'''

    def __init__(self, inp, kernel, dropout):
        self.kernel = kernel
        self.dropout = dropout
        self.input_layer = Input(shape=inp)
        self.generateModel()

    def generateModel(self) -> Model :
        conv1 = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(self.input_layer)
        conv1 = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv1)
        conv1 = BatchNormalization()(conv1)
        mxpool = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool)
        conv = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        conv = BatchNormalization()(conv)
        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv2)
        conv2 = BatchNormalization()(conv2)
        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv3)
        conv3 = BatchNormalization()(conv3)
        mxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv5 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool3)
        conv5 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(self.dropout)(conv5)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(drop5))
        merge7 = concatenate([conv3,up7], axis = 3)
        # att7 = Attention(use_scale=False)([conv3, up7])
        # merge7 = concatenate([att7, merge7], axis=3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        # att8 = Attention(use_scale=False)([conv2, up8])
        # merge8 = concatenate([att8, merge8], axis=3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv,up9], axis = 3)
        # att9 = Attention(use_scale=False)([conv, up9])
        # merge9 = concatenate([att9, merge9], axis=3)
        conv9 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge9)
        conv9 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv9)
        
        up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv9))
        merge = concatenate([conv1,up], axis = 3)
        # att10 = Attention(use_scale=False)([conv1, up])
        # merge = concatenate([att10, merge], axis=3)
        conv = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge)
        conv = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        
        conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
        
        return Model(inputs = self.input_layer, outputs = conv10)
    
    
    

class compUnet():
    '''simple unet model'''

    def __init__(self, inp, kernel, dropout):
        self.kernel = kernel
        self.dropout = dropout
        self.input_layer = Input(shape=inp)
        self.generateModel()

    def residual_block(self, input_layer, filters):
        conv1 = Conv2D(filters, 7, activation='relu', padding='same', kernel_initializer=self.kernel)(input_layer)
        conv1 = BatchNormalization()(conv1)
        
        conv2 = Conv2D(filters, 5, activation='relu', padding='same', kernel_initializer=self.kernel)(conv1)
        conv2 = BatchNormalization()(conv2)
        
        conv3 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv2)
        conv3 = BatchNormalization()(conv3)
        
        residual_output = Add()([input_layer, conv3])
        return residual_output

    def generateModel(self) -> Model :
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(self.input_layer)
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv1)
        conv1 = BatchNormalization()(conv1)
        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool1)
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv2)
        conv2 = BatchNormalization()(conv2)
        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv3)
        conv3 = BatchNormalization()(conv3)
        mxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv4)
        conv4 = BatchNormalization()(conv4)
        mxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool4)
        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv5)
        conv5 = BatchNormalization()(conv5)
        mxpool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool5)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv6)
        conv6 = BatchNormalization()(conv6)
        mxpool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool6)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv7)
        conv7 = BatchNormalization()(conv7)      
        
        drop7 = Dropout(self.dropout)(conv7)

        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(drop7))
        merge8 = concatenate([conv6,up8], axis = 3)
        att8 = Attention(use_scale=False)([conv6, up8])
        merge8 = concatenate([att8, merge8], axis=3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv8)

        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv5,up9], axis = 3)
        att9 = Attention(use_scale=False)([conv5, up9])
        merge9 = concatenate([att9, merge9], axis=3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([conv4,up10], axis = 3)
        att10 = Attention(use_scale=False)([conv4, up10])
        merge10 = concatenate([att10, merge10], axis=3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge10)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv10)
        
        up11 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv10))
        merge11 = concatenate([conv3,up11], axis = 3)
        att11 = Attention(use_scale=False)([conv3, up11])
        merge11 = concatenate([att11, merge11], axis=3)
        conv11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge11)
        conv11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv11)

        up12 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv11))
        merge12 = concatenate([conv2,up12], axis = 3)
        att12 = Attention(use_scale=False)([conv2, up12])
        merge12 = concatenate([att12, merge12], axis=3)
        conv12 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge12)
        conv12 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv12)

        up13 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv12))
        merge13 = concatenate([conv1,up13], axis = 3)
        att13 = Attention(use_scale=False)([conv1, up13])
        merge13 = concatenate([att13, merge13], axis=3)
        conv13 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge13)
        conv13 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv13)
        
        conv14 = Conv2D(4, (1,1), activation = 'softmax')(conv13) # output layer
        
        return Model(inputs = self.input_layer, outputs = conv14)