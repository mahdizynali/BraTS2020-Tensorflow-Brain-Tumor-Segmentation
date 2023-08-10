from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Activation, Attention
import tensorflow.keras as tfk


# class simpleUnet:
#     '''simple unet model'''

#     def __init__(self, input_layer, kernel, dropout) :
#         self.input_layer = input_layer
#         self.kernel = kernel
#         self.dropout = dropout
#     def generateLayers(self):
#         conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(self.input_layer)
#         conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv1)
        
#         mxpool = Maxmxpooling2D(mxpool_size=(2, 2))(conv1)
#         conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool)
#         conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        
#         mxpool1 = Maxmxpooling2D(mxpool_size=(2, 2))(conv)
#         conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool1)
#         conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv2)
        
#         mxpool2 = Maxmxpooling2D(mxpool_size=(2, 2))(conv2)
#         conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool2)
#         conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv3)
        
        
#         mxpool4 = Maxmxpooling2D(mxpool_size=(2, 2))(conv3)
#         conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(mxpool4)
#         conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv5)
#         drop5 = Dropout(self.dropout)(conv5)

#         up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(drop5))
#         merge7 = concatenate([conv3,up7], axis = 3)
#         conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge7)
#         conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv7)

#         up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv7))
#         merge8 = concatenate([conv2,up8], axis = 3)
#         conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge8)
#         conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv8)

#         up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv8))
#         merge9 = concatenate([conv,up9], axis = 3)
#         conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge9)
#         conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv9)
        
#         up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(UpSampling2D(size = (2,2))(conv9))
#         merge = concatenate([conv1,up], axis = 3)
#         conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(merge)
#         conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = self.kernel)(conv)
        
#         conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
        
#         return Model(input_layer = self.input_layer, outputs = conv10)

#==========================================================

class attUnet:
    '''Attention Unet Model'''

    def __init__(self, input_layer, kernel, dropout) :
        self.input_layer = input_layer
        self.kernel = kernel
        self.dropout = dropout
        self.generateLayers()

    def generateLayers(self):
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(self.input_layer)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv1)

        mxpool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv)

        mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv2)

        mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(conv3)

        mxpool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.kernel)(mxpool4)
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

        return tfk.Model(input_layer = self.input_layer, outputs = conv10)