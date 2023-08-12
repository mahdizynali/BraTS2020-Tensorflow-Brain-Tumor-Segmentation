import os
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
from keras.utils import Sequence
np.set_printoptions(precision=3, suppress=True)
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, warp, AffineTransform
from config import TRAIN_DATASET_PATH, IMG_SIZE, VOLUME_START, VOLUME_SLICES
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

def path_to_ids(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = path_to_ids(train_and_val_directories); 

train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15) 

# class DataGenerator(Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         if shuffle == True :
#             self.augment_shuffle()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         Batch_ids = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(Batch_ids)

#         return X, y

#     def augment_shuffle(self):
#         self.indexes = np.arange(len(self.list_IDs))
#         np.random.shuffle(self.indexes)
    
#     def augment_transform(self, data):
#         data = scaler.fit_transform(
#                 data.reshape(-1, data.shape[-1])).reshape(data.shape)    
#         return data    

#     def __data_generation(self, Batch_ids):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
#         y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
#         Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        
#         # Generate data
#         for c, i in enumerate(Batch_ids):
#             case_path = os.path.join(TRAIN_DATASET_PATH, i)

#             data_path = os.path.join(case_path, f'{i}_flair.nii')
#             flair = nib.load(data_path).get_fdata()    
#             flair = self.augment_transform(flair)

#             data_path = os.path.join(case_path, f'{i}_t1ce.nii')
#             ce = nib.load(data_path).get_fdata()
#             ce = self.augment_transform(ce)
            
#             data_path = os.path.join(case_path, f'{i}_seg.nii')
#             seg = nib.load(data_path).get_fdata()
#             seg = self.augment_transform(seg)
        
#             for j in range(VOLUME_SLICES):
#                  X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START], (IMG_SIZE, IMG_SIZE))
#                  X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START], (IMG_SIZE, IMG_SIZE))

#                  y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START]
                    
#         # Generate masks
#         y[y==4] = 3
#         mask = tf.one_hot(y, 4)
#         Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
#         return X/np.max(X), Y


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        if shuffle:
            self.augment_shuffle()

        self.augmentation_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def augment_shuffle(self):
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def augment_transform(self, data):
        return self.augmentation_generator.random_transform(data)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()
            flair = self.augment_transform(flair)
            flair = resize(flair, self.dim)

            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            ce = nib.load(data_path).get_fdata()
            ce = self.augment_transform(ce)
            ce = resize(ce, self.dim)

            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()
            seg = self.augment_transform(seg)
            seg = resize(seg, self.dim)

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :, 0] = flair[:, :, j + VOLUME_START]
                X[j + VOLUME_SLICES * c, :, :, 1] = ce[:, :, j + VOLUME_START]

                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START]

        # Generate masks
        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, self.dim)
        return X / np.max(X), Y