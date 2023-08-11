import os
import cv2
from config import TRAIN_DATASET_PATH, IMG_SIZE
from sklearn.model_selection import train_test_split
import tensorflow as tf
import nibabel as nib
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from keras.utils import Sequence
from skimage.transform import rotate, warp, AffineTransform
from skimage.util import random_noise

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100
VOLUME_START = 20 # first slice of volume that we will include

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


class DataGenerator(Sequence):
    'Generates and augment data for training'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def augmentation(self, image):
        # Apply rotation
        angle = np.random.uniform(-20, 20)
        image = rotate(image, angle, mode='reflect', preserve_range=True)
        
        # Apply affine transformation for shear and zoom
        transform = AffineTransform(
            rotation=np.deg2rad(angle),
            shear=np.random.uniform(-0.2, 0.2),
            scale=(1 + np.random.uniform(-0.2, 0.2), 1 + np.random.uniform(-0.2, 0.2))
        )
        image = warp(image, transform.inverse, mode='reflect', preserve_range=True)

        # Apply random noise
        image = random_noise(image, var=0.01)

        # Apply horizontal flip
        if np.random.rand() < 0.5:
            image = np.fliplr(image)

        return image


    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                augmented_flair = self.augmentation(flair[:, :, j + VOLUME_START])
                augmented_ce = self.augmentation(ce[:, :, j + VOLUME_START])

                X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(augmented_flair, (IMG_SIZE, IMG_SIZE))
                X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(augmented_ce, (IMG_SIZE, IMG_SIZE))

                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START]

        # Generate masks
        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return X / np.max(X), Y