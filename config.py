from tensorflow.keras import optimizers, metrics
import warnings
warnings.filterwarnings('ignore')

NUM_CLASSES=4
IMG_SIZE=128
TRAIN_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData/'
PRE_TRAINED_MODEL_PATH = '/home/mahdi/Desktop/tf2/first_train/model-agust.h5'
PRE_TRAINED_LOG_PATH = '/home/maximum/Desktop/tf2/first_train/training.log'
SAVE_MODEL_PATH = 'result-model.h5'
SAVE_LOG_PATH = 'training.log'


class HyperParameters:
    lossFunction = "categorical_crossentropy"
    learningRate = 0.001
    batchSize = 16
    epochs = 30
    steps = 200 # trainSize / batchSize
    modelKernel = 'he_normal'
    modelDropout = 0.2
    metricsBase = 'accuracy'
    optimizer = optimizers.Adam(learningRate)
    IoU = metrics.MeanIoU(num_classes = NUM_CLASSES)