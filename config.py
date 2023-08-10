import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import optimizers, metrics

NUM_CLASSES=4
IMG_SIZE=128
TRAIN_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData/'
SAVE_MODEL_PATH = 'result-model.h5'
SAVE_LOG_PATH = 'training.log'

# if they are existed before:
PRE_TRAINED_MODEL_PATH = '/home/mahdi/Desktop/tf2/first_train/model-agust.h5'
PRE_TRAINED_LOG_PATH = '/home/maximum/Desktop/tf2/first_train/training.log'


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
    # lr_schedule = optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # optimizer = optimizers.SGD(learning_rate=lr_schedule)
    IoU = metrics.MeanIoU(num_classes = NUM_CLASSES)