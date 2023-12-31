import keras.backend as K
from AttentionUnet import testUnet, simpleUnet2, attUnet
from utils.dataGenerator import *
from keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
from utils.coEFFMatrix import machinLearningMatrix as ml
from utils.plotting import trainingResults
from config import HyperParameters, IMG_SIZE, SAVE_LOG_PATH, SAVE_MODEL_PATH
hyper = HyperParameters()

train_set = DataGenerator(train_ids, batch_size=hyper.batchSize, n_channels=2, augment=False, shuffle=True)
valid_set = DataGenerator(val_ids, batch_size=hyper.batchSize, n_channels=2, augment=False, shuffle=True)


# from keras.models import load_model
# model = load_model("archive/pre.h5", 
#     custom_objects={ 'accuracy' : HyperParameters.IoU,
#         "dice_coef": ml.dice_coef,
#         "precision": ml.precision,
#         "sensitivity":ml.sensitivity,
#         "specificity":ml.specificity,
#         "dice_coef_necrotic": ml.dice_coef_necrotic,
#         "dice_coef_edema": ml.dice_coef_edema,
#         "dice_coef_enhancing": ml.dice_coef_enhancing
# }, compile=False)

model = simpleUnet2((IMG_SIZE, IMG_SIZE, 2), hyper.modelKernel, hyper.modelDropout).generateModel()
model.compile(
    loss=hyper.lossFunction,
    optimizer=hyper.optimizer,
    metrics=[
        hyper.metricsBase,
        hyper.IoU,
        ml.dice_coef,
        ml.precision,
        ml.sensitivity,
        ml.specificity,
        ml.dice_coef_necrotic,
        ml.dice_coef_edema,
        ml.dice_coef_enhancing
    ]
)

csv_logger = CSVLogger(SAVE_LOG_PATH, separator=',', append=False)
cbacks = [
    callbacks.EarlyStopping(monitor='loss', min_delta=0,
                              patience=2, verbose=1, mode='auto'),
    callbacks.ModelCheckpoint(filepath = 'checkpoints/checkpoint_.{epoch:02d}-{val_loss:.6f}.h5',
                            verbose=1, save_best_only=True, save_weights_only = True, period = 5),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1), csv_logger
]

K.clear_session()

history =  model.fit(train_set, epochs=hyper.epochs, steps_per_epoch=hyper.steps,
                     callbacks= cbacks, validation_data = valid_set)  

model.save(SAVE_MODEL_PATH)
# trainingResults(history, "trainingModelResult")