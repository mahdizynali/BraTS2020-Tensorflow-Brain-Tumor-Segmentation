from config import HyperParameters, IMG_SIZE
import keras.backend as K
from AttentionUnet import attUnet
from utils.dataGenerator import *
from keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
from utils.coEFFMatrix import machinLearningMatrix as ml
hyper = HyperParameters()

training_generator = DataGenerator(train_ids, batch_size=hyper.batchSize, n_channels=2)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

model = attUnet((IMG_SIZE, IMG_SIZE, 2), hyper.modelKernel, hyper.modelDropout).generateModel()
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
#     callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
#     callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1), csv_logger]

K.clear_session()

history =  model.fit(training_generator, epochs=hyper.epochs, steps_per_epoch=hyper.steps,
                     callbacks= cbacks, validation_data = valid_generator)  

model.save(SAVE_MODEL_PATH)