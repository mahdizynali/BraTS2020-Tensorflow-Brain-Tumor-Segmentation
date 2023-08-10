from config import *
import keras.backend as K
from AttentionUnet import attUnet
from utils.dataGenerator import *
from keras.callbacks import CSVLogger
from tensorflow.keras import layers, callbacks
from utils.coEFFMatrix import machinLearningMatrix as ml

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import classification_report
# from tensorflow.keras.models import *


training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

input_layer = layers.Input((IMG_SIZE, IMG_SIZE, 2))

model = attUnet(input_layer, 'he_normal', 0.2)
print(type(model))
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=4),
        ml.dice_coef,
        ml.precision,
        ml.sensitivity,
        ml.specificity,
        ml.dice_coef_necrotic,
        ml.dice_coef_edema,
        ml.dice_coef_enhancing
    ]
)

csv_logger = CSVLogger('training.log', separator=',', append=False)
cbacks = [
#     callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
#     callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1), csv_logger]

K.clear_session()

history =  model.fit(training_generator, epochs=30, steps_per_epoch=200,
                     callbacks= cbacks, validation_data = valid_generator)  

model.save("result-model.h5")
