from config import *
from utils.coEFFMatrix.machinLearningMatrix import *


input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

model = build_unet_with_attention(input_layer, 'he_normal', 0.2)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=4),
        dice_coef,
        precision,
        sensitivity,
        specificity,
        dice_coef_necrotic,
        dice_coef_edema,
        dice_coef_enhancing
    ]
)


csv_logger = CSVLogger('training.log', separator=',', append=False)
callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1), csv_logger]

K.clear_session()

history =  model.fit(training_generator, epochs=30, steps_per_epoch=200,
                     callbacks= callbacks, validation_data = valid_generator)  

model.save("result-model.h5")
