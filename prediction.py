import pandas as pd
import tensorflow as tf
from keras.models import load_model
from utils.coEFFMatrix import machinLearningMatrix as ml
from config import PRE_TRAINED_LOG_PATH, PRE_TRAINED_MODEL_PATH

# test_set = DataGenerator(test_ids, batch_size=hyper.batchSize, n_channels=2, shuffle=True)

model = load_model(PRE_TRAINED_MODEL_PATH, 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": ml.dice_coef,
                                                   "precision": ml.precision,
                                                   "sensitivity":ml.sensitivity,
                                                   "specificity":ml.specificity,
                                                   "dice_coef_necrotic": ml.dice_coef_necrotic,
                                                   "dice_coef_edema": ml.dice_coef_edema,
                                                   "dice_coef_enhancing": ml.dice_coef_enhancing
                                                  }, compile=False)

history = pd.read_csv(PRE_TRAINED_LOG_PATH, sep=',', engine='python')
hist=history