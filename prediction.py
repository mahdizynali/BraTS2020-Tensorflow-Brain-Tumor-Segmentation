import pandas as pd
from config import *
import tensorflow as tf
from utils.dataGenerator import *
from keras.models import load_model
from utils.coEFFMatrix import machinLearningMatrix as ml

test_set = DataGenerator(test_ids, batch_size=HyperParameters.batchSize, n_channels=2, shuffle=True)

model = load_model(PRE_TRAINED_MODEL_PATH, 
                                   custom_objects={ 'accuracy' : HyperParameters.IoU,
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