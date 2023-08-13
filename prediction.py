import pandas as pd
from config import *
from utils.dataGenerator import *
from keras.models import load_model
from utils.plotting import trainingResults, prediction
from utils.coEFFMatrix import machinLearningMatrix as ml

testSet = DataGenerator(test_ids, batch_size=HyperParameters.batchSize, n_channels=2, shuffle=True)

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
trainingResults(history, "predictionModelResult")
prediction(model, test_ids, caseNumber=3, start_slices=40) # TODO = random dataset initiate