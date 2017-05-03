
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time

import os
import util.dataprocessing
dp = util.dataprocessing.DataPlotter()



base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_VERS      =   "model-19x1-temp"
MODEL_EXCLUDING =   "model-19"
ONESHOT_CLASS   =   15

OUTPUT_DIRECTORY=   "{}output/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
PARAM_DIRECTORY =   "{}convnet_params/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
EXCLUDING_PARAM_PATH   \
                =   "{}convnet_params/{}/excluding-{}".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS)

x_labels = (1,2,3,4,5,25,100,200)
y_test = np.load("{}output/y_tests/class{}.npy".format(base_dir_path,ONESHOT_CLASS))
y_predictions = np.empty((len(x_labels),len(y_test)))

i = 0
for num_samples in x_labels:
    y_predictions[i] = np.load("{}layers1-samples{}/y_predictions.npy".format(OUTPUT_DIRECTORY,num_samples))
    # dp.plotConfusionMatrix(y_test,y_predictions[i],"{}layers1-samples{}/conf-matr".format(OUTPUT_DIRECTORY,num_samples))
    i+=1


dp.plotAccF1(y_test,y_predictions,x_labels,19,"Learning gesture 15, retrain level 1")
