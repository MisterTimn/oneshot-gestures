
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time

import os
import util.dataprocessing
dp = util.dataprocessing.DataPlotter()


BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_VERS      =   "model-18x2"
MODEL_EXCLUDING =   "model-18"
ONESHOT_CLASS   =   14
ONESHOT_CLASS_2 =   15

OUTPUT_DIRECTORY=   "{}output/{}/class-{}-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS,ONESHOT_CLASS_2)
PARAM_DIRECTORY =   "{}convnet_params/{}/class-{}-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS,ONESHOT_CLASS_2)
EXCLUDING_PARAM_PATH   \
                =   "{}convnet_params/{}/excluding-{}".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS)



# for num_samples in (25,5,2,1):
#
#     y_pred = np.load("{}output/y_tests/{}-samples{}.npy".format(base_dir_path,class_num,num_samples))
#
#     dp.plotConfusionMatrix(labels_test,y_pred)


x_labels = (1,2,3,4,5,25,100,200)
y_test = np.load("{}output/y_tests/class-14-15.npy".format(BASE_DIR))
y_predictions = np.empty((len(x_labels),len(y_test)))

i = 0
for num_samples in x_labels:
    y_pred = np.load("{}layers1-samples{}/y_predictions.npy".format(OUTPUT_DIRECTORY,num_samples))
    # dp.plotConfusionMatrix(y_test,y_pred,
    #                        "{}: L1 S{}".format(MODEL_VERS,num_samples),
    #                        "{}layers1-samples{}/conf-matr".format(OUTPUT_DIRECTORY,num_samples))

    y_predictions[i] = y_pred
    i+=1

dp.plotDoubleClassF1(y_test,y_predictions,x_labels,18,19,"Model 18x2 - L1")
#
# x_labels = (1,2,3,4,5,25,100,200)
# y_test = np.load("{}output/y_tests/class-14-15.npy".format(BASE_DIR))
# y_predictions = np.empty((len(x_labels),len(y_test)))

# x_labels = (1,2,3,4,5,200)
# y_test = np.load("{}output/y_tests/class14.npy".format(BASE_DIR))
# y_predictions = np.empty((len(x_labels),len(y_test)))
#
# i = 0
# for num_samples in x_labels:
#     y_pred = np.load("{}layers1-samples{}/y_predictions.npy".format(OUTPUT_DIRECTORY,num_samples))
#     # dp.plotConfusionMatrix(y_test,y_pred,
#     #                        "{}: L1 S{}".format(MODEL_VERS,num_samples),
#     #                        "{}layers1-samples{}/conf-matr".format(OUTPUT_DIRECTORY,num_samples))
#
#     y_predictions[i] = y_pred
#
#     i+=1
#
# dp.plotAccF1(y_test,y_predictions,x_labels,19,"Model 19x1 - L2")