
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time

import os
import util.dataprocessing
dp = util.dataprocessing.DataPlotter()

base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

class_num = 15


# for num_samples in (25,5,2,1):
#
#     y_pred = np.load("{}output/y_tests/{}-samples{}.npy".format(base_dir_path,class_num,num_samples))
#
#     dp.plotConfusionMatrix(labels_test,y_pred)

y_test = np.load("{}output/y_tests/class{}.npy".format(base_dir_path,class_num))
for num_samples in (1,2,5,25):
    y_pred = np.load("{}output/model-19x1/class-15/layers1-samples{}/y_predictions.npy".format(base_dir_path,num_samples))
    dp.plotConfusionMatrix(y_test,y_pred,"{}output/model-19x1/class-15/layers1-samples{}/conf-matr".format(base_dir_path,num_samples))

