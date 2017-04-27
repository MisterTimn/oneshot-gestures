
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class

import os
import util.dataprocessing
dp = util.dataprocessing.DataPlotter()

base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

class_num = 15
load = load_class.load(15)

x_test, labels_test, indices_test = load.load_testing_set()

for num_samples in (25,5,2,1):

    y_pred = np.load("{}output/y_tests/{}-samples{}.npy".format(base_dir_path,class_num,num_samples))

    dp.plotConfusionMatrix(labels_test,y_pred)