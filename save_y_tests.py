
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class_2 as load_class

import os


base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

for class_num in xrange(20):
    load = load_class.load(class_num)
    x_test, labels_test, indices_test = load.load_testing_set()
    np.save("{}/output/y_tests/class{}".format(base_dir_path,class_num),labels_test)