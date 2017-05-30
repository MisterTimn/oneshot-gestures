
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class
print("Importing convnet")
import convnet as cnn
import os

from sklearn import metrics
from util.dataprocessing import DataPlotter as dp


base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/


# O(n)
# Return mini batches, dynamically excluding the indices of the oneshot class
def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices=np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def getParamPath(class_num, number_of_layers_retrained, num_samples):
    return "{}convnet_params/model-19x1/class-{}/layers{}-samples{}"\
        .format(base_dir_path,class_num,number_of_layers_retrained,num_samples)

def getParamExcludingPath(class_num):
    return "{}convnet_params/model-19/excluding-{}".format(base_dir_path,class_num)

def main():
    class_num = 15
    BASE_DIR = "{}/".format(os.path.dirname(os.path.abspath(__file__)))

    load = load_class.load(19)
    x_test, labels_test, indices_test = load.load_testing_set()
    convnet = cnn.convnet()

    convnet.load_param_values(
        "{}convnet_params/naive_model/param-excl{}".format(BASE_DIR,class_num))

    y_pred = convnet.test_output(x_test)
    print(metrics.classification_report(labels_test, y_pred))

    np.save("{}output/y_pred_naive/y_predictions-0".format(BASE_DIR),y_pred)

    for num_samples in [1,2,5,10,25,50,100,200]:
        for num_layers_retrained in [2]:
            convnet.load_param_values("{}convnet_params/naive_model/param-oneshot{}-layers{}-samples{}"\
                .format(BASE_DIR,class_num, num_layers_retrained, num_samples))
            y_pred = convnet.test_output(x_test)
            print(metrics.classification_report(labels_test,y_pred))

            np.save("{}output/y_pred_naive/y_predictions-{}".format(BASE_DIR,num_samples), y_pred)



if __name__ == "__main__":
    main()