
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class
print("Importing convnet")
import convnet_v2_oneshot as cnn
import os
from util.dataprocessing import DataPlotter as dp

false_positives = np.zeros(20, dtype=np.int)
positives = np.zeros(20, dtype=np.int)
total = np.zeros(20, dtype=np.int)

total_errors = 0

class_accuracies = np.zeros(20, dtype=np.int)

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
    return "{}convnet_params/param-oneshot{}-layers{}-samples{}"\
        .format(base_dir_path,class_num,number_of_layers_retrained,num_samples)

def getParamExcludingPath(class_num):
    return "{}convnet_params/param-excl{}".format(base_dir_path,class_num)

def main():

    load = load_class.load(15)
    x_test, labels_test, indices_test = load.load_testing_set()

    class_num = 15
    global positives
    global total
    global false_positives
    global total_errors

    base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__)))  # "/home/jasper/oneshot-gestures/

    y_tests = np.zeros(len(labels_test))
    for num_samples in [25,5,2,1]:
        for num_layers_retrained in [1]:
            convnet = cnn.convnet_oneshot()
            convnet.load_param_values(getParamPath(class_num, num_layers_retrained, num_samples))
            # convnet.load_param_values(getParamExcludingPath(class_num))

            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(x_test, labels_test, 100):
                inputs, targets = batch
                convnet.test_output(inputs)
                err, acc = convnet.validate(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1

            print("\nclass-{}-retrain-{}-samples-{}".format(class_num,num_layers_retrained,num_samples))
            print("TEST-ACC:{:7.3f}%".format(test_acc / test_batches * 100))

            if not os.path.exists("{}/output/y_tests/".format(base_dir_path)):
                os.makedirs("{}/output/y_tests/".format(base_dir_path))

            y_tests = convnet.test_output(x_test)

            np.save("{}/output/y_tests/{}-samples{}".format(base_dir_path,class_num,num_samples),y_tests)




if __name__ == "__main__":
    main()