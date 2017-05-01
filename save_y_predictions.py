
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class
print("Importing convnet")
import convnet_19x1 as cnn
import os
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

    load = load_class.load(class_num)
    x_test, labels_test, indices_test = load.load_testing_set()
    convnet = cnn.convnet_oneshot()

    for num_samples in [1,2,5,25,100]:
        for num_layers_retrained in [1]:

            convnet.load_param_values(getParamPath(class_num, num_layers_retrained, num_samples))
            # convnet.load_param_values(getParamExcludingPath(class_num))
            # convnet.load_param_values("{}convnet_params/param-allclasses".format(base_dir_path))

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

            directory = "{}output/model-19x1/class-{}/layers{}-samples{}".format(base_dir_path, class_num,
                                                                                  num_layers_retrained, num_samples)

            with open("{}/test-acc.txt".format(directory),'ab') as f:
                f.write("layers{};samples{};{}".format(num_layers_retrained,num_samples,1.0*test_acc/test_batches))
            if not os.path.exists(directory):
                os.makedirs(directory)

            y_predictions = convnet.test_output(x_test)

            # np.save("{}/output/y_tests/{}-samples{}".format(base_dir_path,class_num,num_samples),y_tests)
            np.save("{}".format(directory),y_predictions)
    # np.save("{}/output/y_tests/y_test".format(base_dir_path),labels_test)


if __name__ == "__main__":
    main()