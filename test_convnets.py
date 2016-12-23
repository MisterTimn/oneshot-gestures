
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class
print("Importing convnet")
import convnet_oneshot
import os

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

def getParamOneshotPath(class_num,number_of_layers_retrained):
    return "/home/jasper/oneshot-gestures/convnet_params/oneshot/alpha-001/param-oneshot-class{}-layers-{}".format(class_num,number_of_layers_retrained)

def getParamExcludingPath(class_num):
    return "/home/jasper/oneshot-gestures/convnet_params/excluding/param-excl-class-{}".format(class_num)

def filterFalsePredictions(targets, predictions):
    assert(len(targets)==len(predictions))
    print("{:5}-->{:5}".format("Label","Predict"))
    for i in xrange(len(targets)):
        if(targets[i] != predictions[i]):
            print("{:5}-->{:5}".format(targets[i],predictions[i]))

def main():
    convnet = convnet_oneshot.convnet_oneshot()
    load = load_class.load()
    x_test, labels_test, indices_test = load.load_testing_set()

    for class_num in [17]:

        # convnet.load_param_values(getParamExcludingPath(class_num))
        # test_err = 0
        # test_acc = 0
        # test_batches = 0
        # for batch in iterate_minibatches(x_test, labels_test, 100):
        #     inputs, targets = batch
        #     filterFalsePredictions(targets,convnet.test_output(inputs))
        #     err, acc = convnet.validate(inputs, targets)
        #     test_err += err
        #     test_acc += acc
        #     test_batches += 1
        # print("\ttest-acc:{:7.3f}%".format(test_acc / test_batches * 100))

        convnet.load_param_values(getParamOneshotPath(class_num,2))
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(x_test, labels_test, 100):
            inputs, targets = batch
            filterFalsePredictions(targets, convnet.test_output(inputs))
            err, acc = convnet.validate(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("\ttest-acc:{:7.3f}%".format(test_acc / test_batches * 100))


if __name__ == "__main__":
    main()