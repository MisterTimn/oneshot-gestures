
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

false_positives = np.zeros(20, dtype=np.int)
errors = np.zeros(20, dtype=np.int)
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

def filterFalsePredictions(targets, predictions):
    assert(len(targets)==len(predictions))
    global total_errors

    for i in xrange(len(targets)):
        total[targets[i]]+=1
        if(targets[i] != predictions[i]):
            total_errors += 1
            errors[targets[i]] += 1
            false_positives[predictions[i]] += 1

def main():
    convnet = cnn.convnet_oneshot()
    load = load_class.load()
    x_test, labels_test, indices_test = load.load_testing_set()
    # x_test, labels_test, indices_test = load.load_validation_set()



    class_num = 19
    global errors
    global total
    global false_positives
    global total_errors

    base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__)))  # "/home/jasper/oneshot-gestures/

    with open("{}output/processed/test-results-19".format(base_dir_path),'w') as out_f:
        out_f.write("layers")
        out_f.write("samples")
        out_f.write("test-acc")
        for class_index in xrange(20):
            out_f.write("{}".format(class_index))
        for class_index in xrange(20):
            out_f.write("FP {}".format(class_index))

    for num_samples in [200,100,50,25,10,5,2,1]:
        for num_layers_retrained in [3,2,1]:
        # num_layers_retrained = 3
        # for num_samples in [200,100,50,25,10]:

            errors = np.zeros(20, dtype=np.int)
            total = np.zeros(20, dtype=np.int)
            false_positives = np.zeros(20, dtype=np.int)
            total_errors=0
            # convnet.load_param_values(getParamPath(class_num, num_layers_retrained, num_samples))
            convnet.load_param_values(getParamExcludingPath(class_num))
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
            print("\nclass-{}-retrain-{}-samples-{}".format(class_num,num_layers_retrained,num_samples))
            # print("class\terror\tfalse pos")
            # for class_index in range(20):
            #     print("{:2}:\t{:5.2f}%\t{:5.2f}%"
            #           .format(class_index,
            #                   100.0*errors[class_index] / total[class_index],
            #                   100.0*false_positives[class_index] / total_errors))


        with open("{}output/processed/test-results-19".format(base_dir_path),'r+') as out_f:
            lines = in_f.readlines()
            out_f.write("{};{}".format(lines[0],num_layers_retrained))
            out_f.write("{};{}".format(lines[1],num_samples))
            out_f.write("{};{}".format(lines[2],test_acc / test_batches))
            for class_index in range(20):
                out_f.write("{};{}".format(lines[3+class_index],1 - (errors[class_index]/total[class_index])))
            for class_index in range(20):
                out_f.write("{};{}".format(lines[23+class_index],1 - (false_positives[class_index]/total_errors)))

            print("TEST-ACC:{:7.3f}%".format(test_acc / test_batches * 100))




if __name__ == "__main__":
    main()