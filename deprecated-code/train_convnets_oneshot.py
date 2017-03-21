
from __future__ import print_function
print("start imports")
import sys
import numpy as np
import time
print("Importing load_class")
import load_class
print("Importing convnet")
import convnet_oneshot

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

def iterate_minibatches_oneshotv2(inputs, targets, oneshot_indices, oneshot_class, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    i = 0
    indices=[]
    for class_num in xrange(20):
        indices.extend(oneshot_indices[class_num][:20])
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]





base_dir_path = "/home/jasper/oneshot-gestures/"
test_accuracies = []

load = load_class.load(size_ratio=1.0)
# Load data
x_validate, labels_validate, indices_validate = load.load_validation_set()
x_train, labels_train, indices_train = load.load_training_set()
x_test, labels_test, indices_test = load.load_testing_set()
for num_layers in [1,2,3]:
    convnet = convnet_oneshot.convnet_oneshot(num_output_units=20, num_layers_retrain=num_layers)
    #convnet.save_param_values("{}/default_param".format(base_dir_path))

    # [2,8,9,18]
    for oneshot_class in [0,2,4,8,9,14,17,18]:
        print("Oneshotting class {}".format(oneshot_class))

        save_param_path = "{}convnet_params/param-oneshot-class{}-layers-{}".format(base_dir_path, oneshot_class, num_layers)

        # Load trained model excluding the class we want to oneshot
        convnet.load_param_values("{}convnet_params/excluding/param-excl-class-{}".format(base_dir_path,oneshot_class))
        try:
            fo1 = open("{}output/oneshot-{}-layers-{}.csv".format(base_dir_path,oneshot_class,num_layers),"w")
            fo1.write("training_loss;validation_loss;validation_accuracy;epoch_time\n")
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:
            print("Unexpected error")
            raise

        # Init sizes for correct % output
        train_size = len(labels_train) * 1.0
        validate_size = len(labels_validate) * 1.0
        test_size = len(labels_test) * 1.0

        batch_size = 20
        min_val_err = 20
        num_epochs = 50
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches_oneshotv2(x_train,labels_train,
                                             indices_train, oneshot_class,
                                             batch_size,True):
                inputs, targets = batch
                train_err += convnet.train(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(x_validate,labels_validate,batch_size,True):
                inputs, targets = batch
                err, acc = convnet.validate(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
                if( val_err < min_val_err ):
                    min_val_err = val_err
                    convnet.save_param_values(save_param_path)

            print('\rEpoch {} / {}\tval acc:{:5.2f}\ttime:{:5.2f}'.format(epoch+1, num_epochs, val_acc / val_batches, time.time() - start_time), end="");sys.stdout.flush()
            fo1.write("%.6f;%.6f;%.6f;%.6f\n" % (train_err / train_batches,
                                                 val_err / val_batches,
                                                 val_acc / val_batches,
                                                 time.time() - start_time))
        convnet.load_param_values(save_param_path)
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(x_test, labels_test, 100, shuffle=False):
            inputs, targets = batch
            err, acc = convnet.validate(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("\ttest-acc:{:7.3f}%".format(test_acc / test_batches * 100))
        test_accuracies.append(test_acc / test_batches * 100)

        #convnet.load_param_values("{}/default_param".format(base_dir_path))

print(test_accuracies)
fo1.close()