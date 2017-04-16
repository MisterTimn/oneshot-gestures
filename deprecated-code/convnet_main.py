from __future__ import print_function
import sys
import numpy as np
import os
import time


data_ratio = 1.0

# O(n)
# Return mini batches, dynamically excluding the indices of the oneshot class
def iterate_minibatches(inputs, targets, oneshot_indices, oneshot_class, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    i = 0
    indices=[]
    for index in np.arange(len(inputs)):
        if ( oneshot_class > -1 and i < len(oneshot_indices[oneshot_class]) and index == oneshot_indices[oneshot_class][i]):
            i+=1
            #print("Index {} overeenkomstig label {}".format(index,targets[index]));sys.stdout.flush()
        else:
            indices.append(index)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]


import load_class
load = load_class.load(data_ratio)

import convnet
convnet = convnet.convnet(20)

import augmentation as aug
augmenter = aug.augmenter()

try:
    i = 0
    path = "/home/jasper/oneshot-gestures/"
    while os.path.exists("{}output/acc-cost_{}.csv".format(path,i)):
        i += 1
    fo1 = open("{}output/acc-cost_{}.csv".format(path,i), "w")
    fo1.write("training_loss;validation_loss;validation_accuracy;epoch_time\n")
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
    raise
except:
    print("unexpected error")
    raise

save_param_path = "{}convnet_params/param_model_20_augmented".format(path)
#convnet.load_param_values("model_parameters/param_model")
print("Loading data")
sys.stdout.flush()
x_validate, labels_validate, oneshot_indices_validate = load.load_validation_set()
x_train, labels_train, oneshot_indices_train = load.load_training_set()
x_test, labels_test, oneshot_indices_test = load.load_testing_set()
print("Data loaded")
sys.stdout.flush()

train_size = len(labels_train) * 1.0
validate_size = len(labels_validate) * 1.0
test_size = len(labels_test) * 1.0
oneshot_class = 4

batch_size = 20
# x=raw_input("Number of epochs to perform: ")
# while(not x.isdigit()):
#     x=raw_input("Not a valid number, again: ")
#
# num_epochs=int(x)
num_epochs = 100
go=True
try:
    while (go):
        min_val_err = 20
        for epoch in range(num_epochs):

            # Full pass over training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(x_train, labels_train,
                                             oneshot_indices_train,
                                             oneshot_class, batch_size,
                                             shuffle=True):
                inputs_pre_aug, targets = batch
                inputs = augmenter.scale_crop(augmenter.rotate_crop(inputs_pre_aug))

                print('\rTraining phase {:6.1f}%'.format(train_batches * batch_size / train_size * 100), end="");sys.stdout.flush()
                train_err += convnet.train(inputs, targets)
                train_batches += 1
            print('\rTraining phase {:6.1f}%'.format(100))

            # Full pass over validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(x_validate, labels_validate,
                                             oneshot_indices_validate,
                                             oneshot_class, batch_size,
                                             shuffle=False):
                inputs, targets = batch
                print('\rValidation phase {:6.1f}%'.format(val_batches * batch_size / validate_size * 100), end="");sys.stdout.flush()
                err, acc = convnet.validate(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
                if( val_err < min_val_err ):
                    min_val_err = val_err
                    convnet.save_param_values(save_param_path)
            print('\rValidation phase {:6.1f}%'.format( 100))

            # Print out results each epoch and write to outputfile
            print("Epoch {} of {} took {:.2f}s".format(	epoch + 1,
                                                        num_epochs,
                                                        time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
            fo1.write("%.6f;%.6f;%.6f;%.6f\n" % (	train_err / train_batches,
                                                    val_err / val_batches,
                                                    val_acc / val_batches,
                                                    time.time() - start_time))
            print()
        # Compute and print test error
        convnet.load_param_values(save_param_path)
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(x_test, labels_test,
                                         oneshot_indices_test,
                                         oneshot_class, 100,
                                         shuffle=False):
            inputs, targets = batch
            err, acc = convnet.validate(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

        x=raw_input("num_epochs or q(uit): ")
        while (not x.isdigit()):
            if (x == 'q' or x == 'quit' or x == 'Q' or x == 'QUIT'):
                go = False
                print("quitting convnet_main")
                break
            x=raw_input('Not a valid number, again: ')
        num_epochs=int(x)

except KeyboardInterrupt:
    print("\nProgram forced to stop via KeyboardInterrupt")
except:
    raise

fo1.close()
