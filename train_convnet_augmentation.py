from __future__ import print_function
import augmentation as aug
import load_class
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import pylab
import pickle
import time
import numpy as np
import multiprocessing as mp
import SharedArray as sa
import convnet
import sys

augmenter = aug.augmenter()
loader = load_class.load()
samples, labels, oneshot_indices_test = loader.load_training_set()
x_validate, labels_validate, indices_validate = loader.load_validation_set()
x_test, labels_test, indices_test = loader.load_testing_set()

convnet = convnet.convnet(num_output_units=20)
base_dir_path = "/home/jasper/oneshot-gestures/"
save_param_path = "{}convnet_params/param-augmented2".format(base_dir_path)

try:
    fo1 = open("{}output/augmentation.csv".format(base_dir_path), "w")
    fo1.write("training_loss;validation_loss;validation_accuracy;epoch_time\n")
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error")
    raise

num_classes = 20
oneshot_class = -1
num_oneshot_samples = 200
batch_size=20
minvalerr=20
max_batch_iterations = num_classes * num_oneshot_samples / batch_size

def worker(q):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")
    start_index=0
    indices = []

    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'epoch':
            indices = []
            for gesture_class in xrange(num_classes):
                if (gesture_class == oneshot_class):
                    indices.extend[oneshot_indices_test[gesture_class][:num_oneshot_samples]]
                else:
                    np.random.shuffle(oneshot_indices_test[gesture_class])
                    indices.extend(oneshot_indices_test[gesture_class][:num_oneshot_samples])
            np.random.shuffle(indices)
            start_index=0
        if ((cmd == 'batch' or cmd == 'epoch') and start_index < len(indices)):
            excerpt = indices[start_index:start_index+batch_size]
            images = augmenter.transfMatrix(samples[excerpt])
            np.copyto(sharedSampleArray,images)
            np.copyto(sharedLabelArray,labels[excerpt])
            start_index+=batch_size
        q.task_done()

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices=np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def validate(min_val_err):
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(x_validate, labels_validate, batch_size, True):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
        if (val_err < min_val_err):
            min_val_err = val_err
            convnet.save_param_values(save_param_path)
    print("val err:\t{:5.2f}\nval acc:\t{:5.1f}%".format(val_err / val_batches,
                                                      100.0 * val_acc / val_batches ))
    fo1.write("%.6f;%.6f;%.6f;%.6f\n" % (train_err / train_batches,
                                         val_err / val_batches,
                                         val_acc / val_batches,
                                         time.time() - start_time))

def test():
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, labels_test, 100, shuffle=False):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("test-acc:{:7.2f}%".format(test_acc / test_batches * 100))

if __name__=='__main__':
    try:

        q = mp.JoinableQueue()
        proc = mp.Process(target=worker, args=[q])
        proc.daemon = True
        proc.start()

        sharedSampleArray   = sa.create("shm://samples", (batch_size, 12, 64, 64), dtype='float32')
        sharedLabelArray    = sa.create("shm://labels", batch_size, dtype='int32')

        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')

        #convnet.load_param_values(save_param_path)
        num_epochs = 100
        for j in xrange(num_epochs):
            #Initialise new random permutation of data
            #And load first batch of augmented samples
            q.put('epoch')
            end_epoch = False
            train_err = 0
            train_batches=0
            batch_iteration = 0
            start_time = time.time()

            print("\t--- EPOCH {} ---".format(j+1))

            #wait for data
            q.join()
            while batch_iteration < max_batch_iterations:
                # Data kopieren om daarna te starten met augmentatie volgende batch
                np.copyto(sample_batch,sharedSampleArray)
                np.copyto(label_batch,sharedLabelArray)
                q.put('batch')

                #trainen op de gekopieerde data
                train_err += convnet.train(sample_batch, label_batch)
                train_batches += 1
                print("\rtrain err:\t{:5.2f}".format(train_err / train_batches), end="");
                sys.stdout.flush()

                batch_iteration+=1
                print("\t{:5.0f}%".format(100.0 * batch_iteration / max_batch_iterations), end="");sys.stdout.flush()

                #wachten op nieuwe data
                q.join()
            print()
            validate(minvalerr)

        q.put('done')

        convnet.load_param_values(save_param_path)
        test()

    except:
        raise
    finally:
        q.put('done')
        sa.delete("samples")
        sa.delete("labels")
        fo1.close()
    print("End of program")