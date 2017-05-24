from __future__ import print_function

import SharedArray as sa
import multiprocessing as mp
import numpy as np
import sys
import time
import os

import augmentation as aug
import convnet_19 as cnn
import load_class

from sklearn import metrics

augmenter = aug.augmenter()

BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_EXCLUDING =   "model-19"

PARAM_PATH   \
                =   "{}convnet_params/{}/".format(BASE_DIR,MODEL_EXCLUDING)
if not os.path.exists(PARAM_PATH):
    os.makedirs(PARAM_PATH)


TOTAL_BACKPROPS = 30000
BACKPROPS_PER_EPOCH = 1000
NUM_EPOCHS = TOTAL_BACKPROPS / BACKPROPS_PER_EPOCH

NUM_CLASSES = 20
BATCH_SIZE = 32

def worker_backprop(q,samples,labels,indices_train):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")
    indices = np.empty(BATCH_SIZE, dtype='int32')

    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'batch':
            classes = np.random.randint(NUM_CLASSES - 1, size=BATCH_SIZE)
            for i in xrange(BATCH_SIZE):
                indices[i] = indices_train[classes[i]][np.random.randint(len(indices_train[classes[i]]))]
            np.copyto(sharedSampleArray,augmenter.transfMatrix(samples[indices]))
            np.copyto(sharedLabelArray,labels[indices])
        q.task_done()

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices=np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def validate(convnet,x_validate,labels_validate):
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(x_validate, labels_validate, BATCH_SIZE, True):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    return val_err/val_batches, val_acc/val_batches

def test(convnet,x_test,labels_test):
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, labels_test, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    return test_acc / test_batches

if __name__=='__main__':
    try:
        sharedSampleArray = sa.create("shm://samples", (BATCH_SIZE, 12, 64, 64), dtype='float32')
        sharedLabelArray = sa.create("shm://labels", BATCH_SIZE, dtype='int32')

        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')

        for ONESHOT_CLASS in [0,7,8]:

            EXCLUDING_PARAM_PATH \
                = "{}convnet_params/{}/excluding-{}".format(BASE_DIR, MODEL_EXCLUDING, ONESHOT_CLASS)

            loader = load_class.load(ONESHOT_CLASS)
            print(loader.get_oneshot())

            samples, labels, indices_train = loader.load_training_set()
            x_validate_orig, labels_validate_orig, indices_validate = loader.load_validation_set()
            x_test_orig, labels_test_orig, indices_test = loader.load_testing_set()

            val_indices_to_keep = indices_validate[0]
            test_indices_to_keep = indices_test[0]
            for i in xrange(1, 19):
                val_indices_to_keep = np.concatenate((val_indices_to_keep, indices_validate[i]), axis=0)
                test_indices_to_keep = np.concatenate((test_indices_to_keep, indices_test[i]), axis=0)

            np.random.shuffle(val_indices_to_keep)
            np.random.shuffle(test_indices_to_keep)

            x_validate = np.empty((len(val_indices_to_keep),12,64,64),dtype='float32')
            labels_validate = np.empty(len(val_indices_to_keep),dtype='int32')
            x_test = np.empty((len(test_indices_to_keep),12,64,64),dtype='float32')
            labels_test = np.empty(len(test_indices_to_keep),dtype='int32')

            np.copyto(x_validate,x_validate_orig[val_indices_to_keep])
            np.copyto(labels_validate,labels_validate_orig[val_indices_to_keep])
            np.copyto(x_test,x_test_orig[test_indices_to_keep])
            np.copyto(labels_test, labels_test_orig[test_indices_to_keep])

            min_val_err = 20
            val_loss = 20
            val_acc = 0
            last_improvement = 0

            convnet = cnn.convnet(num_output_units=19)

            try:
                q = mp.JoinableQueue()
                proc = mp.Process(target=worker_backprop, args=[q, samples, labels, indices_train])
                proc.daemon = True
                proc.start()

                for j in xrange(NUM_EPOCHS):
                    #Initialise new random permutation of data
                    #And load first batch of augmented samples
                    q.put('batch')
                    end_epoch = False
                    train_err = 0
                    train_batches=0
                    start_time = time.time()

                    #wait for data
                    q.join()
                    for i in xrange(BACKPROPS_PER_EPOCH):
                        # Data kopieren om daarna te starten met augmentatie volgende batch
                        np.copyto(sample_batch,sharedSampleArray)
                        np.copyto(label_batch,sharedLabelArray)

                        q.put('batch')

                        #trainen op de gekopieerde data
                        train_err += convnet.train(sample_batch, label_batch)
                        train_batches += 1
                        print("\rBP {} - {} ({}):  ".format(j * BACKPROPS_PER_EPOCH + 1,
                                                    j * BACKPROPS_PER_EPOCH + BACKPROPS_PER_EPOCH,
                                                    last_improvement),end="")
                        print("train err: {:5.2f} val acc: {:5.2f}".format(train_err / i,val_acc), end="");sys.stdout.flush()
                        print("   {:5.0f}%".format(100.0 * (i+1) / BACKPROPS_PER_EPOCH), end="");sys.stdout.flush()

                        q.join()
                    train_loss = train_err / BACKPROPS_PER_EPOCH
                    val_loss, val_acc = validate(convnet,x_validate,labels_validate)


                    if (val_loss < min_val_err):
                        min_val_err = val_loss
                        convnet.save_param_values(EXCLUDING_PARAM_PATH)
                        last_improvement=0
                    else:
                        last_improvement+=1


            except KeyboardInterrupt:
                print("Iteration stopped through KeyboardInterrupt")
            except:
                raise
            finally:
                q.put('done')
                if os.path.exists(EXCLUDING_PARAM_PATH):
                    convnet.load_param_values(EXCLUDING_PARAM_PATH)

                test_acc = test(convnet, x_test, labels_test)
                print("test-acc:{:5.2f}%".format(test_acc * 100))

                OUTPUT_DIRECTORY = "{}output/{}/excluding-{}/".format(BASE_DIR, MODEL_EXCLUDING, ONESHOT_CLASS)

                if not os.path.exists(OUTPUT_DIRECTORY):
                    os.makedirs(OUTPUT_DIRECTORY)

                y_predictions = convnet.test_output(x_test)
                np.save("{}y_predictions".format(OUTPUT_DIRECTORY), y_predictions)

                if not os.path.exists("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS)):
                    open("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS), 'w').close()
                with open("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS), 'ab') as f:
                    f.write(metrics.classification_report(labels_test, y_predictions))

    except:
        raise
    finally:
        sa.delete("samples")
        sa.delete("labels")
    print("End of program")
