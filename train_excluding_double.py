from __future__ import print_function

import SharedArray as sa
import multiprocessing as mp
import numpy as np
import sys
import time
import os
from sklearn import metrics

import augmentation as aug
import convnet_18 as cnn
import load_class
from util.dataprocessing import DataSaver

augmenter = aug.augmenter()

# loader = load_class.load()
#
# samples, labels, indices_train = loader.load_training_set()
# x_validate, labels_validate, indices_validate = loader.load_validation_set()
# x_test, labels_test, indices_test = loader.load_testing_set()

BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_EXCLUDING =   "model-18"
ONESHOT_CLASS   =   14
ONESHOT_CLASS_2 =   15

OUTPUT_DIRECTORY=   "{}output/{}/excluding-{}-{}/".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS,ONESHOT_CLASS_2)
EXCLUDING_PARAM_PATH   \
                =   "{}convnet_params/{}/excluding-{}-{}".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS,ONESHOT_CLASS_2)

loader = load_class.load(ONESHOT_CLASS, ONESHOT_CLASS_2)

samples, labels, indices_train = loader.load_training_set()
x_validate, labels_validate, indices_validate = loader.load_validation_set()
x_test, labels_test, indices_test = loader.load_testing_set()

val_indices_to_keep = indices_validate[0]
test_indices_to_keep = indices_test[0]
for i in xrange(1,18):
    val_indices_to_keep = np.concatenate((val_indices_to_keep,indices_validate[i]),axis=0)
    test_indices_to_keep = np.concatenate((test_indices_to_keep,indices_test[i]),axis=0)

x_validate = x_validate[val_indices_to_keep]
labels_validate = labels_validate[val_indices_to_keep]
x_test = x_test[test_indices_to_keep]
labels_test = labels_test[test_indices_to_keep]

BATCH_SIZE = 32

TOTAL_BACKPROPS = 60000
BACKPROPS_PER_EPOCH = 20
NUM_EPOCHS = TOTAL_BACKPROPS / BACKPROPS_PER_EPOCH
NUM_CLASSES = 20
BATCH_SIZE = 32

def worker_backprop(q):
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
            # classes = np.random.choice(class_choices,batch_size)
            classes = np.random.randint(NUM_CLASSES - 2, size=BATCH_SIZE)
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
        precision_score = np.add(precision_score,
                                 metrics.precision_score(targets, convnet.test_output(inputs), xrange(NUM_CLASSES),
                                                         average=None))
        recall_score = np.add(recall_score,
                              metrics.recall_score(targets, convnet.test_output(inputs), xrange(NUM_CLASSES),
                                                   average=None))
        val_batches += 1
    return val_err / val_batches, val_acc / val_batches, precision_score / val_batches, recall_score / val_batches

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

        min_val_err = 20

        ds = DataSaver(('train_loss', 'val_loss', 'val_acc', 'dt'))

        precision_list = np.zeros((NUM_EPOCHS, NUM_CLASSES))
        recall_list = np.zeros((NUM_EPOCHS, NUM_CLASSES))

        convnet = cnn.convnet()

        try:
            q = mp.JoinableQueue()
            proc = mp.Process(target=worker_backprop, args=[q])
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
                    print("\r{:5.0f}-{:5.0f}:\t{:5.0f}%".format(j * BACKPROPS_PER_EPOCH + 1,
                                                                j * BACKPROPS_PER_EPOCH + BACKPROPS_PER_EPOCH,
                                                                100.0 * (i + 1) / BACKPROPS_PER_EPOCH), end="");
                    sys.stdout.flush()

                    q.join()
                train_loss = train_err / BACKPROPS_PER_EPOCH
                val_loss, val_acc, precision_score, recall_score = validate(convnet,x_validate,labels_validate)
                precision_list[j] = precision_score
                recall_list[j] = recall_score

                if (val_loss < min_val_err):
                    min_val_err = val_loss
                    convnet.save_param_values(EXCLUDING_PARAM_PATH)

                print("\r{:5.0f}-{:5.0f}:".format(j * BACKPROPS_PER_EPOCH + 1,
                                                  j * BACKPROPS_PER_EPOCH + BACKPROPS_PER_EPOCH), end="");
                sys.stdout.flush()

                print(" val acc: {:5.2f}%".format(val_acc * 100.0))

                ds.saveValues((train_loss,val_loss,val_acc,time.time()-start_time))

        except KeyboardInterrupt:
            print("Iteration stopped through KeyboardInterrupt")
        except:
            raise
        finally:
            q.put('done')
            if os.path.exists(EXCLUDING_PARAM_PATH):
                convnet.load_param_values(EXCLUDING_PARAM_PATH)
            test_acc = test(convnet,x_test,labels_test)

            y_predictions = convnet.test_output(x_test)

            print("test-acc:{:5.2f}%".format(test_acc * 100))

            if not os.path.exists(OUTPUT_DIRECTORY):
                os.makedirs(OUTPUT_DIRECTORY)

            np.save("{}y_predictions".format(OUTPUT_DIRECTORY), y_predictions)
            ds.saveToArray(OUTPUT_DIRECTORY)
            ds.saveToCsv(OUTPUT_DIRECTORY,"acc_loss")

            np.save("{}precision".format(OUTPUT_DIRECTORY), precision_list)
            np.save("{}recall".format(OUTPUT_DIRECTORY), recall_list)

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
