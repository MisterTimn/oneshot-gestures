from __future__ import print_function

import SharedArray as sa
import multiprocessing as mp
import numpy as np
import sys
import time
import os
from sklearn import metrics

import augmentation as aug
import convnet_19x1 as cnn
import load_class
from util.dataprocessing import DataSaver

BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_VERS      =   "model-19x1"
MODEL_EXCLUDING =   "model-19"
# ONESHOT_CLASS   =   14

# OUTPUT_DIRECTORY=   "{}output/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
# PARAM_DIRECTORY =   "{}convnet_params/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
# EXCLUDING_PARAM_PATH   \
#                 =   "{}convnet_params/{}/excluding-{}".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS)
#
# if not os.path.exists(OUTPUT_DIRECTORY):
#     os.makedirs(OUTPUT_DIRECTORY)
# if not os.path.exists(PARAM_DIRECTORY):
#     os.makedirs(PARAM_DIRECTORY)

TOTAL_BACKPROPS = 10000
BACKPROPS_PER_EPOCH = 500
NUM_EPOCHS = TOTAL_BACKPROPS / BACKPROPS_PER_EPOCH
NUM_CLASSES = 20
BATCH_SIZE = 32

augmenter = aug.augmenter()


def worker_backprop(q,samples,labels,indices_train,indices_train_oneshotclass):
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
            classes = np.random.randint(NUM_CLASSES, size=BATCH_SIZE)
            for i in xrange(BATCH_SIZE):
                indices[i] = indices_train[classes[i]][np.random.randint(len(indices_train[classes[i]]))]
            np.copyto(sharedSampleArray,augmenter.transfMatrix(samples[indices]))
            np.copyto(sharedLabelArray,labels[indices])
        elif cmd == 'change_num_samples':
            q.task_done()
            indices_train[NUM_CLASSES - 1] = indices_train_oneshotclass[:int(q.get())]
            print("Training with {} samples".format(len(indices_train[NUM_CLASSES - 1])))
        q.task_done()
        sys.stdout.flush()

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices=np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def validate(convnet, inputs, targets):
    val_err = 0
    val_acc = 0
    val_batches = 0
    precision_score = np.zeros((NUM_CLASSES))
    recall_score = np.zeros((NUM_CLASSES))

    for batch in iterate_minibatches(inputs, targets, BATCH_SIZE, True):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        val_err += err
        val_acc += acc
        precision_score = np.add(precision_score, metrics.precision_score(targets, convnet.test_output(inputs), xrange(NUM_CLASSES), average=None))
        recall_score    = np.add(recall_score, metrics.recall_score(targets, convnet.test_output(inputs), xrange(NUM_CLASSES), average=None))
        val_batches += 1
    return val_err/val_batches, val_acc/val_batches, precision_score/val_batches, recall_score/val_batches

if __name__=='__main__':
    q = mp.JoinableQueue()
    try:
        sharedSampleArray = sa.create("shm://samples", (BATCH_SIZE, 12, 64, 64), dtype='float32')
        sharedLabelArray = sa.create("shm://labels", BATCH_SIZE, dtype='int32')

        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')


        # retrain_layers = 3
        # for num_oneshot_samples in [200,100,50,25,10]:
        # num_oneshot_samples = 2
        for ONESHOT_CLASS in xrange(1,10):

            OUTPUT_DIRECTORY = "{}output/{}/class-{}/".format(BASE_DIR, MODEL_VERS, ONESHOT_CLASS)
            PARAM_DIRECTORY = "{}convnet_params/{}/class-{}/".format(BASE_DIR, MODEL_VERS, ONESHOT_CLASS)
            EXCLUDING_PARAM_PATH \
                = "{}convnet_params/{}/excluding-{}".format(BASE_DIR, MODEL_EXCLUDING, ONESHOT_CLASS)

            if not os.path.exists(OUTPUT_DIRECTORY):
                os.makedirs(OUTPUT_DIRECTORY)
            if not os.path.exists(PARAM_DIRECTORY):
                os.makedirs(PARAM_DIRECTORY)

            loader = load_class.load(ONESHOT_CLASS)
            samples, labels, indices_train = loader.load_training_set()
            indices_train_oneshotclass = indices_train[NUM_CLASSES - 1]
            x_validate, labels_validate, indices_validate = loader.load_validation_set()
            x_test, labels_test, indices_test = loader.load_testing_set()

            proc = mp.Process(target=worker_backprop,
                              args=[q,samples,labels,indices_train,indices_train_oneshotclass])
            proc.daemon = True
            proc.start()

            for num_oneshot_samples in [1]:
                for retrain_layers in [1]:
                    q.put('change_num_samples')
                    q.join()
                    q.put(num_oneshot_samples)

                    ds = DataSaver(('train_loss', 'val_loss', 'val_acc', 'dt'))

                    precision_list = np.zeros((NUM_EPOCHS, NUM_CLASSES))
                    recall_list = np.zeros((NUM_EPOCHS, NUM_CLASSES))
                    min_val_acc = 0
                    patience = 0

                    convnet = cnn.convnet_oneshot(num_output_units=20, num_layers_retrain=retrain_layers)
                    convnet.preload_excluding_model(path=EXCLUDING_PARAM_PATH)

                    save_param_path = "{}layers{}-samples{}".format(PARAM_DIRECTORY, retrain_layers, num_oneshot_samples)



                    q.join()
                    try:
                        for j in xrange(NUM_EPOCHS):
                            #Initialise new random permutation of data
                            #And load first batch of augmented samples
                            q.put('batch')
                            end_epoch = False
                            train_err = 0
                            train_batches=0
                            start_time = time.time()

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

                            ds.saveValues((train_loss,val_loss,val_acc,time.time()-start_time))

                            if (val_acc > min_val_acc):
                                min_val_acc = val_acc
                                convnet.save_param_values(save_param_path)
                                patience = 0
                            else:
                                patience+=1

                            print("\r{:5.0f}-{:5.0f}:".format(j * BACKPROPS_PER_EPOCH + 1,
                                                              j * BACKPROPS_PER_EPOCH + BACKPROPS_PER_EPOCH), end="");
                            sys.stdout.flush()

                            print(" patience: {:3} val acc: {:5.2f}%, precision: {:5.2f}%, recall: {:5.2f}%"
                                  .format(patience,val_acc * 100.0, precision_score[NUM_CLASSES - 1] * 100.0, recall_score[NUM_CLASSES - 1] * 100.0))

                    except KeyboardInterrupt:
                        print("Iteration stopped through KeyboardInterrupt")
                    except:
                        raise
                    finally:
                        if os.path.exists(save_param_path):
                            convnet.load_param_values(save_param_path)

                        test_acc, test_acc, precision_score, recall_score = validate(convnet,x_test,labels_test)
                        y_predictions = convnet.test_output(x_test)
                        metrics.classification_report(labels_test,y_predictions)

                        print("\ttest-acc:{:5.2f}%".format(test_acc * 100))

                        directory = "{}layers{}-samples{}/".format(OUTPUT_DIRECTORY, retrain_layers, num_oneshot_samples)
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        np.save("{}y_predictions".format(directory),y_predictions)

                        ds.saveToArray(directory)
                        ds.saveToCsv(directory,"acc_loss")
                        np.save("{}precision".format(directory),precision_list)
                        np.save("{}recall".format(directory),recall_list)

                        if not os.path.exists("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS)):
                            open("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS), 'w').close()
                        with open("{}test-acc.txt".format(OUTPUT_DIRECTORY, ONESHOT_CLASS), 'ab') as f:
                            f.write("layers{};samples{};{}\n".format(retrain_layers, num_oneshot_samples, 1.0 * test_acc))
                            f.write(metrics.classification_report(labels_test,y_predictions))
                        q.put('done')


    except:
        raise
    finally:
        sa.delete("samples")
        sa.delete("labels")
    print("End of program")
