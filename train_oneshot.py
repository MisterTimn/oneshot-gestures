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
#import convnet as cnn
import load_class
from util.dataprocessing import DataSaver



num_classes = 20
batch_size = 32
augmenter = aug.augmenter()
loader = load_class.load(15)


samples, labels, indices_train = loader.load_training_set()
indices_train_oneshotclass = indices_train[num_classes-1]
x_validate, labels_validate, indices_validate = loader.load_validation_set()
x_test, labels_test, indices_test = loader.load_testing_set()


# convnet = cnn.convnet_oneshot(num_output_units=20, num_layers_retrain=1)


base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

def worker_backprop(q):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")
    indices = np.empty(batch_size,dtype='int32')

    # indices_train[num_classes - 1] = indices_train_oneshotclass[:samples]

    print(len(indices_train[num_classes-1]))

    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'batch':
            classes = np.random.randint(num_classes,size=batch_size)
            for i in xrange(batch_size):
                indices[i] = indices_train[classes[i]][np.random.randint(len(indices_train[classes[i]]))]
            np.copyto(sharedSampleArray,augmenter.transfMatrix(samples[indices]))
            np.copyto(sharedLabelArray,labels[indices])
        elif cmd == 'change_num_samples':
            q.task_done()
            indices_train[num_classes - 1] = indices_train_oneshotclass[:int(q.get())]
            print(len(indices_train[num_classes - 1]))

        q.task_done()

def iterate_minibatches(inputs, targets, batch_size, class_indices, shuffle=False):
    assert len(inputs) == len(targets)

    # indices_all=np.arange(len(inputs))
    # mask = np.ones(len(indices_all), dtype=bool)
    # mask[class_indices[oneshot_class]] = False
    # indices = indices_all[mask, ...]
    indices=np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def validate(convnet):
    val_err = 0
    val_acc = 0
    val_batches = 0
    precision_score = np.zeros((num_classes))
    recall_score = np.zeros((num_classes))

    for batch in iterate_minibatches(x_validate, labels_validate, batch_size, indices_validate, True):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        val_err += err
        val_acc += acc
        precision_score = np.add(precision_score,metrics.precision_score(targets,convnet.test_output(inputs),xrange(num_classes),average=None))
        recall_score    = np.add(recall_score, metrics.recall_score(targets,convnet.test_output(inputs),xrange(num_classes),average=None))
        val_batches += 1
    return val_err/val_batches, val_acc/val_batches, precision_score/val_batches, recall_score/val_batches

if __name__=='__main__':
    q = mp.JoinableQueue()
    try:
        sharedSampleArray = sa.create("shm://samples", (batch_size, 12, 64, 64), dtype='float32')
        sharedLabelArray = sa.create("shm://labels", batch_size, dtype='int32')


        proc = mp.Process(target=worker_backprop, args=[q])
        proc.daemon = True
        proc.start()

        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')

        oneshot_class = 15

        # retrain_layers = 3
        # for num_oneshot_samples in [200,100,50,25,10]:
        # num_oneshot_samples = 2
        for num_oneshot_samples in [50,100,200]:
            for retrain_layers in [1]:
                ds = DataSaver(('train_loss', 'val_loss', 'val_acc', 'dt'))


                convnet = cnn.convnet_oneshot(num_output_units=20, num_layers_retrain=retrain_layers)

                indices_train[num_classes-1] = indices_train_oneshotclass[:num_oneshot_samples]
                print(len(indices_train[num_classes-1]))

                q.put('change_num_samples')
                q.join()
                q.put(num_oneshot_samples)

                save_param_path = "{}convnet_params/model-19x1/class-{}/layers{}-samples{}".format(base_dir_path,oneshot_class,retrain_layers,num_oneshot_samples)
                min_val_acc = 0
                convnet.preload_excluding_model("{}convnet_params/model-19/excluding-{}".format(base_dir_path,oneshot_class))
                q.join()

                ###
                # In case there is need to load old params to continue training
                ###
                # convnet.load_param_values(save_param_path)

                try:
                    backprops_per_epoch = 200
                    num_backprops = 10000 / backprops_per_epoch
                    precision_list = np.zeros((num_backprops,num_classes))
                    recall_list = np.zeros((num_backprops,num_classes))
                    for j in xrange(num_backprops):
                        #Initialise new random permutation of data
                        #And load first batch of augmented samples
                        q.put('batch')
                        end_epoch = False
                        train_err = 0
                        train_batches=0
                        start_time = time.time()

                        q.join()
                        for i in xrange(backprops_per_epoch):
                            # Data kopieren om daarna te starten met augmentatie volgende batch
                            np.copyto(sample_batch,sharedSampleArray)
                            np.copyto(label_batch,sharedLabelArray)

                            q.put('batch')

                            #trainen op de gekopieerde data
                            train_err += convnet.train(sample_batch, label_batch)
                            train_batches += 1
                            print("\r{:5.0f}-{:5.0f}:\t{:5.0f}%".format(j * backprops_per_epoch + 1,
                                                                        j * backprops_per_epoch + backprops_per_epoch,
                                                                        100.0 * (i + 1) / backprops_per_epoch),end="");
                            sys.stdout.flush()

                            q.join()
                        train_loss = train_err / backprops_per_epoch
                        val_loss, val_acc, precision_score, recall_score = validate(convnet)
                        precision_list[j] = precision_score
                        recall_list[j] = recall_score


                        if (val_acc > min_val_acc):
                            min_val_acc = val_acc
                            convnet.save_param_values(save_param_path)

                        print("\r{:5.0f}-{:5.0f}:".format(j * backprops_per_epoch + 1,
                                                                    j * backprops_per_epoch + backprops_per_epoch),end="");
                        sys.stdout.flush()

                        print(" val acc: {:5.2f}%, precision: {:5.2f}%, recall: {:5.2f}%"
                              .format(val_acc * 100.0,precision_score[num_classes-1] * 100.0,recall_score[num_classes-1] * 100.0))
                        ds.saveValues((train_loss,val_loss,val_acc,time.time()-start_time))

                except KeyboardInterrupt:
                    print("Iteration stopped through KeyboardInterrupt")
                except:
                    raise
                finally:
                    if os.path.exists(save_param_path):
                        convnet.load_param_values(save_param_path)

                    test_acc, test_acc, precision_score, recall_score = validate(convnet)
                    y_predictions = convnet.test_output(x_test)

                    print("\ttest-acc:{:5.2f}%".format(test_acc * 100))

                    directory = "{}output/model-19x1/class-{}/layers{}-samples{}/".format(base_dir_path, oneshot_class,retrain_layers,num_oneshot_samples)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    np.save("{}y_predictions".format(directory),y_predictions)

                    ds.saveToArray(directory)
                    ds.saveToCsv(directory,"acc_loss")
                    np.save("{}precision".format(directory),precision_list)
                    np.save("{}recall".format(directory),recall_list)

                    if not os.path.exists("{}output/model-19x1/class-{}/test-acc.txt".format(base_dir_path,oneshot_class)):
                        open("{}output/model-19x1/class-{}/test-acc.txt".format(base_dir_path,oneshot_class),'w').close()
                    with open("{}output/model-19x1/class-{}/test-acc.txt".format(base_dir_path,oneshot_class), 'ab') as f:
                        f.write("layers{};samples{};{}\n".format(retrain_layers, num_oneshot_samples, 1.0 * test_acc))



    except:
        raise
    finally:
        q.put('done')
        sa.delete("samples")
        sa.delete("labels")
    print("End of program")
