from __future__ import print_function

import SharedArray as sa
import multiprocessing as mp
import numpy as np
import sys
import time
import os

import augmentation as aug
import convnet_v2 as cnn
import load_class
from util.dataprocessing import DataSaver

augmenter = aug.augmenter()

# loader = load_class.load()
#
# samples, labels, indices_train = loader.load_training_set()
# x_validate, labels_validate, indices_validate = loader.load_validation_set()
# x_test, labels_test, indices_test = loader.load_testing_set()

base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/"
num_classes = 20
batch_size = 32

def worker_backprop(q,samples,labels,indices_train):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")
    indices = np.empty(batch_size,dtype='int32')

    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'batch':
            # classes = np.random.choice(class_choices,batch_size)
            classes = np.random.randint(num_classes-1, size=batch_size)
            for i in xrange(batch_size):
                indices[i] = indices_train[classes[i]][np.random.randint(len(indices_train[classes[i]]))]
            np.copyto(sharedSampleArray,augmenter.transfMatrix(samples[indices]))
            np.copyto(sharedLabelArray,labels[indices])
        elif cmd == 'oneshot':
            q.task_done()
            class_choices = []
            for class_num in xrange(20):
                if class_num != num_classes-1:
                    class_choices.append(class_num)
        q.task_done()

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices=np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

def getClassAccuracy(targets, predictions, class_num):
    assert(len(targets)==len(predictions))
    predict_count = 0
    class_count = 0
    for i in xrange(len(targets)):
        if(targets[i] == class_num):
            class_count += 1
            if(targets[i] == predictions[i]):
                predict_count += 1
    return predict_count, class_count

def validate(convnet,x_validate,labels_validate):
    val_err = 0
    val_acc = 0
    val_batches = 0
    num_valid_class_acc = 0
    class_acc = 0
    for batch in iterate_minibatches(x_validate, labels_validate, batch_size, True):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        val_err += err
        val_acc += acc
        predict_count, class_count = getClassAccuracy(targets, convnet.test_output(inputs), oneshot_class)
        if ( class_count != 0 ):
            class_acc += 1.0 * predict_count / class_count
            num_valid_class_acc += 1
        val_batches += 1
    return val_err/val_batches, val_acc/val_batches, class_acc/num_valid_class_acc

def test(convnet,x_test,labels_test):
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, labels_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = convnet.validate(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    return test_acc / test_batches

if __name__=='__main__':
    try:
        sharedSampleArray = sa.create("shm://samples", (batch_size, 12, 64, 64), dtype='float32')
        sharedLabelArray = sa.create("shm://labels", batch_size, dtype='int32')



        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')

        global loader
        global samples, labels, indices_train
        global x_validate, labels_validate, indices_validate
        global x_test, labels_test, indices_test

        for oneshot_class in xrange(8,14):

            loader = load_class.load(oneshot_class)
            print(loader.get_oneshot())

            samples, labels, indices_train = loader.load_training_set()
            x_validate, labels_validate, indices_validate = loader.load_validation_set()
            labels_validate = labels_validate[np.in1d(range(len(labels_validate)),indices_validate[oneshot_class])]
            x_test, labels_test, indices_test = loader.load_testing_set()
            labels_test = labels_test[np.in1d(range(len(labels_test)),indices_test[oneshot_class])]


            min_val_err = 20

            ds = DataSaver(('train_loss', 'val_loss', 'val_acc', 'dt'))

            convnet = cnn.convnet(num_output_units=19)
            save_param_path = "{}convnet_params/excluding-{}".format(base_dir_path, oneshot_class)

            try:
                q = mp.JoinableQueue()
                proc = mp.Process(target=worker_backprop, args=[q, samples, labels, indices_train])
                proc.daemon = True
                proc.start()

                backprops_per_epoch = 200
                num_backprops = 3*20000 / backprops_per_epoch
                for j in xrange(num_backprops):
                    #Initialise new random permutation of data
                    #And load first batch of augmented samples
                    q.put('batch')
                    end_epoch = False
                    train_err = 0
                    train_batches=0
                    start_time = time.time()
                    print("\t--- BACKPROP {} to {} ---".format(j*backprops_per_epoch+1,j*backprops_per_epoch+backprops_per_epoch))
                    #wait for data
                    q.join()
                    for i in xrange(backprops_per_epoch):
                        # Data kopieren om daarna te starten met augmentatie volgende batch
                        np.copyto(sample_batch,sharedSampleArray)
                        np.copyto(label_batch,sharedLabelArray)

                        q.put('batch')

                        #trainen op de gekopieerde data
                        train_err += convnet.train(sample_batch, label_batch)
                        train_batches += 1
                        print("\rtrain err:\t{:5.2f}".format(train_err / i), end="");sys.stdout.flush()
                        print("\t{:5.0f}%".format(100.0 * (i+1) / backprops_per_epoch), end="");sys.stdout.flush()

                        q.join()
                    train_loss = train_err / backprops_per_epoch
                    val_loss, val_acc, class_acc = validate(convnet,x_validate,labels_validate)

                    if (val_loss < min_val_err):
                        min_val_err = val_loss
                        convnet.save_param_values(save_param_path)

                    print("\nval err:\t{:5.2f}\nval acc:\t{:5.2f}%"
                          .format(val_loss,val_acc * 100.0))
                    ds.saveValues((train_loss,val_loss,val_acc,time.time()-start_time))

            except KeyboardInterrupt:
                print("Iteration stopped through KeyboardInterrupt")
            except:
                raise
            finally:
                q.put('done')
                convnet.load_param_values(save_param_path)
                test_acc = test(convnet,x_test,labels_test)
                print("test-acc:{:5.2f}%".format(test_acc * 100))

                directory = "{}output/excluding-{}/".format(base_dir_path, oneshot_class)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                ds.saveToArray(directory)
                ds.saveToCsv(directory,"acc_loss")

    except:
        raise
    finally:
        sa.delete("samples")
        sa.delete("labels")
    print("End of program")
