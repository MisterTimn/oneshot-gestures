from __future__ import print_function

import SharedArray as sa
import multiprocessing as mp
import numpy as np
import sys
import time
from skimage.viewer import ImageViewer

import augmentation as aug
import convnet
import load_class
from util.dataprocessing import DataSaver

augmenter = aug.augmenter()
loader = load_class.load()
ds = DataSaver(('train_loss','val_loss','val_acc','dt'))

samples, labels, indices_train = loader.load_training_set()
x_validate, labels_validate, indices_validate = loader.load_validation_set()
x_test, labels_test, indices_test = loader.load_testing_set()

convnet = convnet.convnet(num_output_units=20)
base_dir_path = "/home/jasper/oneshot-gestures/"
#save_param_path = "{}convnet_params/param-augmented-v3".format(base_dir_path)



num_classes = 20
oneshot_class = -1
num_oneshot_samples = 200
num_samples_per_class = 200
batch_size=64
minvalerr=20
max_batch_iterations = (num_classes-1) * num_samples_per_class / batch_size

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
            q.task_done()
            oneshot_class = q.get()
            indices = []
            for gesture_class in xrange(num_classes):
                if (gesture_class != oneshot_class):
                    np.random.shuffle(indices_train[gesture_class])
                    indices.extend(indices_train[gesture_class][:num_samples_per_class])
                # else:
                #     for i in xrange(num_samples_per_class / num_oneshot_samples):
                #         indices.extend[oneshot_indices_test[gesture_class][:num_oneshot_samples]]
            np.random.shuffle(indices)
            start_index=0
        if ((cmd == 'batch' or cmd == 'epoch') and start_index < len(indices)):
            excerpt = indices[start_index:start_index+batch_size]
            images = augmenter.transfMatrix(samples[excerpt])
            np.copyto(sharedSampleArray,images)
            np.copyto(sharedLabelArray,labels[excerpt])
            start_index+=batch_size
        q.task_done()

def worker_backprop(q):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")
    start_index=0
    indices = np.empty(batch_size,dtype='int32')

    class_choices = []

    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'batch':
            #classes = np.random.randint(num_classes,size=batch_size)
            classes = np.random.choice(class_choices,64)
            for i in xrange(batch_size):
                indices[i] = indices_train[classes[i]][np.random.randint(len(indices_train[classes[i]]))]
            np.copyto(sharedSampleArray,augmenter.transfMatrix(samples[indices]))
            np.copyto(sharedLabelArray,labels[indices])
        elif cmd == 'oneshot':
            q.task_done()
            class_choices = []
            oneshot_class = q.get()
            for class_num in xrange(20):
                if class_num != oneshot_class:
                    class_choices.append(class_num)
            print(class_choices)

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
    return val_err/val_batches, val_acc/val_batches

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
    return test_acc / test_batches

if __name__=='__main__':
    try:
        backprops_per_epoch = 20
        q = mp.JoinableQueue()
        proc = mp.Process(target=worker_backprop, args=[q])
        proc.daemon = True
        proc.start()

        sharedSampleArray   = sa.create("shm://samples", (batch_size, 12, 64, 64), dtype='float32')
        sharedLabelArray    = sa.create("shm://labels", batch_size, dtype='int32')

        sample_batch    = np.empty(sharedSampleArray.shape, dtype='float32')
        label_batch     = np.empty(sharedLabelArray.shape, dtype='int32')

        #convnet.load_param_values(save_param_pth)
        for oneshot_class in [-1]:
            save_param_path = "{}convnet_params/param-augmented-v4-excl_{}".format(base_dir_path,oneshot_class)
            q.put('oneshot')
            q.put(oneshot_class)

            num_backprops = 60
            for j in xrange(num_backprops):
                #Initialise new random permutation of data
                #And load first batch of augmented samples
                q.put('batch')
                end_epoch = False
                train_err = 0
                train_batches=0
                start_time = time.time()
                print("\t--- EPOCH {} ---".format(j+1))
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
                    print("\rtrain err:\t{:5.2f}".format(train_err / i), end="");
                    sys.stdout.flush()

                    batch_iteration+=1
                    print("\t{:5.0f}%".format(100.0 * i / backprops_per_epoch), end="");sys.stdout.flush()

                    #wachten op nieuwe data
                    batch_waiting_time = time.time()
                    q.join()
                    print("\tlost {:4.2f}s".format(batch_waiting_time - time.time()),end="");sys.stdout.flush()
                    print()
                train_loss = train_err / backprops_per_epoch
                al_loss, val_acc = validate(minvalerr)

                print("val err:\t{:5.2f}\nval acc:\t{:5.1f}%".format(val_loss,
                                                                     val_acc * 100.0))
                ds.saveToArray((train_loss,val_loss,val_acc,start_time-time.time()))

            q.put('done')

            convnet.load_param_values(save_param_path)
            test_acc = test()
            print("test-acc:{:7.2f}%".format(test_acc * 100))

            directory = "{}output/data-{}/".format(base_dir_path, oneshot_class)
            if not os.path.exists(directory):
                os.makedirs(directory)
            ds.saveToArray(directory)
            ds.saveToCsv(directory,"acc_loss")

    except:
        raise
    finally:
        q.put('done')
        sa.delete("samples")
        sa.delete("labels")
    print("End of program")