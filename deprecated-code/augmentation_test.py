import augmentation as aug
import load_class
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import pylab
from numpy.random import randint
import pickle
import time
import numpy as np

import multiprocessing as mp
import SharedArray as sa

with open("/home/jasper/oneshot-gestures/data-chalearn/samples_augm", 'rb') as f:
    samples = pickle.load(f)
with open("/home/jasper/oneshot-gestures/data-chalearn/labels_augm", 'rb') as f:
    labels = pickle.load(f)

augmenter = aug.augmenter()
# loader = load_class.load()
samples, labels, oneshot_indices_test = loader.load_training_set()

num_classes = 20
oneshot_class = -1
num_oneshot_samples = 200
batch_size=50

def worker(q):
    #Data voor volgende iteratie ophalen en verwerken
    #Opslaan in shared memory
    done = False
    sharedSampleArray = sa.attach("shm://samples")
    sharedLabelArray = sa.attach("shm://labels")

    indices = []
    for gesture_class in xrange(num_classes):
        if (gesture_class == oneshot_class):
            indices.extend[oneshot_indices_test[gesture_class][:num_oneshot_samples]]
        else:
            np.random.shuffle(oneshot_indices_test[gesture_class])
            indices.extend(oneshot_indices_test[gesture_class][:num_oneshot_samples])
    np.random.suffle(indices)

    start_index=0
    while not done:
        cmd = q.get()
        if cmd == 'done':
            done = True
        elif cmd == 'augment':
            excerpt = indices[start_index:start_index+batch_size]
            images = augmenter.transfMatrix(samples[excerpt])

            np.copyto(sharedSampleArray,images)
            np.copyto(sharedLabelArray,labels[excerpt])

            start_index+=batch_size
        if start_index >= len(indices):
            done = True
        q.task_done()

if __name__=='__main__':
    try:

        q = mp.JoinableQueue()
        proc = mp.Process(target=worker, args=[q])
        proc.daemon = True
        proc.start()

        sharedSampleArray = sa.create("shm://samples", (batch_size, 12, 64, 64), dtype='float32')
        sharedLabelArray =  sa.create("shm://labels", batch_size)

        f = pylab.figure()
        pylab.gray()

        num_col = 10
        num_samples = 60
        num_row = int(round( num_samples / num_col ))

        image_copy = np.empty((100,12,64,64))
        for j in xrange(3):
            q.put('augment')

            q.join()

            for i in xrange(20):
                rand_subsample=0
                f.add_subplot(num_row, num_col, j*20+i+1)
                pylab.imshow(image_copy[i+j*20][rand_subsample])
                pylab.axis('off')

        pylab.show()
    except:
        raise
    finally:
        q.put('done')
        sa.delete("samples")
    print("End of program")