import h5py
import theano
import numpy as np
import time
from pylearn2.format.target_format import OneHotFormatter

print "something"

def floatX(x):
        return np.asarray(x, dtype=theano.config.floatX)

def convertToOneHot(labels_original, max_labels=20):
    fmt = OneHotFormatter(max_labels, dtype=theano.config.floatX)
    labels_original = np.asarray(labels_original,dtype='uint8')
    return fmt.format(labels_original)

print
print "Initializing load_module..."
start_time=time.time()

data_path = "/home/jasper/Thesis/data-chalearn/"
file = h5py.File(data_path+"data.hdf5","r")

samples = file["samples"]    
labels_original = file["labels"]
labels_original = labels_original - np.ones(labels_original.shape)
labels_original = np.asarray(labels_original,dtype='uint8')
sample_size = int(samples.shape[0])

"""
Select all depth images (body, lhand and rhand)
Select 4 frames out of 32
3 by 4 = 12 feature maps
Reshape to fit the model
"""
frame_selection_time = time.time()
samples = samples[0:sample_size,0:samples.shape[1],(7,13,18,25)]
print "Frame selection time: %s" % (time.time() - frame_selection_time)

video_selection_time = time.time()
samples = samples[0:sample_size,(1,3,5)]
print "Video selection time: %s" % (time.time() - video_selection_time)

conversion_time = time.time()
samples = floatX(samples.reshape((sample_size,12,64,64)))/255.
print "Conversion time: %s" % (time.time() - conversion_time)

"""
Split samples into two datasets
Sets with only the class 19 for one-shot learning
Sets with the leftover classes (19 in total) for pretraining
"""
indices = []
samples_oneshot = []
labels_original_oneshot = []
class_to_filter=19
aantal = 0
for i, j in enumerate(labels_original[0:sample_size]):
    if j == class_to_filter:
        samples_oneshot.append(samples[i])
        labels_original_oneshot.append(class_to_filter)
        indices.append(i)
        aantal += 1
samples_oneshot = np.asarray(samples_oneshot)
labels_original_oneshot = np.asarray(labels_original_oneshot)
labels_oneshot = convertToOneHot(labels_original_oneshot,20)

aantal = 0
samples_19cl = []
labels_original_19cl = []
j=0
for i in range(0,len(samples)):
    if(j< len(indices) and i==indices[j]):
        j+=1
    else:
        samples_19cl.append(samples[i])
        labels_original_19cl.append(labels_original[i])
        aantal +=1
samples_19cl = np.asarray(samples_19cl)
labels_original_19cl = np.asarray(labels_original_19cl)
labels_19cl = convertToOneHot(labels_original_19cl,19)

"""
Export as hdf5 dataset
"""
h5f = h5py.File(data_path+'data_oneshotlearning.hdf5', 'w')
h5f.create_dataset('samples_oneshot', data=samples_oneshot)
h5f.create_dataset('labels_oneshot', data=labels_oneshot)
h5f.create_dataset('labels_original_oneshot', data=labels_original_oneshot)
h5f.create_dataset('samples_19cl', data=samples_19cl)
h5f.create_dataset('labels_19cl', data=labels_19cl)
h5f.create_dataset('labels_original_19cl', data=labels_original_19cl)
h5f.close()

file.close()

print "Load_module initialized in %s seconds." % (time.time() - start_time)
print
