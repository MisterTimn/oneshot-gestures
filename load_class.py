from __future__ import print_function
import h5py
import numpy as np
import sys


#from pylearn2.format.target_format import OneHotFormatter

class load(object):

    def __init__(self, size_ratio=1):
        print
        print("Initializing load_module... (0/4)",end="")
        sys.stdout.flush()
        #data_path = "/home/jasper/oneshot-gestures/data-chalearn/"
        data_path = "/home/jveessen/"

        file = h5py.File(data_path+"data_ints.hdf5","r")
        print("\rInitializing load_module... (1/4)", end="");sys.stdout.flush()

        self.samples = np.asarray(file["samples"], dtype='float32')
        print("\rInitializing load_module... (2/4)", end="");sys.stdout.flush()

        self.labels = file["labels"]
        print("\rInitializing load_module... (3/4)", end="");sys.stdout.flush()

        self.labels_original = file["labels_original"]
        print("\rInitializing load_module... (4/4)");sys.stdout.flush()

        self.sample_size = int((self.samples.shape[0])*size_ratio)

    def load_training_set(self):
        print("Loading training set...");sys.stdout.flush()
        x_train = self.samples[0:int(self.sample_size*0.6)]
        labels_train = self.labels_original[0:int(self.sample_size*0.6)]
        class_indices = self.initialize_class_array(labels_train)
        return x_train, labels_train, class_indices

    def load_validation_set(self):
        print("\rLoading validation set...");sys.stdout.flush()
        x_validate = self.samples[int(self.sample_size*0.6):int(self.sample_size*0.8)]
        labels_validate = self.labels_original[int(self.sample_size*0.6):int(self.sample_size*0.8)]
        class_indices = self.initialize_class_array(labels_validate)
        return x_validate, labels_validate, class_indices

    def load_testing_set(self):
        print("\rLoading test set...");sys.stdout.flush()
        x_test = self.samples[int(self.sample_size*0.8):self.sample_size]
        labels_test = self.labels_original[int(self.sample_size*0.8):self.sample_size]
        class_indices = self.initialize_class_array(labels_test)
        return x_test, labels_test, class_indices

    def initialize_class_array(self,labels):
        class_arrays= [[] for i in xrange(20)]
        i=0
        for label in labels:
            class_arrays[label].append(i)
            i+=1
        return class_arrays
