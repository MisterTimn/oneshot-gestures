import h5py
import theano
import numpy as np
import time
from pylearn2.format.target_format import OneHotFormatter

class load(object):

    def __init__(self, size_ratio=1):
        print
        print "Initializing load_module..."
        start_time=time.time()

        data_path = "/home/jasper/oneshot-gestures/data-chalearn/"
        """
        data_revised
        data_oneshotlearning
        """
        # file = h5py.File(data_path+"data_oneshotlearning.hdf5","r")
        
        # # self.samples = file["samples_oneshot"] 
        # # self.labels = file["labels_oneshot"]
        # # self.labels_original = file["labels_original_oneshot"]

        # self.samples = file["samples_19cl"]
        # self.labels = file["labels_19cl"]
        # self.labels_original = file["labels_original_19cl"]
        
        file = h5py.File(data_path+"data_revised.hdf5","r")

        self.samples = file["samples"] 
        self.labels = file["labels"]
        self.labels_original = file["labels_original"]       

        self.sample_size = int((self.samples.shape[0])*size_ratio)

        print "Load_module initialized in %s seconds." % (time.time() - start_time)
        print

    def load_training_set(self):
        x_train = self.samples[0:int(self.sample_size*0.6)]
        t_train = self.labels[0:int(self.sample_size*0.6)]
        return x_train, t_train

    def load_validation_set(self):
        x_validate = self.samples[int(self.sample_size*0.6):int(self.sample_size*0.8)]
        labels_validate = self.labels_original[int(self.sample_size*0.6):int(self.sample_size*0.8)]
        return x_validate, labels_validate

    def load_testing_set(self):
        x_test = self.samples[int(self.sample_size*0.8):self.sample_size]
        labels_test = self.labels_original[int(self.sample_size*0.8):self.sample_size]
        return x_test, labels_test