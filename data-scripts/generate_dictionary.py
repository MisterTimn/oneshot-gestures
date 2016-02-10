import h5py
import theano
import numpy as np
import pickle

import load_class
load = load_class.load(1)

# labels_validate = load.load_validation_set()
labels_train, labels_validate, labels_test = load.load_labels()
# labels_test = load.load_testing_set()

train_dict = {}
validate_dict = {}
test_dict = {}

for i in range(0, len(labels_train)):
	class_nr = labels_train[i]
	if class_nr not in train_dict:
		train_dict[class_nr] = []
	train_dict[class_nr].append(i)

	if i < len(labels_validate):
		class_nr = labels_validate[i]
		if class_nr not in validate_dict:
			validate_dict[class_nr] = []
		validate_dict[class_nr].append(i)

		class_nr = labels_test[i]
		if class_nr not in test_dict:
			test_dict[class_nr] = []
		test_dict[class_nr].append(i)

with open("train_dictionary", 'wb') as f:
	pickle.dump(train_dict,f)
with open("validate_dictionary", 'wb') as f:
	pickle.dump(validate_dict,f)
with open("test_dictionary", 'wb') as f:
	pickle.dump(test_dict,f)

print "Dictionaries saved succesfully"
