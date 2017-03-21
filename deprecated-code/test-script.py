from __future__ import print_function

import numpy as np

import load_class

# def iterate_minibatches(inputs, targets, oneshot_indices, batch_size, shuffle=False):
#     assert len(inputs) == len(targets)
#     i = 0
#     indices=[]
#     for index in np.arange(len(inputs)):
#         if ( i < len(oneshot_indices) and index == oneshot_indices[i]):
#             i+=1
#         else:
#             indices.append(index)
#     if shuffle:
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
#         excerpt = indices[start_idx:start_idx + batch_size]
#         yield inputs[excerpt], targets[excerpt]

loader = load_class.load(1.0, True)
# x_train, labels_train, class_array = loader.load_training_set()
# x_train, labels_train, class_array = loader.load_validation_set()
x_test, labels_test, oneshot_indices_test = loader.load_training_set()
# print(len(oneshot_indices_test))
# print(len(x_test))
# for batch in iterate_minibatches(x_test, labels_test, oneshot_indices_test, len(x_test)-len(oneshot_indices_test), True):
#     inputs, targets = batch
#     print(len(inputs))
#     print("for end")

# for label in class_array:
#     print(label)
#     print
