import theano
import theano.tensor as T
import numpy as np
import os
import time

import load_class
load = load_class.load(1)

import convnet
convnet = convnet.convnet()

"""
Initialize output File-ID and file
"""
i = 0
while os.path.exists("output/06-05/acc-cost_%s.csv" % i):
    i += 1
print "File-ID: %s" % i
fo1 = open("output/06-05/acc-cost_%s.csv" % i, "w")

save_param_path = "model_params/param_model"

x_validate, labels_validate = load.load_validation_set()
x_train, t_train = load.load_training_set()

batch_size = 20
iterationIndex=0
x=raw_input("Number of iterations to perform: ")
while(not x.isdigit()):
	x=raw_input("Not a valid number, again: ")

iterations=int(x)
go=True
try:
	while (go):
		start_time = time.time()
		iterationIndex=iterationIndex+1
		list_cost = []

		start_iteration_time = time.time()
		for start in range(0, len(x_train), batch_size):
			x_batch = x_train[start:start + batch_size]
			t_batch = t_train[start:start + batch_size]
			cost = convnet.train(x_batch, t_batch)
			list_cost.append(cost)

		predictions_validate = []
		for start in range(0, len(x_validate), batch_size):
			predictions_validate.extend(convnet.predict(x_validate[start:start+batch_size]))

		cost_avg = np.average(list_cost)
		accuracy = np.mean(predictions_validate == labels_validate)

		print "%3d: acc = %.5f || cost = %.5f" % (iterationIndex, accuracy, cost_avg)
		fo1.write("%.5f;%.5f;%.5f\n" % (accuracy, cost_avg, time.time() - start_iteration_time))

		#Save the model parameters at certain intervals
		#For safety
		if(iterationIndex%25 == 0):
			convnet.save_param_values(save_param_path)
			print "Model parameters saved."

		if(iterationIndex==iterations):
			print("--- Execution time: %s seconds ---" % (time.time() - start_time))
			x=raw_input("#Iterations or q(uit): ")
			if(x=='q' or x=='quit'):
				go=False
			else:
				while(not x.isdigit()):
					x=raw_input("Not a valid number, again: ")
				iterations+=int(x)

except KeyboardInterrupt:
	print "Program forced to stop via KeyboardInterrupt"
except:
	raise

convnet.save_param_values(save_param_path)

x_test, labels_test = load.Load_testing_set()
predictions_test = convnet.predict(x_test)
test_accuracy = np.mean(predictions_test == labels_test)

print "Accuracy of testset prediction: %.5f" % test_accuracy
fo1.close()