import sys
import os

path = "/Users/jaspervaneessen/GitHub/oneshot-gestures/output/datav2-15-retrain-2-samples-2/acc_loss.csv"
path2 = "/Users/jaspervaneessen/GitHub/oneshot-gestures/output/datav2-15-retrain-2-samples-2/data.csv"

with open(path, "r") as in_f, open(path2, "w") as out_f:
    lines = in_f.readlines()
    i = 0
    for line in lines:
    	if i == 0:
    		line = "{};{}".format("epoch",line)
    	else:
    		line = "{};{}".format(i,line)
        out_f.write(line)
        i+=1