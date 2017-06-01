import matplotlib.pyplot as plt
import numpy as np
import pylab
import matplotlib.cm as cm
import Image

import load_class
load = load_class.load(1)



# f = pylab.figure()
# for n, fname in enumerate(('1.png', '2.png')):
#     image=Image.open(fname).convert("L")
#     arr=np.asarray(image)
#     f.add_subplot(2, 1, n)  # this line outputs images on top of each other
#     # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
#     pylab.imshow(arr,cmap=cm.Greys_r)
# pylab.title('Double image')
# pylab.show()

x_validate, labels_validate,indices = load.load_validation_set()


class_to_filter=0
aantal = 0
for i, j in enumerate(labels_validate[0:1000]):
    if j == class_to_filter:
    	f = pylab.figure();print(j)
    	for k in range(12):
    		f.add_subplot(3,4,k+1)
    		pylab.imshow(x_validate[i,k],cmap='binary')
        	# pylab.title("%s" % k)
    		pylab.axis('off')
        class_to_filter+=1
        pylab.show()
    elif 19 < class_to_filter:
    	break
