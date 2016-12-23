import augmentation as aug
import load_class
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import pylab
from numpy.random import randint


load = load_class.load(0.1)
augmenter = aug.augmenter(p=1.0)

samples, labels, indices = load.load_testing_set()
print samples.shape
f = pylab.figure()
test_sample = randint(low=0, high=len(samples)-1)

num_col = 10
num_samples = 50
num_row = int(round( num_samples / num_col * 2 ))

images = augmenter.scale_crop(samples)
for i in xrange(num_samples):
    rand_subsample = randint(low=0, high=11)

    f.add_subplot(num_row, num_col, (i % num_col) + ((i/num_col)*2)*num_col)
    pylab.imshow(samples[i][rand_subsample])
    pylab.axis('off')

    f.add_subplot(num_row, num_col, (i % num_col) + ((i/num_col)*2+1)*num_col)
    pylab.imshow(images[i][rand_subsample])
    pylab.axis('off')

    print("{}: {} {}".format(i,i / (2*num_col),i / (2*num_col) + 1))

pylab.show()
   #print("{}, {}; ".format((i % num_col) + (i / num_col) * num_col,(i % num_col) + (i / num_col + 1) * num_col))

# for i in xrange(3):
#     f.add_subplot(2,3,i+1)
#     pylab.imshow(samples[test_sample][i*4])
#     pylab.title("%s" % labels[test_sample])
#     pylab.axis('off')



# images = augmenter.flip_horizontal(samples)
# for i in xrange(6):
#     f.add_subplot(2,6,i+1)
#     pylab.imshow(images[i][0])
#     pylab.axis('off')
# images = augmenter.scale_crop(samples)
# for i in xrange(3):
#     f.add_subplot(2,3,i+4)
#     pylab.imshow(images[test_sample][i*4])
#     pylab.axis('off')


pylab.savefig("voorbeeld")