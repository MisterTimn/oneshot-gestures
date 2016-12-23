from __future__ import print_function

import numpy as np
from numpy.random import randint, uniform
from skimage.transform import rescale

class augmenter(object):
    def __init__(self, p=0):
        self.p = p




    def flip_horizontal(self, images):
        samples = uniform(size=len(images))
        images_copy = np.empty(images.shape)
        np.copyto(images_copy,images)
        for i in xrange(len(images)):
            if ( samples[i] <= self.p ):
                images_copy[i][0]=np.flipud(images_copy[i][0])
        return images_copy


    """
    Scales the image up 0 - 20%
    Then crops a 64x64 pixel image from it
        starting in a randomly selected top left corner

    Return an array of the same shape with the augmented images
    """
    def scale_crop(self, images):
        images_copy = np.empty(images.shape)
        for i in xrange(len(images)):
            row_scale = uniform(low=1, high=1.2)
            col_scale = uniform(low=1, high=1.2)
            x_max = int( round( 64 * (row_scale - 1)))
            y_max = int( round( 64 * (col_scale - 1)))
            p_x = randint(0,x_max) if x_max>0 else 0
            p_y = randint(0,y_max) if y_max>0 else 0
            #print("row: {:.2f}, col: {:.2f}, x: {}, y: {}".format(row_scale,col_scale,p_x,p_y))
            for j in xrange(12):
                image = rescale(images[i][j], (row_scale,col_scale))
                image = image[p_x:p_x+64,p_y:p_y+64]
                images_copy[i][j]=image
                assert(image.shape==(64,64))
        return images_copy