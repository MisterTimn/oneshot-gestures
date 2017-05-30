from __future__ import print_function

import numpy as np
import math
from numpy.random import randint, uniform
#from skimage.transform import rescale, rotate, SimilarityTransform, warp
import skimage.transform
import skimage.io
from math import pi,sin,cos

class augmenter(object):
    default_augmentation_params = {
        'zoom_range': (1 / 1.2, 1.2),
        'rotation_range': (-4, 4),
        'shear_range': (0, 0),
        'translation_range': (-8, 8),
        'do_flip': True,
        'allow_stretch': True,
    }

    def __init__(self):
        self.p_flip         = 0.5
        self.max_upscale    = 1.2
        self.max_rotation   = 3
        self.num_subsamples = 12

    def transfMatrix(self, images):
        images_copy = np.empty(images.shape, dtype='float32')
        #np.copyto(images_copy,images)
        for i in xrange(len(images)):
            for j in xrange(self.num_subsamples):
                image = self.perturb(images[i][j],self.default_augmentation_params)
                assert(image.shape==(64,64))
                images_copy[i][j] = image
        return images_copy

    def fast_warp(self, img, tf, output_shape=(64, 64), mode='constant', order=1):
        """
        This wrapper function is faster than skimage.transform.warp
        """
        m = tf.params  # tf._matrix is
        return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)

    def flip_horizontal(self, images):
        samples = uniform(size=len(images))
        images_copy = np.empty(images.shape, dtype='float32')
        np.copyto(images_copy,images)
        for i in xrange(len(images)):
            if (samples[i] <= self.p_flip):
                for j in xrange(self.num_subsamples):
                    images_copy[i][j]=np.fliplr(images_copy[i][j])
        return images_copy


    """
    Scales the image up 0 - 20%
    Then crops a 64x64px image from it
        starting in a randomly selected top left corner

    Return an array of the same shape with the augmented images
    """
    def scale_crop(self, images):
        images_copy = np.empty(images.shape, dtype='float32')
        for i in xrange(len(images)):
            row_scale = uniform(low=1, high=self.max_upscale)
            col_scale = uniform(low=1, high=self.max_upscale)
            x_max = int( round( 64 * (row_scale - 1)))
            y_max = int( round( 64 * (col_scale - 1)))
            p_x = randint(0,x_max) if x_max>0 else 0
            p_y = randint(0,y_max) if y_max>0 else 0
            for j in xrange(self.num_subsamples):
                image = skimage.transform.rescale(images[i][j], (row_scale,col_scale))
                image = image[p_x:p_x+64,p_y:p_y+64]
                images_copy[i][j]=image
                assert(image.shape==(64,64))
        return images_copy

    def find_max_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return wr, hr

    """
    Rotates the image a random numer of degrees [-20, 20]
    It then finds the largest rectangle with data (no black border)
    Crops it and scales it to a 64x64 image

    Returns an array of the same shape with augmented images
    """
    def rotate_crop(self, images):
        images_copy = np.empty(images.shape, dtype='float32')
        for i in xrange(len(images)):
            rot_degree = uniform(low=-self.max_rotation, high=self.max_rotation)
            for j in xrange(self.num_subsamples):
                image = skimage.transform.rotate(images[i][j], rot_degree)
                w, h = image.shape
                w_crop, h_crop = self.find_max_rect(w,h,rot_degree*pi/180)
                image = image[w/2-w_crop/2:w/2+w_crop/2,h/2-h_crop/2:h/2+h_crop/2]
                w_crop, h_crop = image.shape
                w_scale = 64.0 / (w_crop)
                h_scale = 64.0 / (h_crop)
                assert(w_scale==h_scale)
                image = skimage.transform.rescale(image,(w_scale,h_scale))
                images_copy[i][j] = image
                assert(image.shape==(64,64))
        return images_copy


    """
    Code from gihub.com/benanne
    """

    def perturb(self, img, augmentation_params, target_shape=(64, 64), rng=np.random):
        tform_centering = self.build_centering_transform(img.shape, target_shape)
        tform_center, tform_uncenter = self.build_center_uncenter_transforms(img.shape)
        tform_augment = self.random_perturbation_transform(rng=rng, **augmentation_params)
        tform_augment = tform_uncenter + tform_augment + tform_center  # shift to center, augment, shift back (for the rotation/shearing)
        return self.fast_warp(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant').astype('float32')

    def build_centering_transform(self, image_shape, target_shape=(50, 50)):
        rows, cols = image_shape
        trows, tcols = target_shape
        shift_x = (cols - tcols) / 2.0
        shift_y = (rows - trows) / 2.0
        return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))

    def build_center_uncenter_transforms(self, image_shape):
        """
        These are used to ensure that zooming and rotation happens around the center of the image.
        Use these transforms to center and uncenter the image around such a transform.
        """
        center_shift = np.array(
            [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
        tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
        tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
        return tform_center, tform_uncenter

    def fast_warp(self, img, tf, output_shape=(64, 64), mode='edge', order=1):
        """
        This wrapper function is faster than skimage.transform.warp
        """
        m = tf.params  # tf._matrix is
        return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)

    def random_perturbation_transform(self, zoom_range, rotation_range, shear_range, translation_range, do_flip=True,
                                      allow_stretch=False, rng=np.random):
        shift_x = rng.uniform(*translation_range)
        shift_y = rng.uniform(*translation_range)
        translation = (shift_x, shift_y)

        rotation = rng.uniform(*rotation_range)
        shear = rng.uniform(*shear_range)

        if do_flip:
            flip = (rng.randint(2) > 0)  # flip half of the time
        else:
            flip = False

        # random zoom
        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(rng.uniform(*log_zoom_range))
            stretch = np.exp(rng.uniform(*log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(rng.uniform(*log_zoom_range))
            zoom_y = np.exp(rng.uniform(*log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
            # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

        return self.build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

    def build_augmentation_transform(self,zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
        if flip:
            shear += 180
            rotation += 180
            # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
            # So after that we rotate it another 180 degrees to get just the flip.

        tform_augment = skimage.transform.AffineTransform(scale=(1 / zoom[0], 1 / zoom[1]),
                                                          rotation=np.deg2rad(rotation), shear=np.deg2rad(shear),
                                                          translation=translation)

        return tform_augment