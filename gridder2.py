import sys
import os.path
import json
from itertools import product

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import scipy


import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)

from toolz.curried import *
import click
import peakutils

import imgz
import spotzplot




def find_rotation_angle(bimg, theta_range = (-10, 10), ntheta=None, scale=0.1):
    mintheta, maxtheta = min(theta_range), max(theta_range)
    if ntheta is None:
        ntheta = (maxtheta - mintheta) * 4 + 1
    theta = np.linspace(mintheta, maxtheta, ntheta)
    sinogram = transform.radon(transform.rescale(bimg, scale=scale), 
                               theta, circle=False)
    sinogram_max = np.max(sinogram, axis=0)
    peak_indices = peakutils.indexes(sinogram_max, thres=0.999)
    interpolated_peaks = peakutils.interpolate(theta, sinogram_max, 
                                              ind=peak_indices)
    return sinogram, interpolated_peaks[0]
