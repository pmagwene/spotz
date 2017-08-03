from __future__ import print_function
import os.path
import json

import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)

from toolz.curried import *
import click

import imgz, spotzplot

#-------------------------------------------------------------------------------    

def estimate_transform_from_regions(regions1, regions2):
    in1 = set(r.label for r in regions1)
    in2 = set(r.label for r in regions2)
    in1and2 = sorted(list(in1 & in2))
    centroids1 = np.array([r.centroid for r in regions1 if r.label in in1and2])
    centroids2 = np.array([r.centroid for r in regions2 if r.label in in1and2])
    tform = transform.estimate_transform("similarity", np.fliplr(centroids1),
                                         np.fliplr(centroids2))
    return tform


def apply_transform(limg, geomtransform, shape):
    wlimg = transform.warp(limg, geomtransform.inverse, output_shape=shape,
                           preserve_range=True, order=0)
    wlimg = np.round(wlimg).astype(np.int64)
    regions = sort_regions_by_label(measure.regionprops(wlimg))
    return wlimg, regions

def align_regions(labeled_img1, labeled_img2, regions1, regions2):
    tform = estimate_transform_from_regions(regions1, regions2)
    mapped_labels, mapped_regions = apply_transform(labeled_img1, 
                                                    tform, 
                                                    labeled_img2.shape)

    mapped_labeled_img = mapped_labels
    mapped_regions = mapped_regions
