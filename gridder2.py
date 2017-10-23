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


def fix_rotation(bimg):
    sinogram, angle = find_rotation_angle(bimg)
    return transform.rotate(bimg, -angle, resize = False, preserve_range = True).astype(np.bool)



def estimate_grid_parameters(bimg, threshold = 0.2, min_dist = 20):
    rowsums = np.sum(bimg, axis=1)
    rowpks = peakutils.indexes(rowsums, thres = threshold, min_dist = min_dist)
    row_spacing = np.median(rowpks[1:] - rowpks[:-1])

    colsums = np.sum(bimg, axis=0)
    colpks = peakutils.indexes(colsums, thres = threshold, min_dist = min_dist)
    col_spacing = np.median(colpks[1:] - colpks[:-1])

    labeled_img = morphology.label(bimg)
    regions = measure.regionprops(labeled_img)
    radii = [region.equivalent_diameter/2.0 for region in regions]
    radius = np.median(radii)
    return row_spacing, col_spacing, radius


def construct_grid_template(nrows, ncols, row_spacing, col_spacing, radius):
    rwidth = int(nrows * row_spacing)
    cwidth = int(ncols * col_spacing)
    template = np.zeros((rwidth, cwidth), dtype = np.uint16)
    
    row_centers = row_spacing * np.arange(1, nrows+1) - row_spacing/2.0
    col_centers = col_spacing * np.arange(1, ncols+1) - col_spacing/2.0
    
    row_centers = row_centers.astype(np.int)
    col_centers = col_centers.astype(np.int)

    radius = int(radius)

    centers = list(product(row_centers, col_centers))
    for i, ctr in enumerate(centers):
        upctr = ctr[0] - radius
        dnctr = ctr[0] + radius + 1
        ltctr = ctr[1] - radius
        rtctr = ctr[1] + radius + 1
        
        template[upctr:dnctr, ltctr:rtctr] = morphology.disk(radius) * (i+1)
    
    return template, row_centers, col_centers


def compute_shift(x, y):
    """Compute shift of signal y that maximizes cross correlation between two signals.

    shift < 0 means that y starts 'shift' time steps before x 
    shift > 0 means that y starts 'shift' time steps after x

    Algorithm from http://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
    """
    c = scipy.signal.correlate(x, y, mode = "same")
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - (np.argmax(c) - 1)
    return shift     


def estimate_grid_offset(bimg, template):
    btemplate = template.astype(np.bool)
    pbimg, ptemplate, offset1, offset2 = imgz.pad_to_same_size(bimg, btemplate)

    i_rowsums = np.sum(pbimg, axis=1)
    i_colsums = np.sum(pbimg, axis=0)

    t_rowsums = np.sum(ptemplate, axis=1)
    t_colsums = np.sum(ptemplate, axis=0)

    row_shift = compute_shift(i_rowsums, t_rowsums)
    col_shift = compute_shift(i_colsums, t_colsums)

    return offset1, offset2, (row_shift, col_shift)

def estimate_grid_ctrs(bimg, template, template_centers):
    ctrs = np.array(template_centers)
    offset1, offset2, shifts = estimate_grid_offset(bimg, template)
    newctrs = ctrs + np.array(offset1) + np.array(offset2) - np.array(shifts) 
    return newctrs

@curry
def estimate_grid(nrows, ncols, bimg):
    row_spacing, col_spacing, radius = estimate_grid_parameters(bimg)
    template, trow_centers, tcol_centers = construct_grid_template(nrows, ncols, row_spacing, col_spacing, radius)
    template_centers = np.array(list(product(trow_centers, tcol_centers)))
    class GridData:
        pass
    g = GridData()
    g.row_width = row_spacing
    g.col_width = col_spacing
    g.centers = estimate_grid_ctrs(bimg, template, template_centers)
    g.bboxes = bboxes_from_centers(g.centers, g.row_width, g.col_width)
    return g

            
def bboxes_from_centers(centers, rwidth, cwidth):
    uprow = rwidth/2
    dwrow = rwidth/2#- uprow
    ltcol = cwidth/2
    rtcol = cwidth/2#- ltcol 
    bboxes = []
    for ctr in centers:
        minr = int(ctr[0] - uprow )
        minc = int(ctr[1] - ltcol)
        maxr = int(ctr[0] + dwrow)
        maxc = int(ctr[1] + rtcol)
        bboxes.append((minr, minc, maxr, maxc))
    return bboxes

@curry
def threshold_grid_units(bboxes, img, threshold_func = imgz.threshold_li, border=10):
    """Threshold each grid unit independently.
    """
    thresh_img = np.zeros_like(img, dtype = np.bool)
    nrows, ncols = img.shape
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        minr, minc = max(0, minr - border), max(0, minc - border)
        maxr, maxc = min(maxr + border, nrows-1), min(maxc + border, ncols - 1)
        local_thresh = threshold_func(img[minr:maxr, minc:maxc])
        #thresh_img[minr:maxr, minc:maxc] = local_thresh
        thresh_img[minr:maxr, minc:maxc] = np.logical_or(local_thresh, thresh_img[minr:maxr, minc:maxc])
    return thresh_img
