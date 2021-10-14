from __future__ import print_function
from functools import update_wrapper
import operator

import numpy as np
import scipy as sp

import matplotlib
from matplotlib.widgets import RectangleSelector, Button
from matplotlib import pyplot as plt

from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)

from toolz.curried import *


class PersistentRectangleSelector(RectangleSelector):
    def release(self, event):
        super(PersistentRectangleSelector, self).release(event)
        self.to_draw.set_visible(True)
        self.canvas.draw()   


class SelectorContainer(object):
    def __init__(self, selector):
        self.selector = selector

    def extents(self, event):
        return self.selector.extents

    def quit(self, event):
        plt.close("all")



def select_ROI(img, cmap='gray'):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom = 0.2)  
    ax.imshow(img, cmap=cmap)

    selector = PersistentRectangleSelector(ax,
                                           lambda e1,e2: None,
                                           drawtype = "box",
                                           useblit = False,
                                           button = [1],
                                           spancoords = "data",
                                           interactive = True)

    ax_done = plt.axes([0.45, 0.05, 0.2, 0.075])
    btn_done = Button(ax_done, "Done")
    btn_done.on_clicked(lambda e: plt.close("all"))

    plt.show(block=True)
    xmin, xmax, ymin, ymax =  selector.extents
    return (int(xmin), int(xmax), int(ymin), int(ymax))
    

def equalize_from_ROI(img, roi):
    xmin, xmax, ymin, ymax = roi
    mask = np.zeros(img.shape)
    mask[ymin:ymax, xmin:xmax] = 1
    return exposure.equalize_hist(img, mask = mask)




read_image = io.imread
invert = util.invert
equalize_adaptive = exposure.equalize_adapthist
equalize_hist = exposure.equalize_hist
clear_border = segmentation.clear_border
disk = disk_selem = morphology.disk
binary_opening = morphology.binary_opening
binary_closing = morphology.binary_closing
binary_erosion = morphology.binary_erosion
binary_dilation = morphology.binary_dilation
opening = morphology.opening
closing = morphology.closing
erosion = morphology.erosion
dilation = morphology.dilation
thin = morphology.thin
watershed = segmentation.watershed


@curry
def rescale(scale, img):
    return transform.rescale(img, scale,
                             mode = "constant",
                             preserve_range = True).astype(img.dtype)

def threshold_mean(img):
    return img > filters.threshold_mean(img)

def threshold_triangle(img):
    return img > filters.threshold_triangle(img)

def threshold_otsu(img):
    return img > filters.threshold_otsu(img)

def threshold_li(img):
    return img > filters.threshold_li(img)

def threshold_yen(img):
    return img > filters.threshold_yen(img)

def threshold_isodata(img):
    return img > filters.threshold_isodata(img)

@curry
def threshold_gaussian(block_size, sigma, img):
    return img > filters.threshold_local(img, block_size,
                                         method = "gaussian",
                                         param = sigma)

@curry
def remove_small_objects(min_size, img, **args):
    return morphology.remove_small_objects(img, min_size, **args)

@curry
def remove_small_holes(min_size, img, **args):
    return morphology.remove_small_holes(img, min_size, **args)

@curry
def disk_opening(radius, img):
    return morphology.binary_opening(img, selem = morphology.disk(radius))

@curry
def disk_closing(radius, img):
    return morphology.binary_closing(img, selem = morphology.disk(radius))

@curry
def disk_erosion(radius, img):
    return morphology.binary_erosion(img, selem = morphology.disk(radius))


@curry
def imshowg(img, ax = None):
    """Show image using grayscale color map.
    """
    vmin, vmax = util.dtype_limits(img, clip_negative = True)
    if ax is None:
        ax = plt.imshow(img, cmap = "gray", vmin = vmin, vmax = vmax)
    else:
        ax.imshow(img, cmap = "gray", vmin = vmin, vmax = vmax)
    return ax


@curry
def subregion(bbox, img):
    minr, minc, maxr, maxc = bbox
    return img[minr:maxr, minc:maxc]


@curry
def mask_outside_bbox(bbox, img, background = 0):
    minr, minc, maxr, maxc = bbox
    bkgd = np.array(background).astype(img.dtype)
    mask_img = np.ones_like(img, dtype = img.dtype) * bkgd
    row_slice, col_slice = slice(minr, maxr), slice(minc, maxc)
    mask_img[row_slice, col_slice] = img[row_slice, col_slice].copy()
    return mask_img

@curry
def mask_border(size, img, background = 0):
    nrows, ncols = img.shape
    bkgd = np.array(background).astype(img.dtype)
    mask_img = np.ones_like(img, dtype = img.dtype) * bkgd
    row_slice = slice(size, nrows - size)
    col_slice = slice(size, ncols - size)
    mask_img[row_slice, col_slice] = img[row_slice, col_slice].copy()
    return mask_img
    

@curry
def bbox_mask(bbox, img):
    minr, minc, maxr, maxc = bbox
    mask_img = np.zeros_like(img, dtype = np.bool)
    mask_img[minr:maxr, minc:maxc] = True
    return mask_img

@curry
def image_center(r_hwidth, c_hwidth, img):
    nrows, ncols = img.shape
    rctr, cctr = nrows//2, ncols//2
    minr = max(0, rctr - r_hwidth)
    minc = max(0, cctr - c_hwidth)
    maxr = min(nrows, rctr + r_hwidth)
    maxc = min(ncols, cctr + c_hwidth)
    return img[minr:maxr, minc:maxc]


@curry
def extract_bbox(bbox, img):
    minr, minc, maxr, maxc = bbox
    return img[minr:maxr, minc:maxc]

def inscribed_bbox(bbox):
    minr, minc, maxr, maxc = bbox
    minor_axis = min(maxr-minr, maxc-minc)/2
    center = (minr+maxr)/2, (minc+maxc)/2
    radius = minor_axis * 0.70710678118654757
    iminr = int(center[0] - radius)
    imaxr = int(center[0] + radius)
    iminc = int(center[1] - radius)
    imaxc = int(center[1] + radius)
    return (iminr, iminc, imaxr, imaxc)

    

def pad_to_same_size(img1, img2, mode = "edge"):
    r1, c1 = img1.shape
    r2, c2 = img2.shape
    
    rmax = max(r1,r2)
    cmax = max(c1,c2)
    
    rdiff1, rdiff2 = rmax - r1, rmax - r2
    cdiff1, cdiff2 = cmax - c1, cmax - c2
    
    rpad1 = int(rdiff1/2), int(rdiff1 - rdiff1/2)
    rpad2 = int(rdiff2/2), int(rdiff2 - rdiff2/2)
    
    cpad1 = int(cdiff1/2), int(cdiff1 - cdiff1/2)
    cpad2 = int(cdiff2/2), int(cdiff2 - cdiff2/2)
        
    pimg1 = np.pad(img1, (rpad1, cpad1), mode = mode)
    pimg2 = np.pad(img2, (rpad2, cpad2), mode = mode)    

    offset1 = (rpad1[0], cpad1[0])
    offset2 = (rpad2[0], cpad2[0])
    
    return pimg1, pimg2, offset1, offset2

