from __future__ import print_function

import numpy as np
import scipy as sp

import matplotlib
from matplotlib.widgets import RectangleSelector, Button
from matplotlib import pyplot as plt

from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io)

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
disk_selem = morphology.disk
binary_opening = morphology.binary_opening
binary_erosion = morphology.binary_erosion

@curry
def rescale(scale, img):
    return transform.rescale(img, scale,
                             mode = "constant",
                             preserve_range = True).astype(img.dtype)

def threshold_otsu(img):
    return img > filters.threshold_otsu(img)

def threshold_li(img):
    return img > filters.threshold_li(img)

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
def disk_erosion(radius, img):
    return morphology.binary_erosion(img, selem = morphology.disk(radius))

