import numpy as np

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)


def draw_bboxes(bboxes, ax=None, color='red', linewidth=1, **kw):
    if ax is None:
        ax = plt.gca()
    patches = [mpatches.Rectangle((i[1], i[0]), i[3] - i[1], i[2] - i[0])
               for i in bboxes]
    boxcoll = PatchCollection(patches, **kw)
    boxcoll.set_facecolor('none')
    boxcoll.set_edgecolor(color)
    boxcoll.set_linewidth(linewidth)
    #ax.collections = []
    ax.add_collection(boxcoll)
    return ax

def draw_region_labels(regions, ax=None, fontsize=7, **kw):
    if ax is None:
        ax = plt.gca()
    for region in regions:
        cent = region.centroid
        t = Text(cent[1], cent[0], str(region.label), fontsize=fontsize, **kw)
        t.set_clip_on(True)
        ax.add_artist(t)
    return ax

def colorize_grayscale(img, mask, clr=[1, 0, 0, 0.65]):
    clrimg = color.gray2rgb(util.img_as_float(img), alpha=True)
    clrimg[mask, :] *= clr
    return clrimg

def draw_image_and_labels(img, labeled_img, mask_cmap = "Reds", alpha = 0.35, 
                          fontsize=7, textcolor = "tan", ax = None):
    regions = measure.regionprops(labeled_img)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap = "gray")
    ax.imshow(labeled_img > 0, cmap = mask_cmap, alpha = alpha)
    draw_region_labels(regions, ax, fontsize = fontsize, color = textcolor)
    return plt.gcf(), ax

