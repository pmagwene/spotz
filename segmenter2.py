from __future__ import print_function
import os.path, sys
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
    

@curry
def threshold_bboxes(bboxes, img, threshold_func = imgz.threshold_li, border=10):
    """Threshold each bbox region independently, stitching together into total image.

    The total image in the logical_or of thresholding each bbox independently.

    border -- gives buffer region around bbox to include for each bbox, allowing bboxes to be
    increased/decreased in size a uniform amount.
    """
    thresh_img = np.zeros_like(img, dtype = np.bool)
    nrows, ncols = img.shape
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        minr, minc = max(0, minr - border), max(0, minc - border)
        maxr, maxc = min(maxr + border, nrows-1), min(maxc + border, ncols - 1)
        local_thresh = threshold_func(img[minr:maxr, minc:maxc])
        thresh_img[minr:maxr, minc:maxc] = np.logical_or(local_thresh, thresh_img[minr:maxr, minc:maxc])
    return thresh_img    


def assign_objects_to_grid(grid_centers, bimg, maxdist = 20):
    regions = measure.regionprops(morphology.label(bimg))
    region_centroids = np.array([region.centroid for region in regions])
    nregions = len(regions)
    kdtree = sp.spatial.cKDTree(region_centroids)
    dists, hits = kdtree.query(np.array(grid_centers), distance_upper_bound = maxdist)
    new_limg = np.zeros_like(bimg, dtype=np.uint32)
    new_regions = []
    for i, hit in enumerate(hits):
        if hit >= nregions:
            new_regions.append(None)
            continue
        region = regions[hit]
        region.label = i + 1
        new_regions.append(region)
        new_limg[region.coords[:,0], region.coords[:,1]] = i + 1
    return new_limg, new_regions


def watershed_segment_bbox(bbox_edges, bbox_seed, include_boundary = True):
    n = np.max(bbox_seed)
    nrows, ncols = bbox_edges.shape
    bbox_seed[1, 1] = n + 1
    bbox_seed[1, ncols - 2] = n + 2
    bbox_seed[nrows - 2, ncols - 2] = n + 3
    bbox_seed[nrows - 2, 1] = n + 4 
    wshed = morphology.watershed(bbox_edges, bbox_seed)
    wshed[wshed != n] = 0
    if include_boundary:
        boundary = segmentation.find_boundaries(wshed, mode = "outer") * n
        wshed += boundary 
    return wshed


def grow_bbox(bbox, amount, nrows, ncols):
    minr, minc, maxr, maxc = bbox   
    minr, minc = max(0, minr - amount), max(0, minc - 1)
    maxr, maxc = min(maxr + amount, nrows-1), min(maxc + amount, ncols - 1)   
    return minr, minc, maxr, maxc   


@curry
def watershed_segment_bboxes(centers, bboxes, img, bimg, seed_width = 5):
    edges = filters.scharr(img)
    seeds = np.zeros_like(img, dtype = int)
    wshed = np.zeros_like(img, dtype = np.uint32)
    nrows, ncols = img.shape

    # watershed each bbox independently
    for i, ctr in enumerate(centers):
        if bboxes[i] is None:
            continue
        ctr = tuple(ctr)
        upctr = ctr[0] - seed_width
        dnctr = ctr[0] + seed_width
        ltctr = ctr[1] - seed_width
        rtctr = ctr[1] + seed_width
        minr, minc, maxr, maxc = bboxes[i]   
        minr, minc = max(0, minr - 1), max(0, minc - 1)
        maxr, maxc = min(maxr + 1, nrows-1), min(maxc + 1, ncols - 1)     
        if np.any(bimg[upctr:dnctr, ltctr:rtctr]):
            seeds[upctr:dnctr, ltctr:rtctr] = bimg[upctr:dnctr, ltctr:rtctr] * i + 1
            bbox_edges = edges[minr:maxr, minc:maxc]
            bbox_seed = seeds[minr:maxr, minc:maxc]
            wshed[minr:maxr, minc:maxc] = watershed_segment_bbox(bbox_edges, bbox_seed)
    return wshed
    




def save_sparse_mask(labeled_img, fname):
    sp.sparse.save_npz(sp.sparse.coo_matrix(labeled_img))



#-------------------------------------------------------------------------------    
@click.command()

@click.option("--opensize",
              help = "Size of element for morphological opening.",
              type = int,
              default = 3,
              show_default = True)
@click.option("--closesize",
              help = "Size of element for morphological closing.",
              type = int,
              default = 3,
              show_default = True)
@click.option("--minhole",
              help = "Minimum hole size (in pixels).",
              type = int,
              default = 25,
              show_default = True)
@click.option("--minobject",
              help = "Minimum object size (in pixels).",
              type = int,
              default = 25,
              show_default = True)
@click.option("--maxdist",
              help = "Max distance btw object and grid centroids (in pixels).",
              type = int,
              default = 30,
              show_default = True)
@click.option("--border",
              help = "bbox border size for thresholding step (in pixels).",
              type = int,
              default = 25,
              show_default = True)
@click.option("--seedwidth",
              help = "half width of watershed seed region (in pixels).",
              type = int,
              default = 5,
              show_default = True)
@click.option('--threshold',
              help = "Thresholding function to use",
              type=click.Choice(['otsu', 'li', "triangle", "mean"]),
              default = "li")
@click.option("--invert/--no-invert",
              help = "Whether to invert the image before analyzing",
              default = True,
              show_default = True)
@click.option("--autoexpose/--no-autoexpose",
              help = "Whether to apply exposure equalization before analyzing",
              default = False,
              show_default = True)
@click.option("--display/--no-display",
              help = "Whether to display segmented objects.",
              default = False,
              show_default = True)
@click.option("-p", "--prefix",
              help = "Prefix for output files",
              type = str,
              default = "MASK",
              show_default = True)
@click.argument("imgfiles",
                nargs = -1,
                type = click.Path(exists = True, dir_okay = False))
@click.argument("gridfile",
                type = click.Path(exists = True, dir_okay = False))
@click.argument("outdir", 
                type = click.Path(exists = True, file_okay = False,
                                  dir_okay = True))

def main(imgfiles, gridfile, outdir, prefix,
         opensize = 3, closesize = 3, minhole = 25, minobject = 25, 
         border=10, maxdist=30, seedwidth=5, threshold="li",
         invert = True, autoexpose = False, display = False):
    """Segment microbial colonies in an image of a pinned plate.

    Input is one or more image files, a JSON "grid file" created by
    the gridder program, and the name of the directory to write the
    segmentation mask to.
    
    Segmentation involves:
    - Inversion (required if dark colonies on light) and auto exposure (optional)
    - Thresholding based on grid
    - Filtering of small objects and holes
    - Morphological closing
    - Watershed dtermination of objects within grid
    """
    threshold_dict = {"otsu" : imgz.threshold_otsu,
                      "li" : imgz.threshold_li,
                      "triangle" : imgz.threshold_triangle,
                      "mean" :  imgz.threshold_mean}
    threshold_func = threshold_dict[threshold]    

    grid_data = json.load(open(gridfile, "r"))
    grid_centers = np.array(grid_data["centers"])
    grid_bboxes = grid_data["bboxes"]

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))

        if invert:
            iimg = imgz.invert(img)
        else:
            iimg = img
        if autoexpose:
            iimg = imgz.equalize_adaptive(iimg)
      

        # threshold
        thresh_img = pipe(iimg,
                        threshold_bboxes(grid_bboxes, threshold_func = threshold_func, border = border),
                        imgz.remove_small_objects(minobject),
                        imgz.remove_small_holes(minhole),
                        imgz.disk_closing(closesize),
                        imgz.disk_opening(opensize),
                        imgz.clear_border)   
     

        filtered_img, filtered_regions = assign_objects_to_grid(grid_centers, thresh_img, maxdist = maxdist) 
        filtered_bboxes = [r.bbox if r else None for r in filtered_regions]

        watershed_img = watershed_segment_bboxes(grid_centers, filtered_bboxes, iimg, thresh_img, seed_width = seedwidth)

        root, _ = os.path.splitext(os.path.basename(imgfile))
        outfile = os.path.join(outdir, "{}-{}.npz".format(prefix, root))
        sp.sparse.save_npz(outfile, sp.sparse.coo_matrix(watershed_img))

        if display:
            fig, ax = plt.subplots(1,1)
            ax.imshow(color.label2rgb(watershed_img, img, bg_label = 0))
            plt.show()

if __name__ == "__main__":
    main()


