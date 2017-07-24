import sys
import os.path
import json
from itertools import product

import numpy as np

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)

from toolz.curried import *
import click

import imgz
import spotzplot

#-------------------------------------------------------------------------------    
    
def find_grid_bboxes(binary_img, min_gap, min_n):
    """Find gridded objects in a binary image.

    * min_gap is the minimum gap, in pixels, between objects in grid
    * min_n is the minimum number of objects found in a row or column
      to count as a valid grid dimension
    """
    labeled_img = morphology.label(binary_img)
    regions = measure.regionprops(labeled_img)
    if not len(regions):
        raise RuntimeError("No objects found in image.")

    centroids = np.vstack([r.centroid for r in regions])
    row_centers, col_centers = estimate_grid_centers(centroids, min_gap, min_n)

    if not(len(row_centers)) or not(len(col_centers)):
        raise RuntimeError("No grid found in image.")
    
    grid_centers = list(product(row_centers, col_centers))
    grid_bboxes = grid_bboxes_from_centers(row_centers, col_centers, labeled_img.shape)
    return grid_bboxes, grid_centers
    

def connected_intervals(vals, min_gap):
    """Find intervals where distance between successive values > min_gap
    """
    vals = np.sort(vals)
    dist = vals[1:] - vals[:-1]
    rbreaks = np.argwhere(dist > min_gap).flatten() + 1
    lbreaks = rbreaks + 1
    lbreaks = [0] + lbreaks.tolist()
    rbreaks = rbreaks.tolist() + [len(vals)-1]
    intervals = zip(lbreaks, rbreaks)
    return vals, intervals


def interval_means(vals, intervals):
    """Find means of connected intervals.
    """
    return [np.mean(vals[i[0]:i[1]]) for i in intervals]
    

def estimate_grid_centers(centroids, min_gap, min_n):
    """From centroids of gridded objects, estimate the centers pts of grid.
    """
    row_vals, row_int = connected_intervals(centroids[:,0], min_gap)
    col_vals, col_int = connected_intervals(centroids[:,1], min_gap)

    valid_rows = [i for i in row_int if (i[1] - i[0]) > min_n]
    valid_cols = [i for i in col_int if (i[1] - i[0]) > min_n]

    row_centers = interval_means(row_vals, valid_rows)
    col_centers = interval_means(col_vals, valid_cols)

    return np.round(row_centers).astype(np.int), np.round(col_centers).astype(np.int)


def grid_bboxes_from_centers(row_centers, col_centers, shape):
    """Estimate bounding boxes of grid elements.

    row_centers and col_centers are the center points of each grid element
    shape is the total size of the area under consideration (usually image.shape)
    """
    row_centers = np.asarray(row_centers)
    col_centers = np.asarray(col_centers)
    maxr, maxc = shape
    
    row_dists = 0.5 * (row_centers[1:] - row_centers[:-1])
    col_dists = 0.5 * (col_centers[1:] - col_centers[:-1])

    rowFirst = [max(0, row_centers[0] - row_dists[0])]
    rowLast =  [min(maxr, row_centers[-1] + row_dists[-1])]
    row_borders = np.concatenate((rowFirst,
                                  row_centers[:-1] + row_dists,
                                  rowLast))

    colFirst = [max(0, col_centers[0] - col_dists[0])]
    colLast =  [min(maxc, col_centers[-1] + col_dists[-1])]
    col_borders = np.concatenate((colFirst,
                                  col_centers[:-1] + col_dists,
                                  colLast))

    row_borders = np.round(row_borders).astype(np.int)
    col_borders = np.round(col_borders).astype(np.int)

    row_pairs = zip(row_borders[:-1], row_borders[1:])
    col_pairs = zip(col_borders[:-1], col_borders[1:])

    rc_pairs = product(row_pairs, col_pairs)
    bboxes = [(p[0][0], p[1][0], p[0][1], p[1][1]) for p in rc_pairs]
    return bboxes


def threshold_and_open(img,
                       threshold_func = imgz.threshold_otsu, selem_size = None,
                       max_size = 7):
    """Threshold image and apply binary opening.
    """
    min_dim, max_dim = min(img.shape), max(img.shape)
    if selem_size is None:
        selem_size = int(round(min(max_size, min_dim/100. + 1)))
    binary_img = threshold_func(img)
    binary_img = imgz.disk_opening(selem_size, binary_img)
    return binary_img, selem_size
    

def find_grid(binary_img, nrows, ncols,
              min_gap = None, min_n = None):

    rdim, cdim = binary_img.shape
    if min_gap is None:
        min_gap = int(min(rdim/nrows * 0.2, cdim/ncols * 0.2))
    if min_n is None:
        min_n = int(0.5 * min(nrows, ncols))

    bboxes, centers = find_grid_bboxes(binary_img, min_gap, min_n)
    return dict(bboxes = bboxes, centers = centers,
                min_gap = min_gap, min_n = min_n)


#-------------------------------------------------------------------------------    

@click.command()
@click.option("-r", "--rows",
              help = "Number of rows in grid",
              type = int,
              default = 8,
              show_default = True)
@click.option("-c", "--cols",
              help = "Number of cols in grid",
              type = int,
              default = 12,
              show_default = True)
@click.option('--threshold',
              help = "Thresholding function to use",
              type=click.Choice(['otsu', 'li', "isodata"]),
              default = "otsu")
@click.option("--elemsize",
              help = "Size of element for morphological opening.",
              type = int,
              default = None,
              show_default = True)
@click.option("--min_gap",
              help = "Mininum gap (in pixels) between objects in grid",
              type = int,
              show_default = True,
              default = None)
@click.option("--min_n",
              help = "Mininum number of objects found to count as dimension of grid.""",
              type = int,
              show_default = True,
              default = None)
@click.option("--display/--no-display",
              help = "Whether to display found grid.",
              default = False,
              show_default = True)
@click.option("--invert/--no-invert",
              help = "Whether to invert the image before analyzing",
              default = False,
              show_default = True)
@click.option("--autoexpose/--no-autoexpose",
              help = "Whether to apply exposure equalization before analyzing",
              default = False,
              show_default = True)
@click.option("-p", "--prefix",
              help = "Prefix for output files",
              type = str,
              default = "GRID",
              show_default = True)
@click.argument("imgfiles",
                type = click.Path(exists = True),
                nargs = -1)
@click.argument("outdir", 
                type = click.Path(exists = True, file_okay = False,
                                  dir_okay = True))
def main(imgfiles, outdir, rows, cols, prefix = "grid",
         threshold = "otsu", elemsize = None, min_gap = None, min_n = None,
         display = False, invert = False, autoexpose = False):
    """Infer the coordinates of a gridded set of objects in an image.
    
    Grid finding involves three key steps: 

      1. Image thresholding to define foreground vs background and
      generate a binary image

      2. Morphological opening of the binary image

      3. Inference of the grid coordinates from foreground objects in
      the binary image.
    
    User can optionally choose to invert and apply exposure equalization to
    the input image. Inversion is required when the objects of
    interest are dark objects on a light background (e.g. transparency
    scanning).
    """

    threshold_dict = {"otsu":imgz.threshold_otsu,
                      "li":imgz.threshold_li,
                      "isodata":imgz.threshold_isodata}
    threshold_func = threshold_dict[threshold]

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))
        if invert:
            img = imgz.invert(img)
        if autoexpose:
            img = imgz.equalize_adaptive(img)

        binary_img, selem_size = threshold_and_open(img, threshold_func)

        try:
            grid_data = find_grid(binary_img, rows, cols, min_gap, min_n)
        except RuntimeError:
            print("No grid found in {}".format(imgfile))
            sys.exit(1)
        s = json.dumps(grid_data, indent = 1)

        root, _ = os.path.splitext(os.path.basename(imgfile))
        outfile = os.path.join(outdir, "{}-{}.json".format(prefix, root))
        with open(outfile, "w") as f:
            f.write(s)
        
        if display:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap = "gray")
            ax.imshow(binary_img, cmap = "Reds", alpha = 0.45)
            spotzplot.draw_bboxes(grid_data["bboxes"], ax)
            plt.show()
    

if __name__ == "__main__":
    main()
