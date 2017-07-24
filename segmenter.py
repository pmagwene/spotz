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
    

def bbox_to_poly(bbox):
    minr, minc, maxr, maxc = bbox
    pt1 = (minr, minc)
    pt2 = (minr, maxc)
    pt3 = (maxr, maxc)
    pt4 = (maxr, minc)
    return np.array([pt1, pt2, pt3, pt4])
    

def region_encloses_grid_center(region, grid_centers):
    poly = bbox_to_poly(region.bbox)
    hits = measure.points_in_poly(grid_centers, poly)
    if np.sometrue(hits) and (np.sum(hits) == 1):
        return True, np.argwhere(hits).flatten().tolist()
    else:
        return False, None


def keep_regions(labeled_img, labels):
    masked_img = np.in1d(labeled_img.ravel(), np.asarray(labels))
    masked_img.shape = labeled_img.shape
    filtered_img = np.where(masked_img, labeled_img, np.zeros_like(labeled_img))
    return filtered_img

def remove_regions(labeled_img, labels):
    masked_img = np.logical_not(np.in1d(labeled_img.ravel(), np.asarray(labels)))
    masked_img.shape = labeled_img.shape
    filtered_img = np.where(masked_img, labeled_img, np.zeros_like(labeled_img))
    return filtered_img
    

def filter_objects_by_grid(binary_img, grid_centers):
    """Filter objects in binary image whether they are consistent with grid geometry.
    
    The criteria for "consistent" with grid geometry is whether the
    bounding box for an object of interest include the center points
    of at least one object in the labeled.
    """
    labeled_img = morphology.label(binary_img)
    regions = measure.regionprops(labeled_img)

    filtered_regions = []
    grid_positions = []
    for region in regions:
        in_grid, grid_hit = region_encloses_grid_center(region, grid_centers)
        if in_grid:
            filtered_regions.append(region)
            grid_positions.append(min(grid_hit))

    filtered_img = keep_regions(labeled_img, [r.label for r in filtered_regions])

    # masked_img = np.in1d(labeled_img.ravel(), [r.label for r in filtered_regions])
    # masked_img.shape = labeled_img.shape
    # filtered_img = np.where(masked_img, labeled_img, np.zeros_like(labeled_img))

    # filtered_img = np.where(np.isin(labeled_img, [r.label for r in filtered_regions]),
    #                         labeled_img,
    #                         np.zeros_like(labeled_img))
    
    for i, region in enumerate(filtered_regions):
         filtered_img[region.coords[:, 0], region.coords[:, 1]] = grid_positions[i] + 1

    return filtered_img

def filter_by_eccentricity(labeled_img, limit):
    regions = measure.regionprops(labeled_img)
    filtered_regions = [region for region in regions if region.eccentricity < limit]
    return keep_regions(labeled_img, [region.label for region in filtered_regions])
            

def save_sparse_mask(labeled_img, fname):
    sp.sparse.save_npz(sp.sparse.coo_matrix(labeled_img))
        

def segment_image(img, grid_data, 
        threshold = "local", blocksize = None, sigma = None,
        elemsize = None, min_hole = None, min_object = None,
        max_eccentricity = 0.75,
        clear_border = False, invert = False, autoexpose = False):
    
    
    min_dim, max_dim = min(img.shape), max(img.shape)

    if invert:
        img = imgz.invert(img)
    if autoexpose:
        img = imgz.equalize_adaptive(img)

    # Thresholding
    #
    threshold_dict = {"local":imgz.threshold_gaussian,
                      "otsu":imgz.threshold_otsu,
                      "li":imgz.threshold_li,
                      "isodata":imgz.threshold_isodata}

    threshold_func = threshold_dict[threshold]
    if threshold == "local":
        if blocksize is None:
            blocksize = 3
        if sigma is None:
            sigma = 3
        threshold_func = threshold_func(blocksize, sigma)

    binary_img = threshold_func(img)
    
    # Morphological opening
    #
    if elemsize is None:
        elemsize = int(round(min(3, min_dim/100. + 1)))
    binary_img = imgz.disk_opening(elemsize, binary_img)

    # Filter holes, small objects, border
    #
    if min_hole is None:
        min_hole = int(max(1, min_dim * 0.02)**2)
    if min_object is None:
        min_object = int(max(1, min_dim * 0.005)**2)

    binary_img = pipe(binary_img,
                      imgz.remove_small_objects(min_object),
                      imgz.remove_small_holes(min_hole))

    if clear_border:
        binary_img = imgz.clear_border(binary_img)

    # Filter and relabel based on grid
    #
    labeled_img = filter_objects_by_grid(binary_img, grid_data["centers"])

    # Filter by final region properties
    # 
    labeled_img = filter_by_eccentricity(labeled_img, max_eccentricity)
    
    return labeled_img, measure.regionprops(labeled_img)



#-------------------------------------------------------------------------------    

@click.command()
@click.option('--threshold',
              help = "Thresholding function to use",
              type=click.Choice(["local", "otsu", "li", "isodata"]),
              default = "local")
@click.option("--blocksize",
              help = "Size of pixel neighborhood for local thresholding. Must be odd.",
              type = int,
              default = None)
@click.option("--sigma",
              help = "Standard deviation of gaussian local thresholding filter.",
              type = int,
              default = None)
@click.option("--elemsize",
              help = "Size of element for morphological opening.",
              type = int,
              default = None,
              show_default = True)
@click.option("--min-hole",
              help = "Minimum hole size (in pixels).",
              type = int,
              default = None)
@click.option("--min-object",
              help = "Minimum object size (in pixels).",
              type = int,
              default = None)
@click.option("--max-eccentricity",
              help = "Maximum eccentricity of objects.",
              type = float,
              default = 0.65,
              show_default = True)
@click.option("--clear-border/--keep-border",
              help = "Remove objects that touch the border of the image",
              default = True,
              show_default = True)
@click.option("--invert/--no-invert",
              help = "Whether to invert the image before analyzing",
              default = False,
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
         threshold = "local", blocksize = None, sigma = None,
         elemsize = None, min_hole = None, min_object = None, 
         max_eccentricity = 0.65,
         clear_border = False, invert = False, autoexpose = False,
         display = False):
    """Segment microbial colonies in an image of a pinned plate.

    Input is one or more image files, a JSON "grid file" created by
    the gridder program, and the name of the directory to write the
    segmentation mask to.
    
    Segmentation involves:
    - Inversion (required if dark colonies on light) and auto exposure (optional)
    - Thresholding
    - Morphological opening
    - Filtering of small objects and holes
    - Clearing the border region of objects (optional)
    - Filtering objects according to whether they match the grid geometry

    Matches to grid geometry are defined in terms of whether an
    object's bounding box includings the center of one of the grid
    elements.

    """

    grid_data = json.load(open(gridfile, "r"))

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))
        labeled_img, regions = segment_image(img, grid_data, 
                                    threshold = threshold, blocksize = blocksize,
                                    sigma = sigma, elemsize = elemsize,
                                    min_hole = min_hole, min_object = min_object,
                                    max_eccentricity = max_eccentricity,
                                    clear_border = clear_border,
                                    invert = invert, autoexpose = autoexpose)
    
        root, _ = os.path.splitext(os.path.basename(imgfile))
        outfile = os.path.join(outdir, "{}-{}.npz".format(prefix, root))
        sp.sparse.save_npz(outfile, sp.sparse.coo_matrix(labeled_img))

        if display:
            fig, ax = spotzplot.draw_image_and_labels(img, labeled_img)
            plt.show()

    

if __name__ == "__main__":
    main()
