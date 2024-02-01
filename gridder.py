import sys
import os.path
import json
from itertools import product

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import scipy

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)
import peakutils

from toolz.curried import *
import click

import imgz
import spotzplot

#-------------------------------------------------------------------------------    
    
def find_grid_bboxes(binary_img, min_gap, min_n, nrows, ncols, useouter = True):
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

    if useouter:
        row_centers, col_centers = grid_centers_from_outermost(row_centers, col_centers, nrows, ncols)

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
    lbreaks = [0] + rbreaks.tolist()
    rbreaks = rbreaks.tolist() + [len(vals)-1]
    intervals = zip(lbreaks, rbreaks)
    return vals, intervals


def interval_means(vals, intervals):
    """Find means of connected intervals.
    """
    return [np.mean(vals[i[0]:i[1]]) for i in intervals]

def median_center_spacing(ctrs):
    spacing = np.array(ctrs[1:]) - np.array(ctrs[:-1])
    return np.median(np.round(spacing))
    

def estimate_grid_centers(centroids, min_gap, min_n):
    """From centroids of gridded objects, estimate the centers pts of grid.
    """
    row_vals, row_int = connected_intervals(centroids[:,0], min_gap)
    col_vals, col_int = connected_intervals(centroids[:,1], min_gap)

    valid_rows = [i for i in row_int if (i[1] - i[0]) >= min_n]
    valid_cols = [i for i in col_int if (i[1] - i[0]) >= min_n]

    row_centers = interval_means(row_vals, valid_rows)
    col_centers = interval_means(col_vals, valid_cols)

    return np.round(row_centers).astype(int), np.round(col_centers).astype(int)

def grid_centers_from_outermost(row_centers, col_centers, nrows, ncols):
    row_spacing = median_center_spacing(row_centers)
    col_spacing = median_center_spacing(col_centers)

    top = row_centers[0]
    row_centers = top + (np.arange(nrows) * row_spacing)
    left = col_centers[0]
    col_centers = left + (np.arange(ncols) * col_spacing)
    return np.round(row_centers).astype(int), np.round(col_centers).astype(int)


def make_grid_from_dims(row_dim, col_dim, nrows, ncols, row_offset = 0, col_offset = 0):
    row_borders = np.arange(0, nrows + 1) * row_dim + row_offset
    col_borders = np.arange(0, ncols + 1) * col_dim + col_offset
    row_pairs = zip(row_borders[:-1], row_borders[1:])
    col_pairs = zip(col_borders[:-1], col_borders[1:])
    rc_pairs = product(row_pairs, col_pairs)
    bboxes = [(p[0][0], p[1][0], p[0][1], p[1][1]) for p in rc_pairs]
    return bboxes







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

    row_borders = np.round(row_borders).astype(int)
    col_borders = np.round(col_borders).astype(int)

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
              min_gap = None, min_n = None, useouter = True):

    rdim, cdim = binary_img.shape
    if min_gap is None:
        min_gap = int(min(rdim/nrows * 0.2, cdim/ncols * 0.2))
    if min_n is None:
        min_n = int(0.5 * min(nrows, ncols))

    bboxes, centers = find_grid_bboxes(binary_img, min_gap, min_n, nrows, ncols, useouter = useouter)
    total_bbox = (bboxes[0][0], bboxes[0][1], bboxes[-1][2], bboxes[-1][3])
    unit_height = bboxes[0][2] - bboxes[0][0]
    unit_width = bboxes[0][3] - bboxes[0][1]
    return dict(bboxes = bboxes, centers = centers,
                min_gap = min_gap, min_n = min_n, total_bbox = total_bbox,
                unit_height = unit_height, unit_width = unit_width,
                unit_area = unit_height * unit_width)


def threshold_grid_units(grid_data, img, threshold_func = imgz.threshold_otsu):
    """Threshold each grid unit independently.
    """
    timg = np.zeros_like(img, dtype = bool)

    #  threshold each grid unit independently
    for i, ctr in enumerate(grid_data["centers"]):
        ctr = tuple(ctr)
        bbox = grid_data["bboxes"][i]
        minr, minc, maxr, maxc = bbox
        lthresh = threshold_func(img[minr:maxr, minc:maxc])
        timg[minr:maxr, minc:maxc] = lthresh

    return timg





def find_grid_rotation(bimg, theta_range = (-10, 10), ntheta=None, scale=0.1):
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
    
def estimate_grid_parameters(bimg, threshold = 0.1, min_dist = 20):
    rowsums = np.sum(bimg, axis=0)
    rowpks = peakutils.indexes(rowsums, thres = threshold, min_dist = min_dist)
    row_spacing = np.median(rowpks[1:] - rowpks[:-1])

    colsums = np.sum(bimg, axis=1)
    colpks = peakutils.indexes(colsums, thres = threshold, min_dist = min_dist)
    col_spacing = np.median(colpks[1:] - colpks[:-1])

    labeled_img = morphology.label(bimg)
    regions = measure.regionprops(labeled_img)
    radii = [region.equivalent_diameter/2.0 for region in regions]
    radius = np.median(radii)
    return row_spacing, col_spacing, radius

def construct_grid_template(nrows, ncols, row_spacing, col_spacing, radius):
    rwidth = nrows * row_spacing
    cwidth = ncols * col_spacing
    template = np.zeros((rwidth, cwidth), dtype = np.uint16)
    
    row_centers = row_spacing * np.arange(1, nrows+1) - row_spacing/2
    col_centers = col_spacing * np.arange(1, ncols+1) - col_spacing/2
    
    centers = list(product(row_centers, col_centers))
    for i, ctr in enumerate(centers):
        upctr = ctr[0] - radius
        dnctr = ctr[0] + radius + 1
        ltctr = ctr[1] - radius
        rtctr = ctr[1] + radius + 1
        
        template[upctr:dnctr, ltctr:rtctr] = morphology.disk(radius) * (i+1)
    
    return template, row_centers, col_centers

def pad_to_same_size(img1, img2, mode = "edge"):
    r1, c1 = img1.shape
    r2, c2 = img2.shape
    
    rmax = max(r1,r2)
    cmax = max(c1,c2)
    
    rdiff1, rdiff2 = rmax - r1, rmax - r2
    cdiff1, cdiff2 = cmax - c1, cmax - c2
    
    rpad1 = rdiff1/2, rdiff1 - rdiff1/2
    rpad2 = rdiff2/2, rdiff2 - rdiff2/2
    
    cpad1 = cdiff1/2, cdiff1 - cdiff1/2
    cpad2 = cdiff2/2, cdiff2 - cdiff2/2    
        
    pimg1 = util.pad(img1, (rpad1, cpad1), mode = mode)
    pimg2 = util.pad(img2, (rpad2, cpad2), mode = mode)    

    offset1 = (rpad1[0], cpad1[0])
    offset2 = (rpad2[0], cpad2[0])
    
    return pimg1, pimg2, offset1, offset2


 
def cross_correlation_using_fft(x, y):
    """Calculate cross correlation between two signals using FFT.

    Algorithm from http://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
    """
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
 

def compute_shift_orig(x, y):
    """Compute shift of signal y that maximizes cross correlation between two signals.

    shift < 0 means that y starts 'shift' time steps before x 
    shift > 0 means that y starts 'shift' time steps after x

    Algorithm from http://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
    """
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift    

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
    btemplate = template.astype(bool)
    pbimg, ptemplate, offset1, offset2 = pad_to_same_size(bimg, btemplate)

    i_rowsums = np.sum(pbimg, axis=0)
    i_colsums = np.sum(pbimg, axis=1)

    t_rowsums = np.sum(ptemplate, axis=0)
    t_colsums = np.sum(ptemplate, axis=1)

    row_shift = compute_shift(i_rowsums, t_rowsums)
    col_shift = compute_shift(i_colsums, t_colsums)

    return offset1, offset2, (row_shift, col_shift)

def estimate_grid_ctrs(bimg, template, template_centers):
    ctrs = np.array(template_centers)
    offset1, offset2, shifts = estimate_grid_offset(bimg, template)
    newctrs = ctrs - np.array(offset1) + np.array(offset2) + np.array(shifts)
    return newctrs






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
        oimg = np.copy(img)
        if invert:
            img = imgz.invert(img)
        if autoexpose:
            img = imgz.equalize_adaptive(img)

        binary_img, selem_size = threshold_and_open(img, threshold_func)

        try:
            grid_data = find_grid(binary_img, rows, cols, min_gap, min_n)
        except RuntimeError:
            print("No grid found in {}".format(imgfile))
            if display:
                fig, ax = plt.subplots()
                ax.imshow(oimg, cmap = "gray")
                ax.imshow(binary_img, cmap = "Reds", alpha = 0.45)
                plt.show()      
                sys.exit(1)      
        s = json.dumps(grid_data, indent = 1)

        root, _ = os.path.splitext(os.path.basename(imgfile))
        outfile = os.path.join(outdir, "{}-{}.json".format(prefix, root))
        with open(outfile, "w") as f:
            f.write(s)
        
        if display:
            fig, ax = plt.subplots()
            ax.imshow(oimg, cmap = "gray")
            ax.imshow(binary_img, cmap = "Reds", alpha = 0.45)
            spotzplot.draw_bboxes(grid_data["bboxes"], ax)
            plt.show()
    

if __name__ == "__main__":
    main()
