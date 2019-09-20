import sys
import os.path
import json
from itertools import product

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import scipy
from scipy import stats, signal

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




def find_rotation_angle(bimg, theta_range = (-10, 10), ntheta=None, scale=0.1):
    mintheta, maxtheta = min(theta_range), max(theta_range)
    if ntheta is None:
        ntheta = (maxtheta - mintheta) * 4 + 1
    theta = np.linspace(mintheta, maxtheta, ntheta)
    sinogram = transform.radon(
                  transform.rescale(bimg, scale=scale, mode = "constant",
                    multichannel=False, anti_aliasing=False), 
                  theta, circle=False)
    sinogram_max = np.max(sinogram, axis=0)
    peak_indices = peakutils.indexes(sinogram_max, thres=0.999)
    interpolated_peaks = peakutils.interpolate(theta, sinogram_max, 
                                              ind=peak_indices)
    return sinogram, interpolated_peaks[0]


def fix_rotation(bimg):
    sinogram, angle = find_rotation_angle(bimg)
    return transform.rotate(bimg, -angle, resize = False, 
              preserve_range = True, mode = "constant").astype(np.bool)


def cubic_kernel(x):
    """Cubic kernel.""" 
    z = (1.0 - np.abs(x)**3)
    z[np.abs(x) >= 1] = 0
    return z  

def reflect_ends(y, h):
    ystart = y[:h]
    yend = y[-h:]
    Y = np.hstack((ystart[::-1], y, yend[::-1]))
    return Y


def fast_smooth(y, h, kernel=cubic_kernel, kinterval=(-1,1)):
    """Uses convolution to smooth y, using a kernel of half-bandwidth h, where h is # of points.

    Like kernel_smooth but assumes uniform interpoint distances so can
    use numpy.convolve for fast results.
    """
    wts = kernel(np.linspace(kinterval[0], kinterval[1], 2*h + 1))
    sumwts = np.sum(wts)
    Y = reflect_ends(y, h)
    smoothY = np.convolve(Y, wts/sumwts, mode='valid')
    return smoothY


def estimate_grid_parameters(bimg, threshold = 0.2, min_dist = 20):
    h = int(min_dist * 0.5)
    rowsums = fast_smooth(np.sum(bimg, axis=1), h)
    rowpks = peakutils.indexes(rowsums, thres = threshold, min_dist = min_dist)
    row_spacing = np.median(rowpks[1:] - rowpks[:-1])

    colsums = fast_smooth(np.sum(bimg, axis=0), h)
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


def bboxes_from_centers(centers, rwidth, cwidth):
    uprow = rwidth//2
    dwrow = rwidth//2
    ltcol = cwidth//2
    rtcol = cwidth//2
    bboxes = []
    for ctr in centers:
        minr = int(ctr[0] - uprow )
        minc = int(ctr[1] - ltcol)
        maxr = int(ctr[0] + dwrow)
        maxc = int(ctr[1] + rtcol)
        bboxes.append((minr, minc, maxr, maxc))
    return bboxes

@curry
def find_grid(nrows, ncols, bimg, pkthresh = 0.1, pkdist = None):
    if pkdist is None:
        r,c = bimg.shape
        pkdist = (r/(1.0 * nrows) + c/(1.0 * ncols)) * 0.25

    row_spacing, col_spacing, radius = estimate_grid_parameters(bimg, threshold = pkthresh, min_dist = pkdist)
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
              type=click.Choice(['otsu', 'li', "triangle", "mean"]),
              default = "otsu",
              show_default = True)
@click.option("--userthresh",
              help = "User specified global threshold value. Used for thresholding when value > 0.",
              type = int,
              default = 0)
@click.option("--opensize",
              help = "Size of element for morphological opening.",
              type = int,
              default = 3,
              show_default = True)
@click.option("--pkthresh",
              help = "Threshold height (relative to max) to be called a peak.",
              type = float,
              default = 0.1,
              show_default = True)
@click.option("--pkdist",
              help = "Minimum distance (in pixels) between grid peaks.",
              type = int,
              default = None,
              show_default = True)
@click.option("--display/--no-display",
              help = "Whether to display found grid.",
              default = False,
              show_default = True)
@click.option("--invert/--no-invert",
              help = "Whether to invert the image before analyzing",
              default = True,
              show_default = True)
@click.option("--rotate/--no-rotate",
              help = "Whether to estimation the rotation angle to make grid rows, cols align with exes",
              default = True,
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
         threshold = "otsu", userthresh=0,
         opensize = 3, pkthresh = 0.1, pkdist = None,
         display = False, invert = False, autoexpose = False, rotate = True):
    """Infer the coordinates of a gridded set of objects in an image.
    
    Grid finding involves three key steps: 

      1. Image thresholding to define foreground vs background and generate a binary image

      2. Morphological opening of the binary image

      3. Inference of the grid coordinates from foreground objects in the binary image.
    
    User can optionally choose to invert and apply exposure equalization to the input image. Inversion is required when the objects of interest are dark objects on a light background (e.g. transparency scanning).
    """

    threshold_dict = {"otsu" : imgz.threshold_otsu,
                      "li" : imgz.threshold_li,
                      "triangle" : imgz.threshold_triangle,
                      "mean" :  imgz.threshold_mean,
                      "yen" : imgz.threshold_yen}
    threshold_func = threshold_dict[threshold]

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))

        # invert and autoexpose
        if invert:
            iimg = imgz.invert(img)
        else:
            iimg = img
        if autoexpose:
            iimg = imgz.equalize_adaptive(iimg)


        # initial thresholding and rotation correction
        if userthresh > 0:
            rbimg = iimg > userthresh
        else:
            rbimg = pipe(iimg, threshold_func)
        rbimg = pipe(rbimg,
                  imgz.disk_opening(opensize), 
                  imgz.clear_border)

        angle = 0
        if rotate:
            _, angle = find_rotation_angle(rbimg)
            rbimg = fix_rotation(rbimg)
            img = transform.rotate(img, -angle, resize = False, 
                                preserve_range = True, mode = "constant")  

        try:
            # find the grid
            grid = find_grid(rows, cols, rbimg, pkthresh, pkdist)
        except RuntimeError:
            print("No grid found in {}".format(imgfile))
            if display:
                fig, ax = plt.subplots()
                ax.imshow(img, cmap = "gray")
                ax.imshow(rbimg, cmap = "Reds", alpha = 0.45)
                plt.show()      
                sys.exit(1)    

        grid_data = dict(bboxes = grid.bboxes, centers = grid.centers.tolist(),
                        row_width = grid.row_width, col_width = grid.col_width,
                        rotation_angle = angle)

        s = json.dumps(grid_data, indent = 1)

        root, _ = os.path.splitext(os.path.basename(imgfile))
        outfile = os.path.join(outdir, "{}-{}.json".format(prefix, root))
        with open(outfile, "w") as f:
            f.write(s)
        
        if display:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap = "gray")
            ax.imshow(rbimg, cmap = "Reds", alpha = 0.45)
            spotzplot.draw_bboxes(grid.bboxes, ax)
            plt.show()
    

if __name__ == "__main__":
    main()


