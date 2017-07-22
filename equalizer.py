import os.path
import json
import warnings

import numpy as np
from skimage import (transform, io, exposure, util)
import tifffile as TIFF

import click


def load_ROI_dict(fname):
    with open(fname, "r") as f:
        roidict = json.load(f)
        return roidict

def equalize_from_ROI(img, roi_bbox):
    mask = np.zeros(img.shape)
    minr, minc, maxr, maxc = roi_bbox
    mask[minr:maxr, minc:maxc] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        equalized_img = util.img_as_uint(exposure.equalize_hist(img, mask = mask))
    return equalized_img

#--------------------------------------------------------------------------------

@click.command()
@click.argument("roifile", 
    type = click.Path(exists=True))
@click.argument("input",  
    type = click.Path(exists=True,
                      dir_okay = False))
@click.argument("output",  
    type = click.Path())
@click.option("-e", "--extension",
              help = "File extension type.",
              default = "*.tif")
@click.option("--prefix",
              help = "Prefix to prepend to equalized file names.",
              default = "EQ")
def main(roifile, input, output, extension, prefix):
    """Use ROI to equalize and image. .
    """

    roi_dict = load_ROI_dict(roifile)
    roi_bbox = roi_dict.values()[0]
    
    if os.path.isdir(input):
        if os.path.isfile(output):
            raise IOError("When input is directory, output should also be a directory.")
        if not os.path.exists(output):
            os.makedirs(output)
        imagefiles = glob.glob(os.path.join(input, extension))
        for fname in imagefiles:
            img = np.squeeze(io.imread(fname))
            img = equalize_from_ROI(img, roi_bbox)
            basename = os.path.basename(fname)
            outname = "{}-{}".format(prefix, basename)
            outfile = os.path.join(output, outname)
            TIFF.imsave(outfile, img)
    else:
        img = np.squeeze(io.imread(input))
        img = equalize_from_ROI(img, roi_bbox)
        if os.path.isdir(output):
            raise IOError("When input is a file, output should also be a file.")
        TIFF.imsave(output, img)

if __name__ == "__main__":
    main()
