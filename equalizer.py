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
@click.argument("imgfiles",  
                type = click.Path(exists=True,
                                  dir_okay = False),
                nargs = -1)
@click.argument("outdir",  
                type = click.Path(exists = True,
                                  file_okay = False,
                                  dir_okay = True))
@click.option("-p", "--prefix",
              help = "Prefix to prepend to equalized file names.",
              default = "EQLZD")
def main(roifile, imgfiles, outdir, prefix):
    """Use ROI to equalize and image. .
    """

    roi_dict = load_ROI_dict(roifile)
    roi_bbox = roi_dict.values()[0]

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))
        img = equalize_from_ROI(img, roi_bbox)

        root = os.path.basename(imgfile)
        outfile = os.path.join(outdir, "{}-{}".format(prefix, root))
        TIFF.imsave(outfile, img)
    

if __name__ == "__main__":
    main()
