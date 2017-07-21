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
@click.argument("imgfile",  
    type = click.Path(exists=True,
                      dir_okay = False))
@click.argument("outfile",  
    type = click.Path(exists=False,
                      dir_okay = False))
def main(roifile, imgfile, outfile):
    """Use ROI to equalize and image. .
    """
    roi_dict = load_ROI_dict(roifile)
    roi_bbox = roi_dict.values()[0]

    with open(imgfile, "r") as f:
        img = np.squeeze(io.imread(f))

    img = equalize_from_ROI(img, roi_bbox)
    TIFF.imsave(outfile, img)

if __name__ == "__main__":
    main()
