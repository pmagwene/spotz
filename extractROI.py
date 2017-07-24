from __future__ import print_function
import collections
import glob, os, os.path
import json
import warnings

import numpy as np
from skimage import (transform, io, exposure, util)

import click
import yaml

import tifffile as TIFF

#--------------------------------------------------------------------------------

def extract_region(img, minr, minc, maxr, maxc):
    return img[minr:maxr, minc:maxc]


#--------------------------------------------------------------------------------

@click.command()
@click.argument("roifile", 
                type = click.Path(exists=True))
@click.argument("imgfiles",  
                type = click.Path(exists=True, dir_okay = False),
                nargs = -1)
@click.argument("outdir",  
                type = click.Path(exists = True, file_okay = False,
                                  dir_okay = True))

def main(roifile, imgfiles, outdir):
    """Extract regions of interest (ROI) from one or more image files.

    ROIs are defined in a JSON formatted input file. Sub-images are
    written to the user provided sub-directory.

    """
    # get ROIs from infile
    with open(roifile, "r") as f:
        roidict = json.load(f)
        
    # Create output subdirectories
    region_names = roidict.keys()
    for name in region_names:
        os.makedirs(os.path.join(outdir, name))

    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))
        root = os.path.basename(fname)
        for (name, bbox) in roidict.iteritems():
            subimg = extract_region(img, *bbox)
            outfile = os.path.join(outdir, name, "{}-{}".format(name, root))
            TIFF.imsave(outfile, subimg)
        

if __name__ == "__main__":
    main()



