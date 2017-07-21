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
@click.argument("input",  
                type = click.Path(exists=True,
                                  file_okay = True,
                                  dir_okay = True))
@click.argument("output",  
                type = click.Path(exists=True,
                                  file_okay = False,
                                  dir_okay = True))
@click.option("-e", "--extension",
              help = "File extension type.",
              default = "*.tif")

def main(roifile, input, output, extension):
    """Extract regions of interest from every image file in specified input directory,
    writing sub-images to output directory.
    """
    # get ROIs from infile
    with open(roifile, "r") as f:
        roidict = json.load(f)
        
    # Create output subdirectories
    region_names = roidict.keys()
    for name in region_names:
        os.makedirs(os.path.join(output, name))


    if os.path.isfile(input):
        with open(input, "r") as f:
            basename = os.path.basename(input)
            img = np.squeeze(io.imread(f))
            for (name, bbox) in roidict.iteritems():
                subimg = extract_region(img, *bbox)
                outfile = "{}-{}".format(name, basename)
                outname = os.path.join(output, name, outfile)
                TIFF.imsave(outname, subimg)
        

    elif os.path.isdir(input):
        infiles = glob.glob(os.path.join(input,extension))

        # iterate over input files and subregions
        # putting out sub-image for each file
        for fname in infiles:
            basename = os.path.basename(fname)
            with open(fname, "r") as f:
                img = np.squeeze(io.imread(f))
                for (name, bbox) in roidict.iteritems():
                    subimg = extract_region(img, *bbox)
                    outfile = "{}-{}".format(name, basename)
                    outname = os.path.join(output, name, outfile)
                    TIFF.imsave(outname, subimg)

    else:
        pass            



if __name__ == "__main__":
    main()



