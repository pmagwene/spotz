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



class Settings(collections.MutableMapping):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]


def load_config(fname):
    """ROI file name, section string -> Settings object.
    """
    with open(fname, "r") as f:
        #config = yaml.safe_load(f)
        roidict = json.load(f)
        return roidict
        print(config)
        return {key: Settings(**value) for (key,value) in config.iteritems()}

def equalize_from_ROI(img, region):
    mask = np.zeros(img.shape)
    minr, minc, maxr, maxc = region
    mask[minr:maxr, minc:maxc] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eqlzd = util.img_as_uint(exposure.equalize_hist(img, mask = mask))
    return eqlzd


def extract_region(img, minr, minc, maxr, maxc):
    return img[minr:maxr, minc:maxc]


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
@click.option("--equalize",
    type = click.Path(exists=True))
def main(roifile, input, output, equalize, extension):
    """Extract regions of interest from every image file in specified input directory,
    writing sub-images to output directory.
    """
    # get ROIs from infile
    regions = load_config(roifile)

    # Create output subdirectories
    region_names = regions.keys()
    for name in region_names:
        os.makedirs(os.path.join(output, name))

    # if equalization invoked
    if equalize:
        eqlz_region = load_config(equalize).values()[0]

    if os.path.isfile(input):
        with open(input, "r") as f:
            basename = os.path.basename(input)
            img = np.squeeze(io.imread(f))
            if equalize:
                img = equalize_from_ROI(img, eqlz_region)
            for (name, bbox) in regions.iteritems():
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
                if equalize:
                    img = equalize_from_ROI(img, eqlz_region)
                for (name, bbox) in regions.iteritems():
                    subimg = extract_region(img, *bbox)
                    outfile = "{}-{}".format(name, basename)
                    outname = os.path.join(output, name, outfile)
                    TIFF.imsave(outname, subimg)

    else:
        pass            



if __name__ == "__main__":
    main()



