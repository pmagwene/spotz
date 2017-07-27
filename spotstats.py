import os, os.path

import numpy as np
import scipy as sp
import pandas as pd
from skimage import (io, measure)

import click


def region_stats(region, nrows, ncols):
    position = region.label
    row = (position/ncols)
    col = position - (row * ncols)
    centroid_r = region.centroid[0]
    centroid_c = region.centroid[1]
    area = region.area
    perimeter = region.perimeter
    major_axis_length = region.major_axis_length
    minor_axis_length = region.minor_axis_length
    eccentricity = region.eccentricity
    equiv_diameter = region.equivalent_diameter
    mean_intensity = region.mean_intensity
    solidity = region.solidity
    convex_area = region.convex_area
    bbox_minr, bbox_minc, bbox_maxr, bbox_maxc = region.bbox
    return [position, row, col, centroid_r, centroid_c, area, perimeter, 
            major_axis_length, minor_axis_length, eccentricity, equiv_diameter, 
            mean_intensity, solidity, convex_area,
            bbox_minr, bbox_minc, bbox_maxr, bbox_maxc]


def colony_stats(regions, nrows, ncols):
    npos = nrows * ncols
    posdict = dict(zip(range(1, npos+1),[None]*npos))
    for region in regions:
        posdict[region.label] =  region_stats(region, nrows, ncols)

    header = ["label", "row", "col", "centroid_r", "centroid_c", "area", 
              "perimeter", "major_axis_length", "minor_axis_length", 
              "eccentricity", "equiv_diameter", "mean_intensity",
              "solidity", "convex_area",
              "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc"]
    tbl = []
    for i in range(1, npos+1):
        if posdict[i] is None:
            row = [i] +  (["NA"] * (len(header) - 1)) # fill with NA
        else:
            row = posdict[i]     
        tbl.append(row)   
    return pd.DataFrame(tbl, columns=header)


#-------------------------------------------------------------------------------    

@click.command()
@click.argument("imgfiles",
                type = click.Path(exists = True, dir_okay = False),
                nargs = -1)
@click.argument("maskdir",
                type = click.Path(exists = True,
                                  file_okay = False,
                                  dir_okay = True))
@click.argument("outdir",  
                type = click.Path(exists = True,
                                  file_okay = False,
                                  dir_okay = True))
@click.option("-p", "--prefix",
              help = "Prefix for output CSV files",
              type = str,
              default = "STATS",
              show_default = True)
@click.option("-m", "--mask-prefix",
              help = "Prefix for Mask files.",
              type = str,
              default = "MASK",
              show_default = True)
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
def main(imgfiles, maskdir, outdir, prefix, mask_prefix, rows, cols):
    """Extract statistics from labeled objects in an image.
    """
    for imgfile in imgfiles:
        img = np.squeeze(io.imread(imgfile))
        basename = os.path.basename(imgfile)
        root, _ = os.path.splitext(basename)
        outfile = os.path.join(outdir, "{}-{}.csv".format(prefix, root))
        mask_file = os.path.join(maskdir, "{}-{}.npz".format(mask_prefix, root))

        if not os.path.exists(mask_file):
            continue

        labeled_img = sp.sparse.load_npz(mask_file).toarray()
        regions = measure.regionprops(labeled_img, intensity_image = img)
        stats_df = colony_stats(regions, rows, cols)
        stats_df.to_csv(outfile, index = False)

    
if __name__ == "__main__":
    main()
