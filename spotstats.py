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
    return [position, row, col, centroid_r, centroid_c, area, perimeter, 
            major_axis_length, minor_axis_length, eccentricity, equiv_diameter, 
            mean_intensity]


def colony_stats(regions, nrows, ncols):
    npos = nrows * ncols
    posdict = dict(zip(range(npos),[None]*npos))
    for region in regions:
        posdict[region.label] =  region_stats(region, nrows, ncols)

    header = ["label", "row", "col", "centroid_r", "centroid_c", "area", 
              "perimeter", "major_axis_length", "minor_axis_length", 
              "eccentricity", "equiv_diameter", "mean_intensity"]
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
@click.argument("imgfile",
                type = click.Path(exists = True))
@click.argument("maskfile",
                type = click.Path(exists = True))
@click.argument("outfile",
                type = click.File("w"),
                default = "-")
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
def main(imgfile, maskfile, outfile, rows, cols):
    img = io.imread(imgfile)
    labeled_img = sp.sparse.load_npz(maskfile).toarray()
    regions = measure.regionprops(labeled_img, intensity_image = img)
    df = colony_stats(regions, rows, cols)
    df.to_csv(outfile)

    
if __name__ == "__main__":
    main()
