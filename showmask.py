import os, os.path, sys

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)

import click

import spotzplot



#-------------------------------------------------------------------------------    

@click.command()
@click.argument("imgfile",
                type = click.File("r"))
@click.argument("maskfile",
                type = click.File("r"))
@click.option("--color",
              help = "Color map to use to display labeled objects.",
              type = str,
              default = "Reds")
@click.option("--alpha",
              help = "Alpha transparency over labeled object overlay.",
              type = float,
              default = 0.35,
              show_default = True)
def main(imgfile, maskfile, color, alpha):
    """Draw labeled objects from segmentation mask over image.
    """
    img = np.squeeze(io.imread(imgfile))
    labeled_img = sp.sparse.load_npz(maskfile).toarray()
    fig, ax = spotzplot.draw_image_and_labels(img, labeled_img,
                                    mask_cmap = color, alpha = alpha)
    plt.show()



if __name__ == "__main__":
    main()
