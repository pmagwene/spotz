#!/usr/bin/env python

import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io, color)


import click

from toolz import *



read_image = io.imread
invert = util.invert
adjust_log = curry(exposure.adjust_log)
adjust_sigmoid = curry(exposure.adjust_sigmoid)
rescale_intensity = curry(exposure.rescale_intensity)
equalize_adaptive = curry(exposure.equalize_adapthist)
equalize_hist = curry(exposure.equalize_hist)
clear_border = curry(segmentation.clear_border)
disk = disk_selem = curry(morphology.disk)
binary_opening = curry(morphology.binary_opening)
binary_closing = curry(morphology.binary_closing)
binary_erosion = curry(morphology.binary_erosion)
binary_dilation = curry(morphology.binary_dilation)
opening = curry(morphology.opening)
closing = curry(morphology.closing)
erosion = curry(morphology.erosion)
dilation = curry(morphology.dilation)
thin = curry(morphology.thin)
watershed = curry(segmentation.watershed)


def threshold_mean(img):
    return img > filters.threshold_mean(img)

def threshold_triangle(img):
    return img > filters.threshold_triangle(img)

def threshold_otsu(img):
    return img > filters.threshold_otsu(img)

def threshold_li(img):
    return img > filters.threshold_li(img)

def threshold_yen(img):
    return img > filters.threshold_yen(img)

def threshold_isodata(img):
    return img > filters.threshold_isodata(img)

def threshold_niblack(img):
    return img > filters.threshold_niblack(img)

def threshold_sauvola(img):
    return img > filters.threshold_sauvola(img)

@curry
def threshold_gaussian(block_size, sigma, img):
    return img > filters.threshold_local(img, block_size,
                                         method = "gaussian",
                                         param = sigma)

@curry
def remove_small_objects(min_size, img, **args):
    return morphology.remove_small_objects(img, min_size, **args)

@curry
def remove_small_holes(min_size, img, **args):
    return morphology.remove_small_holes(img, min_size, **args)

@curry
def disk_opening(radius, img):
    return morphology.binary_opening(img, footprint= morphology.disk(radius))

@curry
def disk_closing(radius, img):
    return morphology.binary_closing(img, footprint= morphology.disk(radius))

@curry
def disk_erosion(radius, img):
    return morphology.binary_erosion(img, footprint= morphology.disk(radius))

def multiotsu_lower(img):
    t = filters.threshold_multiotsu(img)
    return img > t[0]

def multiotsu_upper(img):
    t = filters.threshold_multiotsu(img)
    return img > t[1]


threshold_dict = {
    "mean": threshold_mean,
    "triangle": threshold_triangle,
    "otsu": threshold_otsu,
    "li": threshold_li,
    "yen": threshold_yen,
    "isodata": threshold_isodata,
    "niblack": threshold_niblack,
    "sauvola": threshold_sauvola,
    "gaussian": threshold_gaussian,
    "multiotsu_lower": multiotsu_lower,
    "multiotsu_upper": multiotsu_upper
}


def binarize_image(img, clip_limit=0.01, threshold_func="multiotsu_lower", minsize=100, disk_size=3):

    bw = pipe(img,
            equalize_adaptive(clip_limit=clip_limit),
            threshold_dict[threshold_func],
            invert,
            disk_closing(disk_size),
            ndimage.binary_fill_holes,
            disk_opening(disk_size)
            )

    lbl_bw = pipe(bw,
                morphology.label,
                remove_small_objects(minsize),
                segmentation.clear_border)
    
    final_bw = lbl_bw > 0
    return final_bw, morphology.label(final_bw)


def watershed_binarized(lbl_bw, mindist):
    obj_distance = ndimage.distance_transform_edt(lbl_bw)
    peaks = feature.peak_local_max(obj_distance, min_distance=mindist, labels=lbl_bw)
    ws_mask = np.zeros(obj_distance.shape, dtype=bool)
    ws_mask[tuple(peaks.T)] = True
    ws_markers, _ = ndimage.label(ws_mask)
    ws_labels = segmentation.watershed(-obj_distance, ws_markers, mask=lbl_bw)
    return ws_labels

@click.command()
@click.argument("infile", type=click.Path(exists=True))
@click.option("--outprefix", type=str, help="Output file prefix")
@click.option("--outdir", type=click.Path(), default="output_dir", help="Output directory") 
@click.option("--minsize", type=int, default=500, help="Minimum size of objects to keep")
@click.option("--mindist", type=int, default=25, help="Minimum distance between peaks for watershed")
@click.option("--disk_size", type=int, default=3, help="Size of disk for morphological operations")
@click.option("--clip_limit", type=float, default=0.01, help="Clip limit for adaptive histogram equalization")
@click.option("--threshold_func", 
              type=click.Choice(threshold_dict.keys()),
                default="multiotsu_lower", 
              help="Thresholding function to use.")
@click.option("--save_eq", is_flag=True, help="Save equalized image", default=False)
@click.option("--save_bin", is_flag=True, help="Save binarized image", default=False)
@click.option("--save_lbl", is_flag=True, help="Save labeled image", default=False)
@click.option("--save_ws", is_flag=True, help="Save watershed image", default=True)
def main(infile, outprefix, outdir, threshold_func, clip_limit, minsize, mindist, disk_size, save_eq, save_bin, save_lbl, save_ws):
    img = io.imread(infile)
    bw, lbl_bw = binarize_image(img, 
                                clip_limit=clip_limit, 
                                threshold_func=threshold_func, 
                                minsize=minsize, 
                                disk_size=disk_size)
    ws_lbl = watershed_binarized(lbl_bw, mindist)

    if not Path(outdir).exists():
        Path(outdir).mkdir()

    basename = Path(infile).stem
    if outprefix:
        basename = f"{outprefix}_{basename}"

    if save_eq:
        io.imsave(f"{outdir}/{basename}_equalized.png", util.img_as_ubyte(adjust_log(img)))
    if save_bin:
        io.imsave(f"{outdir}/{basename}_binarized.png", util.img_as_ubyte(bw))
    if save_lbl:
        io.imsave(f"{outdir}/{basename}_labeled.png", util.img_as_ubyte(color.label2rgb(lbl_bw, image=img)))
    if save_ws:
        io.imsave(f"{outdir}/{basename}_watershed.png", util.img_as_ubyte(color.label2rgb(ws_lbl, image=img)))




if __name__ == "__main__":
    main()