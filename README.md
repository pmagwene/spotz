# spotz

A library for quantifying microbial colony growth using time series imaging

## Requirements

* Numpy/Scipy/Matplotlib/Pandas-- the standard Python numerical analysis stack

* scikit-image -- a Python library for image processing

* [peakutils](https://peakutils.readthedocs.io/en/latest/) -- a Python library for detecting peaks in 1D data.

* [Click](http://click.pocoo.org/) - A Python library for building command-line interfaces

* [toolz](https://github.com/pytoolz/toolz) -- A functional "standard library" for Python.

* PyQt5

All of the above are installable via Conda.


## Usage

`spotz --help` will show you the available commands. 

```
Commands:
  drawROI     Define regions of interest (ROIs) on an image, and return...
  equalizer   Use ROI to equalize and image.
  extractROI  Extract regions of interest (ROI) from one or more image...
  gridder     Infer the coordinates of a gridded set of objects in an image.
  segmenter   Segment microbial colonies in an image of a pinned plate.
  showROI     Draw regions of interest defined in a ROI file (json format).
  showmask    Draw labeled objects from segmentation mask over image.
  spotstats   Extract statistics from labeled objects in an image.
```


Each sub-command has it's own help page as well, e.g. `spotz gridder --help` will show you the help page for the `gridder` sub-command.   
