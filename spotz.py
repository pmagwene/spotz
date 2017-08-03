#!/usr/bin/env python
import click

import drawROI
import equalizer
import extractROI
import gridder
import segmenter
import spotstats
import showmask


@click.group()
def cli():
    pass

cli.add_command(drawROI.main, "drawROI")
cli.add_command(equalizer.main, "equalizer")
cli.add_command(extractROI.main, "extractROI")
cli.add_command(gridder.main, "gridder")
cli.add_command(segmenter.main, "segmenter")
cli.add_command(spotstats.main, "spotstats")
cli.add_command(drawROI.showROI, "showROI")
cli.add_command(showmask.main, "showmask")


if __name__ == "__main__":
    cli()
