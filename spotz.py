#!/usr/bin/env python
import click

import drawROI
import equalizer
import extractROI
import gridder
import segmenter
import spotstats


@click.group()
def cli():
    pass

cli.add_command(drawROI.main, "drawROI")
cli.add_command(equalizer.main, "equalizer")
cli.add_command(extractROI.main, "extractROI")
cli.add_command(gridder.main, "gridder")
cli.add_command(segmenter.main, "segmenter")
cli.add_command(spotstats.main, "spotstats")


if __name__ == "__main__":
    cli()
