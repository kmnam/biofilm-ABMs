"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    7/17/2024
"""

import sys
import numpy as np
from PIL import Image
import pyvista as pv
#pv.start_xvfb()
import seaborn as sns
from utils import read_cells
from plot3d import plot_cells 

#######################################################################
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    args = sys.argv[3:]
    uniform_color = ('--uniform' in args)
    plot_boundary = ('--bound' in args)
    plot_membrane = ('--membrane' in args)
    plot_frame(
        filename, outfilename, view='xy', res=20, uniform_color=uniform_color,
        plot_boundary=plot_boundary, plot_membrane=plot_membrane
    )

