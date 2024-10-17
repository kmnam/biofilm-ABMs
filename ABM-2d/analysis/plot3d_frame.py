"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    7/30/2024
"""

import sys
import pyvista as pv
from plot3d import plot_frame

#######################################################################
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    args = sys.argv[3:]
    uniform_color = ('--uniform' in args)
    plot_boundary = ('--bound' in args)
    plot_membrane = ('--membrane' in args)
    plot_arrested = ('--arrested' in args)
    plot_3d = ('--3d' in args)
    pl = pv.Plotter(off_screen=True)
    #pl.clear()
    plot_frame(
        filename, outfilename, pl, plot_3d=plot_3d, view='xy', res=20,
        uniform_color=uniform_color, plot_boundary=plot_boundary,
        plot_membrane=plot_membrane, plot_arrested=plot_arrested
    )

