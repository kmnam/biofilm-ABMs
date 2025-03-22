"""
Plot a single frame using spherocylinders in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    3/22/2025
"""

import gc
import sys
from time import sleep
import pyvista as pv
from utils import read_cells
from plot3d import plot_frame

#######################################################################
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    xmin, xmax, ymin, ymax = [float(x) for x in sys.argv[3:7]]
    if len(sys.argv[3:]) == 4:
        position = None
    else:
        values = [float(x) for x in sys.argv[7].strip('()').split(',')]
        position = [values[:3], values[3:6], values[6:]]
    args = sys.argv[3:]
    uniform_color = ('--uniform' in args)
    plot_boundary = ('--bound' in args)
    plot_membrane = ('--membrane' in args)
    plot_arrested = ('--arrested' in args)
    _, params = read_cells(filename)
    time = params['t_curr']
    xmin -= params['L0']
    ymin -= params['L0']
    xmax += params['L0']
    ymax += params['L0']
    plot_frame(
        filename, outfilename, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        res=20, time=time, image_scale=5, uniform_color=uniform_color,
        plot_boundary=plot_boundary, plot_membrane=plot_membrane,
        plot_arrested=plot_arrested, position=position, show=False
    )
    gc.collect()
    sleep(1)

