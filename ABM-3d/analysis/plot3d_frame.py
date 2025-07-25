"""
Authors:
    Kee-Myoung Nam

Last updated:
    3/18/2025
"""

import gc
import sys
import re
from time import sleep
import pyvista as pv
from utils import read_cells
from plot3d import plot_frame

#######################################################################
if __name__ == '__main__':
    # Parse input arguments 
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    zmin = None
    zmax = None
    position = None
    view = 'xyz'
    cyan = False
    cross = False
    args = sys.argv[3:]
    
    # Check whether the first six arguments specifies the axes limits 
    if len(args) >= 6 and all(arg.isnumeric() for arg in args[:6]):
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[3:9]]
        # In this case, check whether there is a seventh argument, and whether
        # it specifies the camera position
        if len(args) >= 7:
            regex = r'\('
            for i in range(8):
                regex += r'([\d\.]+),'
            regex += r'([\d\.]+)\)'
            m = re.match(regex, args[6])
            if m is not None:
                values = [float(m[i + 1]) for i in range(9)]
                position = [values[:3], values[3:6], values[6:]]
    # Check whether there are additional arguments 
    if '--basal' in args:
        view = 'xy_bottom'
    elif '--top' in args:
        view = 'xy'
    elif '--cross-xyz' in args:
        view = 'xyz'
        cross = True
    elif '--cross-xz' in args:
        view = 'xz'
        cross = True
    if '--cyan' in args:
        cyan = True

    """
    if len(sys.argv) == 4 and sys.argv[3] == '--basal':
        view = 'xy_bottom'
    elif len(sys.argv) == 9:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[3:9]]
    elif len(sys.argv) == 10:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[3:9]]
        if sys.argv[9] == '--basal':
            view = 'xy_bottom'
        else:
            values = [float(x) for x in sys.argv[9].strip('()').split(',')]
            position = [values[:3], values[3:6], values[6:]]
    elif len(sys.argv) == 11:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[3:9]]
        values = [float(x) for x in sys.argv[9].strip('()').split(',')]
        position = [values[:3], values[3:6], values[6:]]
        if sys.argv[10] == '--basal':
            view = 'xy_bottom'
    """

    time = None
    if xmin is not None:
        _, params = read_cells(filename)
        time = params['t_curr']
        xmin -= params['L0']
        ymin -= params['L0']
        zmin -= params['L0']
        xmax += params['L0']
        ymax += params['L0']
        zmax += params['L0']
    plot_frame(
        filename, outfilename, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        zmin=zmin, zmax=zmax, view=view, res=20, time=time, image_scale=5,
        uniform_color=False, cyan=cyan, cross=cross, position=position,
        show=False
    )
    gc.collect()
    sleep(1)

