"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    7/10/2024
"""

import sys
import numpy as np
from PIL import Image
import pyvista as pv
pv.start_xvfb()
import seaborn as sns
from utils import read_cells
from plot3d import plot_cells 

#######################################################################
def plot_frame(filename, outfilename, view='xy', res=50, uniform_color=False):
    """
    Given an ordered list of files containing cells to be plotted, parse 
    and plot each population of cells and generate a video. 

    Parameters
    ----------
    filename : str
        Path to file containing the cells to be plotted.
    outfilename : str
        Output filename. 
    view : str, 'xy' or 'xz' or 'yz'
        Set the view to the given pair of axes. 
    res : int
        Resolution for plotting each cylinder and hemisphere.
    uniform_color : bool
        If True, color all cells with a single color (blue). 
    """
    palette = [    # Assume a maximum of five groups 
        sns.color_palette('hls', 8)[5],
        sns.color_palette('hls', 8)[0],
        sns.color_palette('hls', 8)[3],
        sns.color_palette('hls', 8)[1],
        sns.color_palette('hls', 8)[6]
    ]
    
    # Parse the cells and infer axes limits
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    xmin = np.floor(cells[:, 0].min() - 4 * L0)
    xmax = np.ceil(cells[:, 0].max() + 4 * L0)
    ymin = np.floor(cells[:, 1].min() - 4 * L0)
    ymax = np.ceil(cells[:, 1].max() + 4 * L0)
    zmin = rz - R
    zmax = rz + R

    # Determine cell colors 
    if uniform_color:
        colors = [palette[0] for i in range(cells.shape[0])]
    else:
        ngroups = int(max(cells[:, 10]))
        palette_ = palette[:ngroups]
        colors = [
            palette_[int(cells[i, 10]) - 1] for i in range(cells.shape[0])
        ]

    # Plot the cells 
    title = 't = {:.10f}, n = {}'.format(params['t_curr'], cells.shape[0])
    print('Plotting {} ({} cells) ...'.format(filename, cells.shape[0]))
    pl = pv.Plotter(off_screen=True)
    pl = plot_cells(
        cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
        title, view=view, res=res
    )

    # Get a screenshot of the plotted cells
    image = Image.fromarray(pl.screenshot())
    print('... saving to {}'.format(outfilename))
    image.save(outfilename)
    pl.close()

#######################################################################
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    uniform_color = (len(sys.argv) == 4 and sys.argv[3] == '--uniform-color')
    plot_frame(
        filename, outfilename, view='xy', res=20, uniform_color=uniform_color
    )

