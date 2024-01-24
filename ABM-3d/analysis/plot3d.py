"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/24/2024
"""

import numpy as np
import pyvista as pv
import seaborn as sns
from utils import read_cells

#######################################################################
def plot_cells(cells, R, colors, filename, res=50):
    """
    Plot the given population of cells with the given colors to the given
    PDF file. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Population of cells to be plotted.
    R : float
        Cell radius.
    colors : list
        List of colors for each cell.
    filename : str
        Output filename. 
    res : int
        Resolution for plotting each cylinder and hemisphere.
    """
    pl = pv.Plotter()

    # Plot each spherocylinder ... 
    for i in range(cells.shape[0]):
        # Define the cylinder and hemispherical caps that constitute each 
        # spherocylinder
        cylinder = pv.Cylinder(
            center=cells[i, :3],
            direction=cells[i, 3:6],
            radius=R,
            height=cells[i, 6],
            resolution=res,
            capping=False
        )
        cap1_center = cells[i, :3] - cells[i, 7] * cells[i, 3:6]
        cap2_center = cells[i, :3] + cells[i, 7] * cells[i, 3:6]
        cap1 = pv.Sphere(
            center=cap1_center,
            direction=cells[i, 3:6],
            radius=R,
            start_phi=90,
            end_phi=180,
            theta_resolution=res,
            phi_resolution=res
        )
        cap2 = pv.Sphere(
            center=cap2_center,
            direction=cells[i, 3:6],
            radius=R,
            start_phi=0,
            end_phi=90,
            theta_resolution=res,
            phi_resolution=res
        )

        # Add the composite surface to the plotter instance with the 
        # corresponding color
        pl.add_mesh(cylinder + cap1 + cap2, color=colors[i])

    # Change the view to bird's eye, add axes directions, and save
    pl.view_xy()
    pl.add_axes()
    pl.save_graphic(filename)

#######################################################################
if __name__ == '__main__':
    cells, params = read_cells('test_iter16460000.txt')
    R = params['R']
    colors = [
        sns.color_palette()[0] if cells[i, 5] >= -0.5 else sns.color_palette()[3]
        for i in range(cells.shape[0])
    ]
    plot_cells(cells, R, colors, 'test.pdf', res=50)

