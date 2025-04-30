"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    3/22/2025
"""

import os
import multiprocessing
import numpy as np
from PIL import Image
import cv2
import pyvista as pv
#pv.start_xvfb()
import seaborn as sns
from utils import read_cells, parse_dir

#######################################################################
# Column indices for cell data
_colidx_id = 0
_colidx_rx = 1
_colidx_ry = 2
_colseq_r = [1, 2]
_colidx_nx = 3
_colidx_ny = 4
_colseq_n = [3, 4]
_colidx_drx = 5
_colidx_dry = 6
_colidx_dnx = 7
_colidx_dny = 8
_colidx_l = 9
_colidx_half_l = 10
_colidx_t0 = 11
_colidx_growth = 12
_colidx_eta0 = 13
_colidx_eta1 = 14
_colidx_maxeta1 = 15
_colidx_group = 16
_colidx_bound = -1

#######################################################################
def plot_cells(cells, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
               title, res=50, image_scale=2, position=None, show=False):
    """
    Plot the given population of cells with the given colors to the given
    PDF file. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Input population of cells. 
    R : float
        Cell radius (including the EPS).
    rz : float
        Cell z-coordinate.
    colors : list or `numpy.ndarray`
        List of colors, one for each cell. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. 
    title : str
        Plot title. 
    res : int
        Spherocylinder resolution. 
    image_scale : int 
        Image scale. 
    position : tuple
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter. 
    show : bool
        If True, show the plot instead of returning the screenshot.

    Returns
    -------
    A screenshot of the plot, plus the camera position. 
    """
    # Define arrays for cell centers and orientations with z-coordinate
    positions = np.hstack((cells[:, _colseq_r], rz * np.ones((cells.shape[0], 1))))
    orientations = np.hstack((cells[:, _colseq_n], np.zeros((cells.shape[0], 1))))
    
    with multiprocessing.Pool(1) as pool:
        # Plot each spherocylinder ... 
        ws = np.array([1024, 768], dtype=np.int64)
        pl = pv.Plotter(window_size=ws*image_scale, off_screen=(not show))
        for i in range(cells.shape[0]):
            # Define the cylinder and hemispherical caps that constitute each 
            # spherocylinder
            cylinder = pv.Cylinder(
                center=positions[i, :],
                direction=orientations[i, :],
                radius=R,
                height=cells[i, _colidx_l],
                resolution=res,
                capping=False
            )
            cap1_center = positions[i, :] - cells[i, _colidx_half_l] * orientations[i, :]
            cap2_center = positions[i, :] + cells[i, _colidx_half_l] * orientations[i, :]
            cap1 = pv.Sphere(
                center=cap1_center,
                direction=orientations[i, :],
                radius=R,
                start_phi=90,
                end_phi=180,
                theta_resolution=res,
                phi_resolution=res
            )
            cap2 = pv.Sphere(
                center=cap2_center,
                direction=orientations[i, :],
                radius=R,
                start_phi=0,
                end_phi=90,
                theta_resolution=res,
                phi_resolution=res
            )

            # Add the composite surface to the plotter instance with the 
            # corresponding color
            pl.add_mesh(cylinder + cap1 + cap2, color=colors[i])

        # Reconfigure axes, add title and axes directions, and save
        pl.add_title(title, font='arial', font_size=12*image_scale)
        pl.view_xy()
        if position is not None:
            pos, focal_point, up = position
            pl.camera.position = pos
            pl.camera.focal_point = focal_point
            pl.camera.up = up
        else:
            position = pl.camera_position
        pl.add_axes()
        if show:
            screenshot = None
            pl.show()
        else:
            screenshot = pl.screenshot()

    print(position)    # Print for use in video script 
    return screenshot, position

#######################################################################
def plot_frame(filename, outfilename=None, xmin=None, xmax=None, ymin=None,
               ymax=None, res=50, time=None, image_scale=2, uniform_color=False,
               plot_boundary=False, plot_membrane=False, plot_arrested=False,
               group=None, position=None, show=False):
    """
    Plot the cells in the given file.

    Parameters
    ----------
    filename : str
        Input filename.
    outfilename : str
        Output filename. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. Inferred from cell coordinates if not given.  
    res : int
        Spherocylinder resolution.
    time : float
        Timepoint. Parsed from input if not given.  
    image_scale : int
        Image scale.
    uniform_color : bool
        If True, plot every cell with the same color; otherwise, plot every
        cell according to its group value.
    plot_boundary : bool
        If True, plot each cell along the boundary with a pastel color.
    plot_membrane : bool
        If True, plot the surrounding membrane (should it exist) as a circle. 
    plot_arrested : bool
        If True, plot each cell with zero growth rate with a pastel color. 
    group : int
        If given, plot only the cells with this group value. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter.
    show : bool
        If True, show the plot instead of returning the screenshot.
    """
    # Parse the cells
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    if group is not None:
        cells = cells[cells[:, _colidx_group] == group, :]

    # Infer common z-coordinate and axes limits
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    if xmin is None:
        xmin = np.floor(cells[:, _colidx_rx].min() - 4 * L0)
    if xmax is None:
        xmax = np.ceil(cells[:, _colidx_rx].max() + 4 * L0)
    if ymin is None:
        ymin = np.floor(cells[:, _colidx_ry].min() - 4 * L0)
    if ymax is None:
        ymax = np.ceil(cells[:, _colidx_ry].max() + 4 * L0)
    zmin = rz - R
    zmax = rz + R
    if time is None:
        time = params['t_curr']

    # Use a spectral colormap
    ngroups = int(params['n_groups'])
    if uniform_color:
        palette = np.array([sns.color_palette('hls')[5]]) * np.ones((ngroups, 4))
    else:
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        idx = np.linspace(0.1, 0.9, ngroups)
        palette = np.array([cmap(i) for i in idx])[:, :3]
    pastel = 0.5 * palette + 0.5 * np.ones((ngroups, 3))
    deep = 0.5 * palette + 0.5 * np.zeros((ngroups, 3))

    # Determine cell colors
    if uniform_color:
        colors = [palette[0, :] for i in range(cells.shape[0])]
    else:
        colors = []
        for i in range(cells.shape[0]):
            group_idx = int(cells[i, _colidx_group]) - 1
            if plot_boundary and cells[i, _colidx_bound] != 0:
                colors.append(deep[group_idx, :])
            elif plot_arrested and cells[i, _colidx_growth] == 0:
                colors.append(pastel[group_idx, :])
            else:
                colors.append(palette[group_idx, :])

    # Plot the circle (if desired)
    if plot_membrane:
        max_area = cells.shape[0] * np.pi * R * R + 2 * R * cells[:, _colidx_l].sum()
        radius = params['confine_rest_radius_factor'] * np.sqrt(max_area / np.pi)
        pl.add_mesh(
            pv.Disc(
                center=(0, 0, rz + 1.5 * R),
                inner=(radius - 0.5 * R), outer=(radius + 0.5 * R),
                normal=(0, 0, 1),
                r_res=2,
                c_res=int(2 * np.pi * radius * 10)
            ),
            color='gray', show_edges=False, line_width=1
        )

    # Plot the cells 
    print('Plotting {} ({} cells) ...'.format(filename, cells.shape[0]))
    title = 't = {:.10f}, n = {}'.format(time, cells.shape[0])
    screenshot, position = plot_cells(
        cells, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax, title,
        res=res, image_scale=image_scale, position=position, show=show
    )

    # Get a screenshot of the plotted cells
    if outfilename is not None and screenshot is not None:
        image = Image.fromarray(screenshot)
        print('... saving to {}'.format(outfilename))
        image.save(outfilename)
    print("(" + ','.join([str(x) for i in range(3) for x in position[i]]) + ")")

