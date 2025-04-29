"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    3/18/2025
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
_colidx_rz = 3
_colseq_r = [1, 2, 3]
_colidx_nx = 4
_colidx_ny = 5
_colidx_nz = 6
_colseq_n = [4, 5, 6]
_colidx_drx = 7
_colidx_dry = 8
_colidx_drz = 9
_colidx_dnx = 10
_colidx_dny = 11
_colidx_dnz = 12
_colidx_l = 13
_colidx_half_l = 14
_colidx_t0 = 15
_colidx_growth = 16
_colidx_eta0 = 17
_colidx_eta1 = 18
_colidx_maxeta1 = 19
_colidx_sigma0 = 20
_colidx_group = 21

#######################################################################
def plot_cells(cells, R, colors, xmin, xmax, ymin, ymax, zmin, zmax, title,
               view='xyz', res=50, image_scale=2, position=None, show=False):
    """
    Plot the given population of cells with the given colors to the given
    output file. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Input population of cells. 
    R : float 
        Cell radius. 
    colors : list or `numpy.ndarray`
        Cell colors. 
    xmin, xmax : float
        Minimum and maximum x-values. 
    ymin, ymax : float
        Minimum and maximum y-values. 
    zmin, zmax : float
        Minimum and maximum z-values. 
    title : str
        Plot title. 
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', or 'xyz'.
    res : int
        Spherocylinder resolution. 
    image_scale : int
        Image scale. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter. 
    show : bool
        If True, show the plot instead of saving it to file.

    Returns
    -------
    A screenshot of the plot, plus the camera position. 
    """
    with multiprocessing.Pool(1) as pool:
        # Plot each spherocylinder ...
        ws = np.array([1024, 768], dtype=np.int64)
        pl = pv.Plotter(window_size=ws*image_scale, off_screen=(not show))
        for i in range(cells.shape[0]):
            # Define the cylinder and hemispherical caps that constitute each 
            # spherocylinder
            cylinder = pv.Cylinder(
                center=cells[i, _colseq_r],
                direction=cells[i, _colseq_n],
                radius=R,
                height=cells[i, _colidx_l],
                resolution=res,
                capping=False
            )
            cap1_center = cells[i, _colseq_r] - cells[i, _colidx_half_l] * cells[i, _colseq_n]
            cap2_center = cells[i, _colseq_r] + cells[i, _colidx_half_l] * cells[i, _colseq_n]
            cap1 = pv.Sphere(
                center=cap1_center,
                direction=cells[i, _colseq_n],
                radius=R,
                start_phi=90,
                end_phi=180,
                theta_resolution=res,
                phi_resolution=res
            )
            cap2 = pv.Sphere(
                center=cap2_center,
                direction=cells[i, _colseq_n],
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
        if view == 'xy':
            pl.view_xy()
        elif view == 'xy_bottom':
            pl.view_xy()
            pl.camera.position = (
                pl.camera.position[0], pl.camera.position[1], -pl.camera.position[2]
            )
            pl.camera.focal_point = (
                pl.camera.focal_point[0], pl.camera.focal_point[1], -pl.camera.focal_point[2]
            )
        elif view == 'xz':
            pl.view_xz()
        elif view == 'yz':
            pl.view_yz()
        elif view == 'xyz':
            pl.view_isometric()
            pl.camera.elevation -= 15
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

    print(position)
    return screenshot, position

#######################################################################
def plot_frame(filename, outfilename=None, xmin=None, xmax=None, ymin=None,
               ymax=None, zmin=None, zmax=None, view='xyz', res=50, time=None,
               image_scale=2, uniform_color=False, xcross=False, ycross=False,
               group=None, position=None, show=False):
    """
    Plot the cells in the given file.

    Parameters
    ----------
    TODO Write
    """
    # Parse the cells
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    if group is not None:
        cells = cells[cells[:, _colidx_group] == group, :]

    # If plotting a cross-section, find the center of mass and shift the 
    # cells accordingly 
    if xcross or ycross:
        center = np.mean(cells[:, _colseq_r], axis=0)
        if xcross:
            yshift = center[1] - np.floor(cells[:, _colidx_ry].min() - 10 * L0)
            for i in range(cells.shape[0]):
                if cells[i, _colidx_rx] > center[0]:
                    cells[i, _colidx_ry] -= yshift
        else:
            xshift = center[0] - np.floor(cells[:, _colidx_rx].min() - 10 * L0)
            for i in range(cells.shape[0]):
                if cells[i, _colidx_ry] > center[1]:
                    cells[i, _colidx_rx] -= xshift

    # Infer axes limits
    if xmin is None:
        xmin = np.floor(cells[:, _colidx_rx].min() - 4 * L0)
    if xmax is None:
        xmax = np.ceil(cells[:, _colidx_rx].max() + 4 * L0)
    if ymin is None:
        ymin = np.floor(cells[:, _colidx_ry].min() - 4 * L0)
    if ymax is None:
        ymax = np.ceil(cells[:, _colidx_ry].max() + 4 * L0)
    if zmin is None:
        zmin = np.floor(cells[:, _colidx_rz].min() - 4 * L0)
    if zmax is None:
        zmax = np.ceil(cells[:, _colidx_rz].max() + 4 * L0)
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
            colors.append(palette[group_idx, :])

    # Plot the cells 
    print('Plotting {} (t = {}, {} cells) ...'.format(filename, time, cells.shape[0]))
    title = 't = {:.10f}, n = {}'.format(time, cells.shape[0])
    screenshot, position = plot_cells(
        cells, R, colors, xmin, xmax, ymin, ymax, zmin, zmax, title,
        view=view, res=res, image_scale=image_scale, position=position,
        show=show
    )

    # Get a screenshot of the plotted cells, if desired
    if outfilename is not None and screenshot is not None:
        image = Image.fromarray(screenshot)
        print('... saving to {}'.format(outfilename))
        image.save(outfilename)
    print("(" + ','.join([str(x) for i in range(3) for x in position[i]]) + ")")

#######################################################################
def plot_simplicial_complex(points, tree, groups, xmin, xmax, ymin, ymax, zmin,
                            zmax, title, view='xyz', res=50, image_scale=2,
                            position=None, show=False):
    """
    TODO Write
    """
    color1 = sns.color_palette('coolwarm', as_cmap=True)(0.1)
    color2 = sns.color_palette('pastel')[0]

    with multiprocessing.Pool(1) as pool:
        # Plot each simplex ... 
        ws = np.array([1024, 768], dtype=np.int64)
        pl = pv.Plotter(window_size=ws*image_scale, off_screen=(not show))
        for simplex, _ in tree.get_simplices():
            # Only plot each simplex if it is full-dimensional or if it has 
            # no cofaces
            if len(tree.get_cofaces(simplex, 1)) == 0:
                if len(simplex) == 2:
                    v1, v2 = simplex
                    pl.add_mesh(
                        pv.Line(pointa=points[v1, :], pointb=points[v2, :]),
                        color='black', line_width=5
                    )
                elif len(simplex) == 3:
                    v1, v2, v3 = simplex
                    t = pv.Triangle([points[v1, :], points[v2, :], points[v3, :]])
                    pl.add_mesh(t, color=color2)
                elif len(simplex) == 4:
                    v1, v2, v3, v4 = simplex
                    t1 = pv.Triangle([points[v1, :], points[v2, :], points[v3, :]])
                    t2 = pv.Triangle([points[v1, :], points[v2, :], points[v4, :]])
                    t3 = pv.Triangle([points[v1, :], points[v3, :], points[v4, :]])
                    t4 = pv.Triangle([points[v2, :], points[v3, :], points[v4, :]])
                    pl.add_mesh(t1, color=color1)
                    pl.add_mesh(t2, color=color1)
                    pl.add_mesh(t3, color=color1)
                    pl.add_mesh(t4, color=color1)

        # Plot the points as spheres 
        for i in range(points.shape[0]):
            if groups[i] == 1:
                pl.add_mesh(
                    pv.Sphere(radius=0.2, center=points[i, :]),
                    color='black'
                )
        
        # Reconfigure axes, add title and axes directions, and save
        pl.add_title(title, font='arial', font_size=12*image_scale)
        if view == 'xy':
            pl.view_xy()
        elif view == 'xz':
            pl.view_xz()
        elif view == 'yz':
            pl.view_yz()
        elif view == 'xyz':
            pl.view_isometric()
            pl.camera.elevation -= 15
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

    print(position)
    return screenshot, position

#######################################################################


