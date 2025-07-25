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
from matplotlib import colormaps
from PIL import Image
import cv2
import pyvista as pv
#pv.start_xvfb()
import seaborn as sns
from utils import read_cells, parse_dir, read_simplicial_complex

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
        Cell radius (including the EPS).
    colors : list or `numpy.ndarray`
        List of colors, one for each cell. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. 
    title : str
        Plot title. 
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    res : int
        Spherocylinder resolution. 
    image_scale : int
        Image scale. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter. 
    show : bool
        If True, show the plot instead of returning the screenshot. 

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

    print(position)    # Print for use in video script 
    return screenshot, position

#######################################################################
def plot_cells_with_gel(cells, Rc, colors, gel, Rg, gel_color, xmin, xmax,
                        ymin, ymax, zmin, zmax, title, view='xyz', res=50,
                        image_scale=2, position=None, show=False):
    """
    Plot the given population of cells and gel particles with the given
    colors to the given output file.  

    Parameters
    ----------
    cells : `numpy.ndarray`
        Input population of cells. 
    Rc : float
        Cell radius (including the EPS).
    colors : list or `numpy.ndarray`
        List of colors, one for each cell.
    gel : `numpy.ndarray`
        Input collection of gel particles. 
    Rg : float
        Gel particle radius. 
    gel_color : tuple
        Gel particle color. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. 
    title : str
        Plot title. 
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    res : int
        Spherocylinder resolution. 
    image_scale : int
        Image scale. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter. 
    show : bool
        If True, show the plot instead of returning the screenshot. 

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
                radius=Rc,
                height=cells[i, _colidx_l],
                resolution=res,
                capping=False
            )
            cap1_center = cells[i, _colseq_r] - cells[i, _colidx_half_l] * cells[i, _colseq_n]
            cap2_center = cells[i, _colseq_r] + cells[i, _colidx_half_l] * cells[i, _colseq_n]
            cap1 = pv.Sphere(
                center=cap1_center,
                direction=cells[i, _colseq_n],
                radius=Rc,
                start_phi=90,
                end_phi=180,
                theta_resolution=res,
                phi_resolution=res
            )
            cap2 = pv.Sphere(
                center=cap2_center,
                direction=cells[i, _colseq_n],
                radius=Rc,
                start_phi=0,
                end_phi=90,
                theta_resolution=res,
                phi_resolution=res
            )

            # Add the composite surface to the plotter instance with the 
            # corresponding color
            pl.add_mesh(cylinder + cap1 + cap2, color=colors[i])

        # Plot each gel particle ... 
        for i in range(gel.shape[0]):
            sphere = pv.Sphere(
                center=gel[i, :],
                radius=Rg,
                theta_resolution=res,
                phi_resolution=res
            )
            pl.add_mesh(sphere, color=gel_color)

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

    print(position)    # Print for use in video script 
    return screenshot, position

#######################################################################
def plot_frame(filename, outfilename=None, xmin=None, xmax=None, ymin=None,
               ymax=None, zmin=None, zmax=None, view='xyz', res=50, time=None,
               image_scale=2, uniform_color=False, cross=False, group=None,
               cyan=False, position=None, show=False):
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
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    res : int
        Spherocylinder resolution.
    time : float
        Timepoint. Parsed from input if not given.  
    image_scale : int
        Image scale.
    uniform_color : bool
        If True, plot every cell with the same color; otherwise, plot every
        cell according to its group value.
    cross : bool
        If True, split open the biofilm along the vertical plane perpendicular
        to the view. 
    ycross : bool
        If True, split open the biofilm along the x-z plane. 
    group : int
        If given, plot only the cells with this group value.
    cyan : bool
        If True, plot the group 1 cells in cyan instead of blue. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter.
    show : bool
        If True, show the plot instead of saving the screenshot to file. 
    """
    # Parse the cells
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    if group is not None:
        cells = cells[cells[:, _colidx_group] == group, :]

    # If plotting a cross-section, remove the cells in front of the
    # corresponding plane 
    if cross:
        if view == 'xyz':
            # Here, the plane is y = -x, so we remove all cells for which 
            # y > -x
            idx = (cells[:, _colidx_ry] <= -cells[:, _colidx_rx])
        elif view == 'xz':
            # Here, the plane is y = 0, so we remove all cells for which 
            # y < 0
            idx = (cells[:, _colidx_ry] >= 0)
        elif view == 'yz':
            # Here, the plane is x = 0, so we remove all cells for which 
            # x < 0
            idx = (cells[:, _colidx_rx] >= 0)
        else:
            # All other views are incompatible with cross = True 
            raise RuntimeError(
                'Choice of view ({}) is incompatible with cross = True'.format(view)
            )
        cells = cells[idx, :]

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
        # Determine colors from the coolwarm (blue to red) colormap 
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        idx = np.linspace(0.1, 0.9, ngroups)
        palette = np.array([cmap(i) for i in idx])[:, :3]
        if cyan:    # Change group 1 color to cyan if desired 
            palette[0] = colormaps['managua'](0.9)[:3]

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
def plot_frame_with_gel(cells_filename, gel_filename, outfilename=None, Rg=0.8,
                        max_ngel=10000, xmin=None, xmax=None, ymin=None, ymax=None,
                        zmin=None, zmax=None, view='xyz', res=50, time=None,
                        image_scale=2, uniform_color=False, cross=True, group=None,
                        cyan=False, position=None, show=False, rng=None):
    """
    Plot the cells in the given file.

    Parameters
    ----------
    cells_filename : str
        Path to input file containing cell data.
    gel_filename : str
        Path to input file containing gel coordinates. 
    outfilename : str
        Output filename.
    Rg : float
        Gel particle radius.
    max_ngel : int
        Maximum number of gel particles. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. Inferred from cell/gel coordinates if not given.  
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    res : int
        Spherocylinder resolution.
    time : float
        Timepoint. Parsed from input if not given.  
    image_scale : int
        Image scale.
    uniform_color : bool
        If True, plot every cell with the same color; otherwise, plot every
        cell according to its group value.
    cross : bool
        If True, split open the biofilm along the vertical plane perpendicular
        to the view. 
    ycross : bool
        If True, split open the biofilm along the x-z plane. 
    group : int
        If given, plot only the cells with this group value.
    cyan : bool
        If True, plot the group 1 cells in cyan instead of blue. 
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter.
    show : bool
        If True, show the plot instead of saving the screenshot to file.
    rng : `numpy.random.Generator`
        Random number generator for downsampling gel particles. 
    """
    # Parse the cells
    cells, params = read_cells(cells_filename)
    Rc = params['R']
    L0 = params['L0']
    if group is not None:
        cells = cells[cells[:, _colidx_group] == group, :]

    # Parse the gel particles 
    gel = np.loadtxt(gel_filename, delimiter=',')

    # Downsample the gel particles 
    if gel.shape[0] > max_ngel:
        if rng is None:
            rng = np.random.default_rng(1234567890)
        gel = gel[rng.choice(np.arange(gel.shape[0]), max_ngel, replace=False), :] 

    # If plotting a cross-section, remove the cells in front of the
    # corresponding plane 
    if cross:
        if view == 'xyz':
            # Here, the plane is y = -x, so we remove all cells for which 
            # y > -x
            cells_idx = (cells[:, _colidx_ry] <= -cells[:, _colidx_rx])
            gel_idx = (gel[:, 1] <= -gel[:, 0])
        elif view == 'xz':
            # Here, the plane is y = 0, so we remove all cells for which 
            # y < 0
            cells_idx = (cells[:, _colidx_ry] >= 0)
            gel_idx = (gel[:, 1] >= 0)
        elif view == 'yz':
            # Here, the plane is x = 0, so we remove all cells for which 
            # x < 0
            cells_idx = (cells[:, _colidx_rx] >= 0)
            gel_idx = (gel[:, 0] >= 0)
        else:
            # All other views are incompatible with cross = True 
            raise RuntimeError(
                'Choice of view ({}) is incompatible with cross = True'.format(view)
            )
        cells = cells[cells_idx, :]
        gel = gel[gel_idx, :]

    # Infer axes limits
    if xmin is None:
        xmin1 = np.floor(cells[:, _colidx_rx].min() - 4 * L0)
        xmin2 = np.floor(gel[:, 0].min())
        xmin = min(xmin1, xmin2)
    if xmax is None:
        xmax1 = np.ceil(cells[:, _colidx_rx].max() + 4 * L0)
        xmax2 = np.ceil(gel[:, 0].max())
        xmax = max(xmax1, xmax2)
    if ymin is None:
        ymin1 = np.floor(cells[:, _colidx_ry].min() - 4 * L0)
        ymin2 = np.floor(gel[:, 1].min())
        ymin = min(xmin1, xmin2)
    if ymax is None:
        ymax1 = np.ceil(cells[:, _colidx_ry].max() + 4 * L0)
        ymax2 = np.ceil(gel[:, 1].max())
        ymax = max(ymax1, ymax2)
    if zmin is None:
        zmin1 = np.floor(cells[:, _colidx_rz].min() - 4 * L0)
        zmin2 = np.floor(gel[:, 2].min())
        zmin = min(zmin1, zmin2)
    if zmax is None:
        zmax1 = np.ceil(cells[:, _colidx_rz].max() + 4 * L0)
        zmax2 = np.ceil(gel[:, 2].max())
        zmax = max(zmax1, zmax2)
    if time is None:
        time = params['t_curr']

    # Use a spectral colormap
    ngroups = int(params['n_groups'])
    if uniform_color:
        palette = np.array([sns.color_palette('hls')[5]]) * np.ones((ngroups, 4))
    else:
        # Determine colors from the coolwarm (blue to red) colormap 
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        idx = np.linspace(0.1, 0.9, ngroups)
        palette = np.array([cmap(i) for i in idx])[:, :3]
        if cyan:    # Change group 1 color to cyan if desired 
            palette[0] = colormaps['managua'](0.9)[:3]

    # Plot the gel particles in gray
    gel_color = (0.75, 0.75, 0.75)

    # Determine cell colors
    if uniform_color:
        colors = [palette[0, :] for i in range(cells.shape[0])]
    else:
        colors = []
        for i in range(cells.shape[0]):
            group_idx = int(cells[i, _colidx_group]) - 1
            colors.append(palette[group_idx, :])

    # Plot the cells 
    print(
        'Plotting {} and {} (t = {}, {} cells) ...'.format(
            cells_filename, gel_filename, time, cells.shape[0]
        )
    )
    title = 't = {:.10f}, n = {}'.format(time, cells.shape[0])
    screenshot, position = plot_cells_with_gel(
        cells, Rc, colors, gel, Rg, gel_color, xmin, xmax, ymin, ymax, zmin,
        zmax, title, view=view, res=res, image_scale=image_scale,
        position=position, show=show
    )

    # Get a screenshot of the plotted cells, if desired
    if outfilename is not None and screenshot is not None:
        image = Image.fromarray(screenshot)
        print('... saving to {}'.format(outfilename))
        image.save(outfilename)
    print("(" + ','.join([str(x) for i in range(3) for x in position[i]]) + ")")

#######################################################################
def plot_simplicial_complex(points, tree, groups, xmin, xmax, ymin, ymax, zmin,
                            zmax, title, maxdim=3, view='xyz', image_scale=2,
                            position=None, show=False):
    """
    Plot the given simplicial complex.

    Parameters
    ----------
    points : `numpy.ndarray`
        Input array of points.
    tree : `gudhi.SimplexTree`
        Input simplicial complex, stored as a simplex tree. 
    groups : list of int
        List of groups corresponding to each point (cell). 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits.
    title : str
        Plot title. 
    maxdim : int
        Maximum simplex dimension to be plotted. 
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    image_scale : int
        Image scale.
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter.
    show : bool
        If True, show the plot instead of returning the screenshot.
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
            if len(simplex) == maxdim + 1 or len(tree.get_cofaces(simplex, 1)) == 0:
                if len(simplex) == 2 and maxdim >= 1:
                    v1, v2 = simplex
                    pl.add_mesh(
                        pv.Line(pointa=points[v1, :], pointb=points[v2, :]),
                        color='black', line_width=5
                    )
                elif len(simplex) == 3 and maxdim >= 2:
                    v1, v2, v3 = simplex
                    t = pv.Triangle([points[v1, :], points[v2, :], points[v3, :]])
                    pl.add_mesh(t, color=color2)
                elif len(simplex) == 4 and maxdim >= 3:
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

    print(position)    # Print for use in video script 
    return screenshot, position

#######################################################################
def save_simplicial_complex_to_file(filename, cells_filename, outfilename=None,
                                    xmin=None, xmax=None, ymin=None, ymax=None,
                                    zmin=None, zmax=None, maxdim=3, view='xyz',
                                    time=None, image_scale=2, position=None,
                                    show=False):
    """
    Plot the cells in the given file.

    Parameters
    ----------
    filename : str
        Path to input file specifying the simplicial complex.
    cells_filename : str
        Path to input file specifying the cell coordinates. 
    outfilename : str
        Output filename. 
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Axes limits. Inferred from cell coordinates if not given.
    maxdim : int
        Maximum simplex dimension to be plotted. 
    view : str
        Camera view; should be either 'xy', 'xz', 'yz', 'xy_bottom', or 'xyz'.
    time : float
        Timepoint. Parsed from input if not given.  
    image_scale : int
        Image scale.
    position : list of tuples
        The camera position, focal point, and up value; if None, the value 
        is set automatically by the plotter.
    show : bool
        If True, show the plot instead of saving the screenshot to file. 
    """
    # Parse the simplicial complex and the cells
    tree = read_simplicial_complex(filename)
    cells, params = read_cells(cells_filename)
    points = cells[:, _colseq_r]
    groups = cells[:, _colidx_group]
    R = params['R']
    L0 = params['L0']
    
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

    # Plot the complex 
    print('Plotting {} (t = {}, {} cells) ...'.format(filename, time, cells.shape[0]))
    title = 't = {:.10f}, n = {}'.format(time, cells.shape[0])
    screenshot, position = plot_simplicial_complex(
        points, tree, groups, xmin, xmax, ymin, ymax, zmin, zmax, title, 
        maxdim=maxdim, view=view, image_scale=image_scale, position=position,
        show=show
    )

    # Get a screenshot of the plotted cells, if desired
    if outfilename is not None and screenshot is not None:
        image = Image.fromarray(screenshot)
        print('... saving to {}'.format(outfilename))
        image.save(outfilename)
    print("(" + ','.join([str(x) for i in range(3) for x in position[i]]) + ")")

