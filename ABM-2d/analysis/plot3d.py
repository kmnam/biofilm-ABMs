"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/15/2024
"""

import os
from time import sleep
import numpy as np
from PIL import Image
import cv2
import pyvista as pv
#pv.start_xvfb()
import seaborn as sns
from utils import read_cells, parse_dir

#######################################################################
def plot_cells(cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
               title, plot_3d=False, view='xy', res=50):
    """
    Plot the given population of cells with the given colors to the given
    PDF file. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Population of cells to be plotted.
    pl : `pyvista.Plotter`
        Plotter instance onto which to plot the cells. 
    R : float
        Cell radius.
    rz : float
        z-coordinate of each cell center. 
    colors : list
        List of colors for each cell.
    xmin, xmax : float, float
        x-axis bounds. 
    ymin, ymax : float, float
        y-axis bounds.
    zmin, zmax : float, float
        z-axis bounds.
    title : str
        Plot title.
    view : str, 'xy' or 'xz' or 'yz'
        Set the view to the given pair of axes. 
    res : int
        Resolution for plotting each cylinder and hemisphere.
    """
    if plot_3d:
        _colidx_id = 0
        _colidx_rx = 1
        _colidx_ry = 2
        _colidx_rz = 3
        _colseq_r = [1, 2, 3]
        _colidx_nx = 4
        _colidx_ny = 5
        _colidx_nz = 6
        _colseq_n = [4, 5, 6]
        _colidx_l = 7          # TODO For now
        _colidx_half_l = 8
        _colidx_group = 9
    else:
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

    # Define arrays for cell centers and orientations with z-coordinate
    if plot_3d:   # If plotting 3-D population, plot only the bottom layer 
        cells = cells[cells[:, _colidx_rz] < rz + R, :]
        positions = cells[:, _colseq_r]
        orientations = cells[:, _colseq_n]
    else:
        positions = np.hstack((cells[:, _colseq_r], rz * np.ones((cells.shape[0], 1))))
        orientations = np.hstack((cells[:, _colseq_n], np.zeros((cells.shape[0], 1))))

    # Plot each spherocylinder ... 
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
    pl.add_title(title, font='arial', font_size=12)
    if view == 'xy':
        pl.view_xy()
        pl.show_bounds(
            bounds=[xmin, xmax, ymin, ymax, zmin, zmax], show_zlabels=False,
            n_xlabels=2, n_ylabels=2, xtitle='', ytitle='', ztitle='',
            font_family='arial', font_size=12
        )
    elif view == 'xz':
        pl.view_xz()
        pl.show_bounds(
            bounds=[xmin, xmax, ymin, ymax, zmin, zmax], show_ylabels=False,
            n_xlabels=2, n_zlabels=2, xtitle='', ytitle='', ztitle='',
            font_family='arial', font_size=12
        )
    elif view == 'yz':
        pl.view_yz()
        pl.show_bounds(
            bounds=[xmin, xmax, ymin, ymax, zmin, zmax], show_xlabels=False,
            n_ylabels=2, n_zlabels=2, xtitle='', ytitle='', ztitle='',
            font_family='arial', font_size=12
        )
    pl.add_axes()
    pl.reset_camera(bounds=[xmin, xmax, ymin, ymax, zmin, zmax])

    return pl

#######################################################################
def plot_frame(filename, outfilename, pl, xmin=None, xmax=None, ymin=None,
               ymax=None, zmin=None, zmax=None, plot_3d=False, view='xy', res=50,
               time=None, uniform_color=False, plot_boundary=False,
               plot_membrane=False, plot_arrested=False):
    """
    Given an ordered list of files containing cells to be plotted, parse 
    and plot each population of cells and generate a video.

    There should be at least 11 columns in the input data. 

    Parameters
    ----------
    filename : str
        Path to file containing the cells to be plotted.
    outfilename : str
        Output filename.
    pl : `pyvista.Plotter`
        Existing Plotter instance. 
    xmin, xmax : float, float
        x-axis bounds. 
    ymin, ymax : float, float
        y-axis bounds.
    zmin, zmax : float, float
        z-axis bounds.
    view : str, 'xy' or 'xz' or 'yz'
        Set the view to the given pair of axes. 
    res : int
        Resolution for plotting each cylinder and hemisphere.
    time : float
        If not None, label the plot with the given time. 
    uniform_color : bool
        If True, color all cells with a single color (blue).
    plot_boundary : bool
        If True, look for boundary indicators and plot the boundary cells 
        with a different color. There should be at least 12 columns in the
        input data. 
    plot_membrane : bool
        If True, plot the elastic membrane with the given rest radius at 
        the origin.
    plot_arrested : bool
        If True, look for non-growing cells and plot them with a different 
        color.
    """
    if plot_3d:
        _colidx_id = 0
        _colidx_rx = 1
        _colidx_ry = 2
        _colidx_rz = 3
        _colseq_r = [1, 2, 3]
        _colidx_nx = 4
        _colidx_ny = 5
        _colidx_nz = 6
        _colseq_n = [4, 5, 6]
        _colidx_l = 7          # TODO For now
        _colidx_half_l = 8
        _colidx_group = 9
    else:
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

    palette = [    # Assume a maximum of two groups
        sns.color_palette('hls', 8)[5],
        sns.color_palette('hls', 8)[0]
    ]
    pastel = [
        sns.color_palette('pastel')[0],
        sns.color_palette('pastel')[3]
    ]
    deep = [
        sns.color_palette('deep')[4], 
        sns.color_palette('deep')[1]
    ]

    # Clear plotter 
    #pl.clear()
    
    # Parse the cells and infer axes limits
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    if xmin is None:
        xmin = np.floor(cells[:, _colidx_rx].min() - 4 * L0)
    if xmax is None:
        xmax = np.ceil(cells[:, _colidx_rx].max() + 4 * L0)
    if ymin is None:
        ymin = np.floor(cells[:, _colidx_ry].min() - 4 * L0)
    if ymax is None:
        ymax = np.ceil(cells[:, _colidx_ry].max() + 4 * L0)
    if zmin is None:
        zmin = rz - R        # Plot only the bottom layer if in 3-D
    if zmax is None:
        zmax = rz + 2 * R    # Plot only the bottom layer if in 3-D
    if time is None:
        time = params['t_curr']

    # Determine cell colors 
    if uniform_color:
        colors = [palette[0] for i in range(cells.shape[0])]
    else:
        ngroups = int(max(cells[:, _colidx_group]))
        palette_ = palette[:ngroups]
        pastel_ = pastel[:ngroups]
        deep_ = deep[:ngroups]
        colors = []
        for i in range(cells.shape[0]):
            group_idx = int(cells[i, _colidx_group]) - 1
            if plot_boundary and cells[i, _colidx_bound] != 0:
                colors.append(deep_[group_idx])
            elif plot_arrested and cells[i, _colidx_growth] == 0:
                colors.append(pastel_[group_idx])
            else:
                colors.append(palette_[group_idx])

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
    pl = plot_cells(
        cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax, title,
        plot_3d=plot_3d, view=view, res=res
    )

    # Get a screenshot of the plotted cells
    image = Image.fromarray(pl.screenshot())
    print('... saving to {}'.format(outfilename))
    image.save(outfilename)

    return pl

#######################################################################
def plot_simulation(filenames, outfilename, xmin, xmax, ymin, ymax, zmin, zmax,
                    plot_3d=False, res=50, fps=20, times=None, uniform_color=False,
                    plot_boundary=False, plot_membrane=False, plot_arrested=False,
                    overwrite_frames=False):
    """
    Given an ordered list of files containing cells to be plotted, parse 
    and plot each population of cells and generate a video. 

    Parameters
    ----------
    filenames : list of str
        Ordered list of paths to files containing the cells to be plotted.
    outfilename : str
        Output filename. 
    R : float
        Cell radius.
    rz : float
        z-coordinate of each cell center. 
    xmin, xmax : float, float
        x-axis bounds. 
    ymin, ymax : float, float
        y-axis bounds.
    zmin, zmax : float, float
        z-axis bounds.
    res : int
        Resolution for plotting each cylinder and hemisphere.
    fps : int
        Frames per second.
    times : list
        If not None, label each frame with the corresponding time in the 
        given list. 
    uniform_color : bool
        If True, color all cells with a single color (blue).
    plot_boundary : bool
        If True, look for boundary indicators and plot the boundary cells 
        with a different color. There should be at least 12 columns in the
        input data. 
    plot_membrane : bool
        If True, plot the elastic membrane with the given rest radius at 
        the origin. There should be at least 12 columns in the input data.
    plot_arrested : bool
        If True, look for non-growing cells and plot them with a different 
        color.
    overwrite_frames : bool
        If False, then any existing .jpg file with the same name as that 
        for a given frame is kept as is; if not, each frame is overwritten. 
    """
    # Set up the plotter
    pl = pv.Plotter(off_screen=True)

    image_filenames = []
    for i, filename in enumerate(filenames):
        # Determine the filename for the frame 
        image_filename = '{}_frame.jpg'.format(filename[:-4])
        image_filenames.append(image_filename)

        # Determine the timepoint for the frame 
        if times is None:
            time = None
        else:
            time = times[i]
        
        # If the image file does not exist or is to be overwritten ... 
        if overwrite_frames or not os.path.exists(image_filename):
            pl = plot_frame(
                filename, image_filename, pl, xmin=xmin, xmax=xmax, ymin=ymin,
                ymax=ymax, zmin=zmin, zmax=zmax, plot_3d=plot_3d, view='xy',
                res=res, time=time, uniform_color=uniform_color,
                plot_boundary=plot_boundary, plot_membrane=plot_membrane,
                plot_arrested=plot_arrested
            )
            image_filenames.append(image_filename)
            #sleep(1.0)

    # Stitch the images together and export as an .avi file
    width = None
    height = None
    with Image.open(image_filenames[0]) as image:
        width, height = image.size
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(
        outfilename, fourcc, fps, (width, height), isColor=True
    )
    for image_filename in image_filenames:
        with Image.open(image_filename) as image:
            video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

