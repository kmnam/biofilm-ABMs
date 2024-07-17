"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    7/17/2024
"""

import os
import numpy as np
from PIL import Image
import cv2
import pyvista as pv
#pv.start_xvfb()
import seaborn as sns
from utils import read_cells, parse_dir

#######################################################################
def plot_cells(cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
               title, view='xy', res=50):
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
    # Define arrays for cell centers and orientations with z-coordinate
    positions = np.hstack((cells[:, :2], rz * np.ones((cells.shape[0], 1))))
    orientations = np.hstack((cells[:, 2:4], np.zeros((cells.shape[0], 1))))

    # Plot each spherocylinder ... 
    for i in range(cells.shape[0]):
        # Define the cylinder and hemispherical caps that constitute each 
        # spherocylinder
        cylinder = pv.Cylinder(
            center=positions[i, :],
            direction=orientations[i, :],
            radius=R,
            height=cells[i, 4],
            resolution=res,
            capping=False
        )
        cap1_center = positions[i, :] - cells[i, 5] * orientations[i, :]
        cap2_center = positions[i, :] + cells[i, 5] * orientations[i, :]
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
def plot_frame(filename, outfilename, xmin=None, xmax=None, ymin=None,
               ymax=None, zmin=None, zmax=None, view='xy', res=50,
               uniform_color=False, plot_boundary=False, plot_membrane=False):
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
    uniform_color : bool
        If True, color all cells with a single color (blue).
    plot_boundary : bool
        If True, look for boundary indicators and plot the boundary cells 
        with a different color. There should be at least 12 columns in the
        input data. 
    plot_membrane : bool
        If True, plot the elastic membrane with the given rest radius at 
        the origin. There should be at least 12 columns in the input data.
    """
    palette = [    # Assume a maximum of five groups 
        sns.color_palette('hls', 8)[5],
        sns.color_palette('hls', 8)[0],
        sns.color_palette('hls', 8)[3],
        sns.color_palette('hls', 8)[1],
        sns.color_palette('hls', 8)[6]
    ]
    pastel = [
        sns.color_palette('pastel')[0],
        sns.color_palette('pastel')[3],
        sns.color_palette('pastel')[2],
        sns.color_palette('pastel')[1],
        sns.color_palette('pastel')[4]
    ]
    
    # Parse the cells and infer axes limits
    cells, params = read_cells(filename)
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    if xmin is None:
        xmin = np.floor(cells[:, 0].min() - 4 * L0)
    if xmax is None:
        xmax = np.ceil(cells[:, 0].max() + 4 * L0)
    if ymin is None:
        ymin = np.floor(cells[:, 1].min() - 4 * L0)
    if ymax is None:
        ymax = np.ceil(cells[:, 1].max() + 4 * L0)
    if zmin is None:
        zmin = rz - R
    if zmax is None:
        zmax = rz + R

    # Determine cell colors 
    if uniform_color:
        colors = [palette[0] for i in range(cells.shape[0])]
    else:
        ngroups = int(max(cells[:, 10]))
        palette_ = palette[:ngroups]
        pastel_ = pastel[:ngroups]
        if not plot_boundary:
            colors = [
                palette_[int(cells[i, 10]) - 1] for i in range(cells.shape[0])
            ]
        else:
            colors = []
            for i in range(cells.shape[0]):
                if cells[i, 11] != 0:
                    colors.append(pastel_[int(cells[i, 10]) - 1])
                else:
                    colors.append(palette_[int(cells[i, 10]) - 1])

    # Plot the cells 
    title = 't = {:.10f}, n = {}'.format(params['t_curr'], cells.shape[0])
    print('Plotting {} ({} cells) ...'.format(filename, cells.shape[0]))
    pl = pv.Plotter(off_screen=True)
    pl = plot_cells(
        cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
        title, view=view, res=res
    )

    # Plot the circle (if desired)
    if plot_membrane:
        max_area = cells.shape[0] * np.pi * R * R + 2 * R * cells[:, 4].sum()
        radius = params['confine_rest_radius_factor'] * np.sqrt(max_area / np.pi)
        pl.add_mesh(
            pv.Circle(radius), color='white', show_edges=True, style='wireframe',
            line_width=3
        )

    # Get a screenshot of the plotted cells
    image = Image.fromarray(pl.screenshot())
    print('... saving to {}'.format(outfilename))
    image.save(outfilename)
    pl.close()

#######################################################################
def plot_simulation(filenames, outfilename, xmin, xmax, ymin, ymax, zmin, zmax,
                    res=50, fps=10, uniform_color=False, plot_boundary=False,
                    plot_membrane=False, overwrite_frames=False):
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
    uniform_color : bool
        If True, color all cells with a single color (blue).
    plot_boundary : bool
        If True, look for boundary indicators and plot the boundary cells 
        with a different color. There should be at least 12 columns in the
        input data. 
    plot_membrane : bool
        If True, plot the elastic membrane with the given rest radius at 
        the origin. There should be at least 12 columns in the input data.
    overwrite_frames : bool
        If False, then any existing .jpg file with the same name as that 
        for a given frame is kept as is; if not, each frame is overwritten. 
    """
    image_filenames = []
    for filename in filenames:
        # Determine the filename for the frame 
        image_filename = '{}_frame.jpg'.format(filename[:-4])
        image_filenames.append(image_filename)
        
        # If the image file does not exist or is to be overwritten ... 
        if overwrite_frames or not os.path.exists(image_filename):
            plot_frame(
                filename, image_filename, xmin=xmin, xmax=xmax, ymin=ymin,
                ymax=ymax, zmin=zmin, zmax=zmax, view='xy', res=res,
                uniform_color=uniform_color, plot_boundary=plot_boundary,
                plot_membrane=plot_membrane
            )
            image_filenames.append(image_filename)

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

