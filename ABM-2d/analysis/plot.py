"""
Functions for visualizing spherocylindrical cells with Matplotlib.

Authors:
    Kee-Myoung Nam

Last updated:
    10/19/2023
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

#########################################################################
def get_cell_outlines(cells, R):
    """
    For each cell in the given population, get a set of coordinates that trace
    the outline of the cell. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Population of cells.
    R : float
        Radius of hemispherical cap on each cell. 

    Returns
    -------
    Array of cell outline coordinates (with each row corresponding to a cell).
    """
    np_per_cap = 20    # Trace each hemispherical cap with this many points
    theta = np.array(
        [np.pi * j / np_per_cap for j in range(np_per_cap + 1)],
        dtype=np.float64
    )
    cap = np.stack((R * np.cos(theta), R * np.sin(theta)), axis=0)

    # Get the origins of the two caps for each cell in the population
    lengths = cells[:, 4]
    lengths.shape = (cells.shape[0], 1)
    origins_1 = cells[:, :2] + (lengths / 2) * cells[:, 2:4]
    origins_2 = cells[:, :2] - (lengths / 2) * cells[:, 2:4]

    # If a cell has direction (nx, ny), then the angle the cell makes with 
    # the positive x-axis is arctan(ny / nx)
    angles = np.arctan2(cells[:, 3], cells[:, 2])

    # For each cell ...
    outlines = np.zeros((cells.shape[0], 2 * np_per_cap + 3, 2), dtype=np.float64)
    for i in range(cells.shape[0]):
        # Get the coordinates of each hemispherical cap by rotation and translation 
        a1, b1 = np.cos(angles[i] - np.pi / 2), np.sin(angles[i] - np.pi / 2)
        a2, b2 = np.cos(angles[i] + np.pi / 2), np.sin(angles[i] + np.pi / 2)
        rotation_1 = np.array([[a1, -b1], [b1, a1]], dtype=np.float64)
        rotation_2 = np.array([[a2, -b2], [b2, a2]], dtype=np.float64)
        cap_1 = np.dot(rotation_1, cap)
        cap_2 = np.dot(rotation_2, cap)
        outlines[i, :(np_per_cap + 1), 0] = origins_1[i, 0] + cap_1[0, :]
        outlines[i, :(np_per_cap + 1), 1] = origins_1[i, 1] + cap_1[1, :]
        outlines[i, (np_per_cap + 1):(2*np_per_cap + 2), 0] = origins_2[i, 0] + cap_2[0, :]
        outlines[i, (np_per_cap + 1):(2*np_per_cap + 2), 1] = origins_2[i, 1] + cap_2[1, :]
        outlines[i, 2*np_per_cap + 2, :] = outlines[i, 0, :]    # Close the outline

    return outlines

#########################################################################
def plot_cells(cells, ax, R, colors=None, linewidth=1):
    """
    Display the cells with Matplotlib.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Population of cells to be shown.
    ax : `matplotlib.pyplot.Axes`
        Axes onto which the cells will be plotted.
    R : float
        Radius of hemispherical cap on each cell.
    colors : str or list of str
        Either one color for the entire population or a list of colors for
        each cell in the population. None by default (in which case the 
        cells are all colored with `sns.color_palette()[0]`).
    linewidth : float
        Line width for each cell. 

    Returns
    -------
    Updated axes with the plotted cells.
    """
    ax.clear()
    if colors is None:
        colors = [sns.color_palette()[0] for _ in range(cells.shape[0])]
    elif isinstance(colors, str) or isinstance(colors, tuple):
        colors = [colors for _ in range(cells.shape[0])]

    # Get the cell outlines
    outlines = get_cell_outlines(cells, R)

    # Plot each cell one by one 
    for i in range(outlines.shape[0]):
        x = outlines[i, :, 0]
        y = outlines[i, :, 1]
        ax.plot(x, y, color=colors[i], linewidth=linewidth)

    # Set aspect ratio
    ax.set_aspect('equal')

    return ax

#########################################################################
def plot_simulation(paths, outpath, R=None, fps=10, colors=None, linewidth=1,
                    figsize=(6.4, 4.8)):
    """
    Given an ordered sequence of file paths, parse the stored simulation data
    and generate a video.

    Parameters
    ----------
    paths : list of str
        List of file paths in the order in which they are to be parsed.
    outpath : str
        Output video file path. Must end with .avi.
    R : float
        Cell radius. None by default (in which case the radius is parsed from
        the file; if not provided, an error is raised)
    fps : int
        Frames per second.
    colors : str or list of str
        Either one color for the entire population or a list of colors for
        each cell *group* in the population. None by default (in which case
        the cells are all colored by group according to the seaborn deep 
        color palette, `sns.color_palette()`).
    linewidth : float
        Line width for each cell. 
    figsize : tuple of two floats
        Figure dimensions in inches.
    """
    # Run through the files once, to plot the cells and obtain the best x-
    # and y-axes limits
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    times = []
    for path in paths:
        # Parse the radius from the file, if need be
        if R is None:
            with open(path) as f:
                for line in f:
                    if line.startswith('# R = '):
                        m = re.match(r'# R = ([0-9e+-\.]+)', line)
                        R = float(m.group(1))
                        break
                    elif not line.startswith('#'):
                        break
        # If the file did not specify a radius, then raise a RuntimeError
        if R is None:
            raise RuntimeError('No cell radius specified')

        # Parse the timepoint associated with the file 
        t = 0
        with open(path) as f:
            for line in f:
                if line.startswith('# t_curr = '):
                    m = re.match(r'# t_curr = ([0-9e+-\.]+)', line)
                    t = float(m.group(1))
                    break
        times.append(t)

        # Load and plot the cells that are stored in each file
        cells = np.loadtxt(path, comments='#', delimiter='\t', skiprows=0)
        if len(cells.shape) == 1:
            cells = cells.reshape((1, -1))
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        plot_cells(cells, ax, R)
        ax.set_aspect('equal')

        # Identify their x- and y-axes limits
        curr_xmin, curr_xmax = ax.get_xlim()
        curr_ymin, curr_ymax = ax.get_ylim()
        if curr_xmin < xmin:
            xmin = curr_xmin
        if curr_xmax > xmax:
            xmax = curr_xmax
        if curr_ymin < ymin:
            ymin = curr_ymin
        if curr_ymax > ymax:
            ymax = curr_ymax
        plt.close(fig)

    # Run through the files again, now to plot the cells and save them 
    images = []
    for path, t in zip(paths, times):
        # Load the cells that are stored in each file
        cells = np.loadtxt(path, comments='#', delimiter='\t', skiprows=0)
        if len(cells.shape) == 1:
            cells = cells.reshape((1, -1))
    
        # Were the cells identified by group?
        if cells.shape[1] > 9:
            # If so, assume the identifiers are given in column 9 (first
            # column after essential data) and label the cells by their
            # groups
            #
            # If colors were specified, then use them in order of the group
            # labels, assuming they are labeled 1, 2, 3 ...
            #
            # Otherwise, use the seaborn deep color palette
            if colors is None:
                groups = cells[:, 9].astype(np.int32) - 1
                colors_by_cell = [
                    sns.color_palette()[groups[i]] for i in range(cells.shape[0])
                ]
            elif type(colors) == str:
                colors_by_cell = [colors for _ in range(cells.shape[0])]
            else:    # type(colors) should be list
                groups = cells[:, 9].astype(np.int32) - 1
                colors_by_cell = [colors[groups[i]] for i in range(cells.shape[0])]
        # Otherwise, label the cells with a single color
        else:
            colors_by_cell = None

        # Plot the cells with the specified colors and configure axes
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        plot_cells(cells, ax, R, colors=colors_by_cell, linewidth=linewidth)
        ax.set_aspect('equal')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Label plot with the timepoint associated with this population and 
        # the population size 
        ax.set_title(r'$t = {:.10f}, n = {}$'.format(t, cells.shape[0]))

        # Get the figure contents as a PIL image
        canvas = fig.canvas
        canvas.draw()
        image = Image.frombytes(
            'RGB', canvas.get_width_height(), canvas.tostring_rgb()
        )
        images.append(image)
        plt.close(fig)

    # Stitch the images together and export as an .avi file
    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), isColor=True)
    for image in images:
        video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

