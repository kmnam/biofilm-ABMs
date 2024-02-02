"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    1/31/2024
"""

import sys
import numpy as np
from PIL import Image
import cv2
import pyvista as pv
pv.start_xvfb()
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
def plot_simulation(filenames, outfilename, R, rz, xmin, xmax, ymin,
                    ymax, zmin, zmax, view='xy', res=50, fps=10,
                    uniform_color=False):
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
    view : str, 'xy' or 'xz' or 'yz'
        Set the view to the given pair of axes. 
    res : int
        Resolution for plotting each cylinder and hemisphere.
    fps : int
        Frames per second.
    uniform_color : bool
        If True, color all cells with a single color (blue). 
    """
    images = []
    palette = [    # Assume a maximum of five groups 
        sns.color_palette('muted')[0],
        sns.color_palette('muted')[3],
        sns.color_palette('muted')[2],
        sns.color_palette('muted')[1],
        sns.color_palette('muted')[4]
    ]

    for filename in filenames:
        # Parse and plot the cells in the given file
        cells, params = read_cells(filename)
        if uniform_color:
            colors = [palette[0] for i in range(cells.shape[0])]
        else:
            ngroups = int(max(cells[:, 10]))
            palette_ = palette[:ngroups]
            colors = [
                palette_[int(cells[i, 10]) - 1] for i in range(cells.shape[0])
            ]
        title = 't = {:.10f}, n = {}'.format(params['t_curr'], cells.shape[0])
        print('Plotting {} ({} cells) ...'.format(filename, cells.shape[0]))
        pl = pv.Plotter(off_screen=True)
        pl = plot_cells(
            cells, pl, R, rz, colors, xmin, xmax, ymin, ymax, zmin, zmax,
            title, view=view, res=res
        )

        # Get a screenshot of the plotted cells
        image = pl.screenshot()

        # Convert to PIL image and append
        images.append(Image.fromarray(image))
        pl.close()

    # Stitch the images together and export as an .avi file
    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(
        outfilename, fourcc, fps, (width, height), isColor=True
    )
    for image in images:
        video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

#######################################################################
if __name__ == '__main__':
    filedir = sys.argv[1]
    outprefix = sys.argv[2]
    uniform_color = (len(sys.argv)== 4 and sys.argv[3] == '--uniform-color')
    filenames = parse_dir(filedir)

    # Get cell radius and final dimensions from final file
    cells, params = read_cells(filenames[-1])
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

    # Plot the simulation in 200-frame increments
    for i in range(len(filenames) // 200 + 1):
        start = i * 200
        end = start + 200
        plot_simulation(
            filenames[start:end], outprefix + '_{}.avi'.format(i), R, rz,
            xmin, xmax, ymin, ymax, zmin, zmax, view='xy', res=20, fps=20,
            uniform_color=uniform_color
        )
        print('Saving video: {}_{}.avi'.format(outprefix, i))

