"""
Module for analyzing cell trajectories in agent-based simulations.

The plotting functions in this module plots the cells (and the biofilm 
boundary) in the r-z plane. 

Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2025
"""
import os
import glob
import re
import multiprocessing
import numpy as np
import seaborn as sns
from PIL import Image
import pyvista as pv
from utils import parse_dir, read_cells

########################################################################
_colidx_id = 0
_colseq_r = [1, 2, 3]
_colseq_n = [4, 5, 6]
_colidx_l = 13
_colidx_group = 21

########################################################################
def parse_lineage(filename):
    """
    Parse the given lineage file.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    Dictionary that maps each cell ID to the ID of its mother cell. 
    """
    parents = {}
    with open(filename) as f:
        for line in f:
            daughter, mother = [int(x) for x in line.strip().split('\t')]
            parents[daughter] = mother

    return parents

########################################################################
def parse_boundary(filename):
    """
    Parse the given boundary file.

    This boundary is assumed to be obtained from an azimuthal projection 
    of the cells into the r-z plane. 

    Parameters
    ----------
    filename : str
        Input filename. 

    Returns
    -------
    Array of boundary points (each row corresponds to a point).
    """
    # Parse the boundary (assuming that the vertices have been specified 
    # in order) ... 
    points = []
    with open(filename) as f:
        for line in f:
            if line.startswith('BOUNDARY_VERTEX'):
                point = np.array([float(x) for x in line.strip().split('\t')[2:]])
                points.append(point)

    return np.array(points)

########################################################################
def parse_boundaries(indir):
    """
    Parse all boundary files within the given directory, corresponding to
    different frames of a simulation. 

    Parameters
    ----------
    indir : str
        Input directory. 

    Returns
    -------
    List of arrays of boundary points, one for each simulation frame.  
    """
    filenames = glob.glob(os.path.join(indir, '*.txt'))
    filenames_sorted = []
    
    # Find the initial file 
    for filename in filenames:
        if filename.endswith('init_boundary.txt'):
            filenames_sorted.append(filename)
            break

    # Run through all intermediate files and sort them in order of iteration
    filenames_iter = []
    idx = []
    for filename in filenames:
        m = re.search(r'iter(\d+)_boundary\.txt$', filename)
        if m is not None:
            filenames_iter.append(filename)
            idx.append(int(m.group(1)))
    sorted_idx = np.argsort(idx)
    filenames_sorted += [filenames_iter[i] for i in sorted_idx]
    
    # Find the final file (if one exists)
    for filename in filenames:
        if filename.endswith('final_boundary.txt'):
            filenames_sorted.append(filename)
            break

    # Run through the files ...
    boundaries = []
    for filename in filenames_sorted:
        boundaries.append(parse_boundary(filename))

    return boundaries

########################################################################
def find_cell_time_window(cell_id, cells_all, times, tmin=None, tmax=None,
                          max_frames=200):
    """
    Find the time window pertaining the given cell and its ancestors.  

    This function identifies the maximal contiguous subset of simulation
    frames for the given cell and its ancestors that begins after `tmin`
    and ends before `tmax`.

    Parameters
    ----------
    cell_id : int
        Cell ID. 
    cells_all : list of `numpy.ndarray`
        List of arrays specifying the cells within each simulation frame.
    times : list of float
        Corresponding list of timepoints. 
    tmin : float
        Earliest timepoint at which to start following the cell and its
        ancestors. If None, this earliest timepoint is set to 0.
    tmax : float 
        Latest timepoint at which to start following the cell. If None, 
        this is set to the latest timepoint at which the cell exists (i.e.,
        before it divides).
    max_frames : int
        Maximum number of frames to include within the time window.

    Returns
    -------
    List of timepoints within the window, and their corresponding indices
    within the input array of timepoints (`times`). 
    """
    # Find the earliest time after tmin such that an ancestor of the cell 
    # exists 
    #
    # This should be 0 or ~tmin, depending on whether the latter is defined
    if tmin is None:
        idx_start = 0
    else:
        try:
            idx_start = next(i for i, t in enumerate(times) if t >= tmin)
        except StopIteration:
            raise RuntimeError(
                'Specified minimum start time is too late: {}'.format(tmin)
            )
    t_start = times[idx_start]

    # Find the latest time before tmax such that an ancestor of the cell
    # exists
    #
    # This should be the last timepoint before the cell divides, the very
    # last frame of the simulation, or ~tmax, depending on whether tmax is 
    # defined
    #
    # First, find the final frame in which the cell exists
    found_cell = (cells_all[0][:, _colidx_id] == cell_id).any()
    idx_end = None
    for i in range(1, len(cells_all)):
        found_cell_next = (cells_all[i][:, _colidx_id] == cell_id).any()
        if found_cell and not found_cell_next:
            idx_end = i - 1
            break
        found_cell = found_cell_next
    if idx_end is None:   # In this case, the cell exists in the very final frame
        idx_end = len(cells_all) - 1
    t_end = times[idx_end]

    # If tmax has been defined, check if t_end is greater than tmax 
    if tmax is not None and t_end > tmax:
        # In this case, find the nearest timepoint before tmax
        for i in range(len(times) - 1):
            if times[i] <= tmax and times[i + 1] > tmax:
                idx_end = i
                t_end = times[i]
                break
   
    # Check that t_end is greater than t_start (this may not be the case if 
    # tmax is smaller than tmin)
    if t_end <= t_start:
        raise RuntimeError('Time window for cell trajectory is undefined')

    # Now subsample from the list of timepoints between t_start and t_end,
    # to get the desired number of frames 
    #
    # Generate a mesh of timepoints with the desired size and identify, for
    # each such timepoint, the timepoint in the window that is closest
    window = times[idx_start:idx_end]
    window_idx = list(range(idx_start, idx_end))
    if len(window) <= max_frames:
        return window, window_idx
    else:
        t_mesh = np.linspace(t_start, t_end, max_frames)
        subwindow = []      # Timepoints for all frames to be plotted
        subwindow_idx = []
        for t in t_mesh:
            nearest_idx = np.argmin(np.abs(np.array(window) - t))
            subwindow.append(window[nearest_idx])
            subwindow_idx.append(idx_start + nearest_idx)
        return subwindow, subwindow_idx

########################################################################
def track_cell(cell_id, cells_all, times, iters, lineage, window, window_idx):
    """
    Track the cell and its ancestors over the given time window. 

    The window is assumed to be chosen in a valid manner, such that every 
    frame therein contains either the cell or one of its ancestors. 

    Parameters
    ----------
    cell_id : int
        Cell ID. 
    cells_all : list of `numpy.ndarray`
        List of arrays specifying the cells within each simulation frame.
    times : list of float
        Corresponding list of timepoints. 
    iters : list of int
        Corresponding list of iteration numbers.
    lineage : dict
        Dictionary containing lineage information.  
    window : list or `numpy.ndarray`
        Input time window, in units of hours. 
    window_idx : list or `numpy.ndarray`
        Indices of timepoints within the window in `times`.

    Returns
    -------
    Array in which each row corresponds to the cell or one of its ancestors
    in each simulation frame within the time window. 
    """
    # Get the full lineage of the cell
    cell_lineage = []
    parent = lineage[cell_id]
    while parent != -1:
        cell_lineage.append(parent)
        parent = lineage[parent]
    cell_lineage = cell_lineage[::-1]
    cell_lineage.append(cell_id)

    # Look for a cell within the lineage within the first frame (note that 
    # one must exist, by definition)
    idx_start = window_idx[0]
    parent_idx = next(
        i for i, x in enumerate(cell_lineage)
        if x in cells_all[idx_start][:, _colidx_id]
    )
    parent = cell_lineage[parent_idx]

    # For each frame within the given time window ...
    trajectory = []
    for i in window_idx:
        # Look for the row corresponding to the current parent
        parent_idx_curr_frame = np.where(cells_all[i][:, _colidx_id] == parent)[0]
        if len(parent_idx_curr_frame) == 0:
            parent_idx += 1
            parent = cell_lineage[parent_idx]
        else:    # len(parent_idx_curr_frame) should be 1
            frame = np.array([iters[i], times[i]])
            frame = np.concatenate((
                frame, cells_all[i][parent_idx_curr_frame, :].reshape(-1)
            ))
            trajectory.append(frame)

    return np.array(trajectory)

########################################################################
def sample_trajectories(indir, rng, n_cells=1, min_rdist=0.9, tmin=None,
                        tmax=None, max_frames=200):
    """
    Sample a subset of single-cell trajectories from the given simulation. 

    Parameters
    ----------
    indir : str
        Input simulation directory.
    rng : `numpy.random.Generator`
        Random number generator. 
    n_cells : int
        Number of trajectories to sample. 
    min_rdist : float
        Sample only cells whose radial distance in the final frame (as a
        fraction of the maximum radial distance) is greater than this value. 
    tmin : float
        Earliest timepoint at which to start following each cell and its
        ancestors. If None, this earliest timepoint is set to 0.
    tmax : float 
        Latest timepoint at which to start following each cell. If None, 
        this is set to the latest timepoint at which each cell exists (i.e.,
        before it divides).
    max_frames : int
        Maximum number of frames to include within the time window for each
        cell.

    Returns
    -------
    List of trajectories. 
    """
    # Parse the simulation files 
    cells_all = []
    times = []
    iters = []
    filenames = parse_dir(os.path.join(indir, '*'))
    for filename in filenames:
        cells, params = read_cells(filename)
        cells_all.append(cells)
        times.append(params['t_curr'])
        if filename.endswith('init.txt'):
            iters.append(0)
        elif filename.endswith('final.txt'):
            iters.append(-1)
        else:
            m = re.search(r'iter(\d+)', filename)
            iters.append(int(m[1]))
    lineage = parse_lineage(glob.glob(os.path.join(indir, '*_lineage.txt'))[0])

    # Randomly sample the given number of cells from the final frame
    origin = np.mean(cells_all[-1][:, _colseq_r], axis=0)
    rdists = np.linalg.norm(cells_all[-1][:, _colseq_r] - origin, axis=1)
    rdists /= np.max(rdists)
    mask = (rdists >= min_rdist)
    cell_ids = rng.choice(cells_all[-1][mask, _colidx_id], n_cells, replace=False).astype(np.int64)

    # Track the chosen cells
    trajectories = []
    for cell_id in cell_ids:
        window, window_idx = find_cell_time_window(
            cell_id, cells_all, times, tmin=tmin, tmax=tmax,
            max_frames=max_frames 
        )
        trajectory = track_cell(
            cell_id, cells_all, times, iters, lineage, window, window_idx
        )
        trajectories.append(trajectory)

    return trajectories

########################################################################
def write_trajectories(trajectories, outprefix, R):
    """
    Write the given trajectory data to file, one file for each trajectory.

    Parameters
    ----------
    trajectories : list of `numpy.ndarray`
        List of input trajectories. 
    outprefix : str
        Output file prefix.
    R : float
        Cell radius (including the EPS).

    Returns
    -------
    List of output filenames, one for each trajectory. 
    """
    outfilenames = []
    for i in range(len(trajectories)):
        outfilename = outprefix + '_trajectory{}.txt'.format(i)
        with open(outfilename, 'w') as f:
            # Write the radius as a header 
            f.write('# R = {}\n'.format(R))

            # For each frame in the trajectory ... 
            for j in range(trajectories[i].shape[0]):
                outstr = ''
                for k in range(trajectories[i].shape[1]):
                    # Write the timepoint index, cell ID, and group as ints
                    if k == 0 or k == _colidx_id + 2 or k == _colidx_group + 2:
                        outstr += '{}\t'.format(int(trajectories[i][j, k]))
                    # Write all other data as floats
                    else:
                        outstr += '{:.10e}\t'.format(trajectories[i][j, k])
                outstr = outstr[:-1] + '\n'
                f.write(outstr)    
        outfilenames.append(outfilename)

    return outfilenames

########################################################################
def parse_trajectory(filename):
    """
    Read the given trajectory file. 

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    Trajectory data (in an array with the same format as that returned by 
    `track_cell()`). 
    """
    # First read the parameters
    params = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                m = re.match(r'# ([A-Za-z0-9_]+) = ([0-9eE+-\.]+)', line)
                try:
                    params[m.group(1)] = float(m.group(2))
                except AttributeError:
                    print(
                        '[WARN] Skipping comment with unexpected format:',
                        line.strip()
                    )

    # Then parse the trajectory data
    trajectory = np.loadtxt(filename, comments='#', delimiter='\t')

    return trajectory, params 

########################################################################
def plot_trajectory_frame(trajectory, idx, boundary, R, plot_prev=True,
                          res=50, image_scale=2, position=None, show=False):
    """
    Plot the given frame of the given trajectory in the r-z plane.
    
    The trajectory is plotted alongside a pre-computed boundary for the
    population.

    Parameters
    ----------
    trajectory : `numpy.ndarray`
        Input trajectory.
    idx : int
        Frame index.
    boundary : `numpy.ndarray`
        Input array of boundary points. The points are assumed to form a 
        simple cycle, with the vertices specified in order. 
    R : float
        Cell radius (including the EPS).
    plot_prev : bool
        If True, plot the preceding points visited by the cell or its 
        ancestors along the trajectory.
    res : int
        Spherocylinder resolution. 
    image_scale : int 
        Image scale. 
    position : tuple
        Camera position. 
    show : bool
        If True, show the plot instead of returning the screenshot. 

    Returns
    -------
    Screenshot of the plot and the camera position. 
    """
    _colidx_id2 = _colidx_id + 2
    _colseq_r2 = np.array(_colseq_r, dtype=np.int64) + 2
    _colseq_n2 = np.array(_colseq_n, dtype=np.int64) + 2
    _colidx_l2 = _colidx_l + 2
    _colidx_group2 = _colidx_group + 2
    if idx < 0:
        idx = trajectory.shape[0] + idx
    time = trajectory[idx, 1]

    # Functions for projecting each point into the r-z plane
    def project_rz(center, orientation=None):
        rnorm = np.linalg.norm(center)
        rdir = center / rnorm
        r = np.linalg.norm(center[:2])
        z = center[2]
        if orientation is None:
            return np.array([r, z])
        else:
            # We need to rotate the orientation vector by theta about the
            # z-axis, where theta is the angle formed by the projection 
            # of the orientation vector onto the x-y plane and the y-axis
            #
            # Note that arccos() returns a value in [0, pi]
            theta = np.arccos(orientation[1] / np.linalg.norm(orientation[:2]))
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            rotated = rot @ orientation.reshape(-1, 1)
            return np.array([r, z]), rotated.reshape(-1)

    # --------------------------------------------------------------- #
    with multiprocessing.Pool(1) as pool:
        ws = np.array([1024, 768], dtype=np.int64)
        pl = pv.Plotter(window_size=ws*image_scale, off_screen=(not show))

        # If desired, plot the previous points along the trajectory
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        if plot_prev:
            points = np.hstack((
                trajectory[:idx+1, _colseq_r2], np.zeros(idx+1).reshape(-1, 1)
            ))
            for i in range(points.shape[0] - 1):
                p1 = project_rz(points[i, :], orientation=None)
                p2 = project_rz(points[i+1, :], orientation=None)
                pl.add_mesh(
                    pv.Line(pointa=[p1[0], 0, p1[1]], pointb=[p2[0], 0, p2[1]]),
                    color='black', line_width=10
                )
            for i in range(points.shape[0] - 1):
                p = project_rz(points[i, :], orientation=None)
                group = trajectory[i, _colidx_group2]
                if group == 1:
                    color = cmap(0.1)
                else:    # group == 2
                    color = cmap(0.9)
                color = (
                    0.5 * color[0] + 0.5, 0.5 * color[1] + 0.5, 0.5 * color[2] + 0.5
                )
                pl.add_mesh(
                    pv.Sphere(radius=0.8 * R, center=[p[0], 0, p[1]]),
                    color=color
                )

        # Plot the spherocylinder ...
        #
        # Define the cylinder and hemispherical caps that constitute each 
        # spherocylinder
        r = trajectory[idx, _colseq_r2]
        n = trajectory[idx, _colseq_n2]
        r_proj, n_proj = project_rz(r, orientation=n)
        r_proj = np.array([r_proj[0], 0, r_proj[1]])
        l = trajectory[idx, _colidx_l2]
        cylinder = pv.Cylinder(
            center=r_proj,
            direction=n_proj,
            radius=R,
            height=l,
            resolution=res,
            capping=False
        )
        cap1_center = r_proj - 0.5 * l * n_proj
        cap2_center = r_proj + 0.5 * l * n_proj
        cap1 = pv.Sphere(
            center=cap1_center,
            direction=n_proj,
            radius=R,
            start_phi=90,
            end_phi=180,
            theta_resolution=res,
            phi_resolution=res
        )
        cap2 = pv.Sphere(
            center=cap2_center,
            direction=n_proj,
            radius=R,
            start_phi=0,
            end_phi=90,
            theta_resolution=res,
            phi_resolution=res
        )

        # Add the composite surface to the plotter instance with the 
        # corresponding color
        group = trajectory[idx, _colidx_group2]
        if group == 1:
            color = cmap(0.1)
        else:    # group == 2
            color = cmap(0.9)
        pl.add_mesh(cylinder + cap1 + cap2, color=color)

        # Plot the boundary as points and lines ...
        boundary = np.hstack((
            boundary[:, 0].reshape(-1, 1),
            np.zeros(boundary.shape[0]).reshape(-1, 1),
            boundary[:, 1].reshape(-1, 1)
        ))
        lines = [[2, i, i + 1] for i in range(boundary.shape[0] - 1)]
        lines.append([2, boundary.shape[0] - 1, 0])
        pl.add_mesh(
            pv.PolyData(boundary, lines=lines), color=(0.7, 0.7, 0.7), line_width=6
        )

        # Reconfigure axes, add title and axes directions, and save
        title = 't = {:.10f} h'.format(time)
        pl.add_title(title, font='arial', font_size=12*image_scale)
        pl.view_xz()
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

########################################################################
def save_trajectory_frame_to_file(trajectory, idx, boundary, R, outfilename,
                                  plot_prev=True, res=50, image_scale=2,
                                  position=None):
    """
    Plot the given frame of the given trajectory and save to an output file.  

    The trajectory is plotted alongside a pre-computed boundary for the
    population. 

    Parameters
    ----------
    trajectory : `numpy.ndarray`
        Input trajectory.
    idx : int
        Frame index.
    boundary : `numpy.ndarray`
        Input array of boundary points. The points are assumed to form a 
        simple cycle, with the vertices specified in order. 
    R : float
        Cell radius (including the EPS).
    outfilename : str
        Output filename. 
    plot_prev : bool
        If True, plot the preceding points visited by the cell or its 
        ancestors along the trajectory.
    res : int
        Spherocylinder resolution. 
    image_scale : int 
        Image scale. 
    position : tuple
        Camera position. 
    """
    # Plot the cells ... 
    screenshot, position = plot_trajectory_frame(
        trajectory, idx, boundary, R, plot_prev=plot_prev, res=res,
        image_scale=image_scale, position=position, show=False
    )

    # ... and save to file 
    image = Image.fromarray(screenshot)
    print('... saving to {}'.format(outfilename))
    image.save(outfilename)

    # Print the camera position (for use in video script)
    print("(" + ','.join([str(x) for i in range(3) for x in position[i]]) + ")")

