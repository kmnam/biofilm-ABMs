"""
Functions for fitting a hidden Markov model to fluorescence trajectories.  

Authors:
    Kee-Myoung Nam

Last updated:
    4/10/2025
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

######################################################################
#                         UTILITY FUNCTIONS                          #
######################################################################
def find_subtrajectory(traj, tmin, tmax):
    """
    Given a vector-valued trajectory over time, return a subtrajectory
    that falls within the given time window. 

    Parameters
    ----------
    traj : `numpy.ndarray`
        2-D array with >= 2 columns, with each row corresponding to a 
        timepoint in the trajectory.
    tmin : int
        Minimum time, as an integer frame index. 
    tmax : int
        Maximum time, as an integer frame index.

    Returns
    -------
    Subarray of `traj` corresponding to the desired subtrajectory. 
    """
    # Cap tmin and tmax
    if tmin < 0:
        tmin = 0
    if tmax > traj[-1, 0]:
        tmax = traj[-1, 0]

    # Find the frame within the trajectory that are closest to tmin and tmax
    i_start = np.argmin(np.abs(traj[:, 0] - tmin))
    i_end = np.argmin(np.abs(traj[:, 0] - tmax))

    # Check that the starting frame satisfies t < tmin and the ending frame
    # satisfies t > tmax
    #
    # If the starting frame has t > tmin, then change to the immediately 
    # preceding frame
    if traj[i_start, 0] > tmin:
        i_start -= 1

    # Do the same for the ending frame 
    if traj[i_end, 0] < tmax:
        i_end += 1

    # Return the desired sub-trajectory
    return traj[i_start:i_end, :]

######################################################################
def impute_missing_frames(traj):
    """
    Given a vector-valued trajectory over time, return a new trajectory 
    in which all missing frames have been imputed. 

    If two consecutive frames in the trajectory are at timepoint t0 and 
    t0 + t, for t > 1, then the missing frames at time t0 + 1 to t0 + t - 1
    are assumed to have the same trajectory values as at time t0.

    Parameters
    ----------
    traj : `numpy.ndarray`
        2-D array with >= 2 columns, with each row corresponding to a 
        timepoint in the trajectory.
    
    Returns
    -------
    New array with imputed frames. 
    """
    # Maintain a new array for the complete trajectory 
    tmin, tmax = int(traj[0, 0]), int(traj[-1, 0])
    new_traj = np.zeros((tmax - tmin + 1, traj.shape[1]), dtype=np.float64)
    new_traj[:, 0] = np.arange(tmin, tmax + 1)

    # Impute missing frames between any two timepoints that differ by > 1
    new_values = []
    for i in range(traj.shape[0] - 1):
        nframes = int(traj[i + 1, 0] - traj[i, 0])
        new_values += ([traj[i, 1:]] * nframes)

    # Add in the final frame
    new_values.append(traj[-1, 1:])
    new_traj[:, 1:] = new_values

    return new_traj

######################################################################
def find_unique_subtrajectories(data):
    """
    Return a set of indices, (i, j), for which the (i, j)-th value in the
    dataset (i-th trajectory, j-th timepoint) unique across all trajectories.

    The trajectories here are assumed to be scalar-valued, so that `data` is
    a 2-D array.

    Whenever two trajectories (i and k) share the same value at the j-th
    timepoint, the value is stored only for whichever trajectory has the 
    smaller index. 

    Parameters
    ----------
    data : `numpy.ndarray`
        2-D array, with each row corresponding to a trajectory.

    Returns
    -------
    Set of indices, (i, j), for which `data[i, j]` is unique over all 
    trajectories (meaning, `data[i, j] != data[k, j]` whenever `i != k`).
    """
    unique_values = {}
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Identify each trajectory value with its timepoint
            #
            # This means that, if there are multiple trajectories that 
            # achieve the same value at the same time, then they will 
            # not be double-counted
            unique_values[(j, '{:.10f}'.format(data[i, j]))] = (i, j)

    return set(unique_values.values())

######################################################################
def make_data_unique(data):
    """
    Use the indices returned by `find_unique_subtrajectories()` to get a 
    set of reduced subtrajectories.

    The input data array is assumed to be a 3-D array with dimensions
    (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.

    Returns
    -------
    List of 2-D arrays, one for each trajectory, in which non-unique frames
    have been deleted.  
    """
    # Use the riboswitch RFIs to determine unique subtrajectories 
    unique_idx = find_unique_subtrajectories(data[:, :, 3])
    data_unique = []

    # For each trajectory ... 
    for i in range(data.shape[0]):
        # Store only the unique subtrajectory
        idx = [j for j in range(data.shape[1]) if (i, j) in unique_idx]
        data_unique.append(data[i, idx, :])

    return data_unique

######################################################################
def violinplot_with_mean(dists, means, ax, violin_color, mean_color=None,
                         orient='v', zorder=0, plot_empirical_means=False,
                         empirical_mean_color=None, **kwargs):
    """
    """
    data = pd.DataFrame({i: dists[i] for i in range(len(dists))})
    ax = sns.violinplot(
        data=data, ax=ax, color=violin_color, orient=orient, inner=None,
        split=False, zorder=zorder, **kwargs
    )
    patches = ax.get_children()[:len(dists)]
    print(patches)

    if mean_color is None:
        mean_color = sns.color_palette()[1]
    if plot_empirical_means and empirical_mean_color is None:
        empirical_mean_color = sns.color_palette()[2]

    if orient == 'v':
        # For each distribution ...
        for i in range(len(dists)):
            # Locate the patch that was plotted for this distribution 
            patch = patches[i]

            # Identify the vertex in the patch that is closest to the mean 
            vertices = patch.get_paths()[0].vertices
            mean = means[i]
            mean_vertex = vertices[np.argmin(np.abs(vertices[:, 1] - mean)), :]

            # Plot the horizontal line
            if mean_vertex[0] < i:
                xmin, xmax = mean_vertex[0], i + np.abs(i - mean_vertex[0])
            else:
                xmin, xmax = i - np.abs(mean_vertex[0] - i), mean_vertex[0]
            ax.plot(
                [i + 0.98 * (xmin - i), i + 0.98 * (xmax - i)],
                [mean, mean], color=mean_color
            )

            # Plot the same line for the empirical mean, if desired 
            if plot_empirical_means:
                mean = np.mean(dists[i])
                mean_vertex = vertices[np.argmin(np.abs(vertices[:, 1] - mean)), :]
                if mean_vertex[0] < i:
                    xmin, xmax = mean_vertex[0], i + np.abs(i - mean_vertex[0])
                else:
                    xmin, xmax = i - np.abs(mean_vertex[0] - i), mean_vertex[0]
                ax.plot(
                    [i + 0.98 * (xmin - i), i + 0.98 * (xmax - i)],
                    [mean, mean], color=empirical_mean_color
                )
    else:   # orient == 'h'
        # For each distribution ... 
        for i in range(len(dists)):
            # Locate the patch that was plotted for this distribution 
            patch = patches[i]

            # Identify the vertex in the patch that is closest to the mean 
            vertices = patch.get_paths()[0].vertices
            mean = means[i]
            #mean = np.mean(dists[i])
            mean_vertex = vertices[np.argmin(np.abs(vertices[:, 0] - mean)), :]

            # Plot the vertical line
            if mean_vertex[1] < i:
                ymin, ymax = mean_vertex[1], i + np.abs(i - mean_vertex[1])
            else:
                ymin, ymax = i - np.abs(mean_vertex[1] - i), mean_vertex[1]
            ax.plot(
                [mean, mean],
                [i + 0.98 * (ymin - i), i + 0.98 * (ymax - i)],
                color=mean_color
            )

            # Plot the same line for the empirical mean, if desired 
            if plot_empirical_mean:
                mean = np.mean(dists[i])
                mean_vertex = vertices[np.argmin(np.abs(vertices[:, 0] - mean)), :]
                if mean_vertex[1] < i:
                    ymin, ymax = mean_vertex[1], i + np.abs(i - mean_vertex[1])
                else:
                    ymin, ymax = i - np.abs(mean_vertex[1] - i), mean_vertex[1]
                ax.plot(
                    [mean, mean],
                    [i + 0.98 * (ymin - i), i + 0.98 * (ymax - i)],
                    color=empirical_mean_color
                )

    return ax

######################################################################
#                       FITTING AND PREDICTING                       # 
######################################################################
def fit_hmm(data, verbose=False):
    """
    Fit a two-state HMM to the given trajectories.

    The input data array is assumed to be a list of 2-D arrays or a 3-D array
    with dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray` or list
        3-D array or list of 2-D arrays of trajectory data, as specified
        above.
    verbose : bool
        Verbosity flag. 

    Returns
    -------
    Fitted HMM instance, together with the mean state lifetimes and emission
    distribution parameters.
    """
    nsample = len(data)

    # Extract the cdGreen2 fluorescence intensities
    #
    # Note that the data may be a 3-D array or a list of 2-D arrays 
    data_xy = [data[i][:, 2] for i in range(nsample)]
    n_total = sum(data[i].shape[0] for i in range(nsample))
    data_ymin = sum([data[i][:, 4].sum() for i in range(nsample)]) / n_total
    data_ymax = sum([data[i][:, 5].sum() for i in range(nsample)]) / n_total

    # Define normal distributions for fluorescence intensity emissions from 
    # each state
    #
    # Initialize the mean of the normal emission distribution of the low
    # (high) state to be the time-averaged minimum (maximum) cdGreen2 RFI
    d1 = Normal(means=[data_ymin], covs=[0.1], covariance_type='diag')
    d2 = Normal(means=[data_ymax], covs=[0.1], covariance_type='diag')

    # Define the HMM
    model = DenseHMM(verbose=verbose)
    model.add_distributions([d1, d2])
    model.add_edge(model.start, d1, 0.5)
    model.add_edge(model.start, d2, 0.5)
    model.add_edge(d1, d1, 0.999)
    model.add_edge(d1, d2, 0.001)
    model.add_edge(d2, d1, 0.001)
    model.add_edge(d2, d2, 0.999)

    # Fit the model and extract the resulting transition matrix 
    model.fit([traj.reshape(-1, 1) for traj in data_xy])
    transition_matrix = np.exp(model.edges)

    # Normalize each row of the transition matrix (disregarding the end 
    # probability for each state)
    for i in range(2):
        transition_matrix[i, :] /= transition_matrix[i, :].sum()

    # The mean number of frames required to switch from state 1 to 2 is 
    # given by 1 / transition_matrix[0, 1]
    lifetime1 = (5 / transition_matrix[0, 1]) / 60
    if verbose:
        print('Mean lifetime of state 1 (hours):', lifetime1)

    # The mean number of frames required to switch from state 2 to 1 is 
    # given by 1 / transition_matrix[1, 0]
    lifetime2 = (5 / transition_matrix[1, 0]) / 60
    if verbose:
        print('Mean lifetime of state 2 (hours):', lifetime2)

    # Get the mean and standard deviation of each RFI distribution
    mean1 = model.distributions[0].means[0].item()
    mean2 = model.distributions[1].means[0].item()
    std1 = np.sqrt(model.distributions[0].covs[0].item())
    std2 = np.sqrt(model.distributions[1].covs[0].item())
    if verbose:
        print('Emission statistics for state 1: mean = {:.6f}, sd = {:.6f}'.format(mean1, std1))
        print('Emission statistics for state 2: mean = {:.6f}, sd = {:.6f}'.format(mean2, std2))

    return model, np.array([lifetime1, lifetime2, mean1, mean2, std1, std2])

######################################################################
def predict_from_hmm(data, model):
    """
    Infer a maximum-likelihood assignment of hidden states along a collection
    of trajectories, according to the given HMM.

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray` or list
        3-D array or list of 2-D arrays of trajectory data, as specified
        above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance. 

    Returns
    -------
    Array of state assignments, both as a binary vector and as a vector of
    idealized RFI values. 
    """
    # Get state assignments for each trajectory
    #
    # The input data may be a 3-D array or a list of 2-D arrays 
    nsample = len(data)
    assign = [
        model.predict(data[i][:, 2].reshape((1, -1, 1))).numpy().reshape(-1)
        for i in range(nsample)
    ]

    # Get the mean emission values for the two hidden states 
    mean1 = model.distributions[0].means[0].item()
    mean2 = model.distributions[1].means[0].item()

    # Transform the state assignments into idealized RFI values 
    delta = mean2 - mean1
    values = [mean1 + assign_i * delta for assign_i in assign]

    if isinstance(data, np.ndarray):
        return np.array(assign), np.array(values)
    else:
        return assign, values

######################################################################
#       SIMULATING AND COMPUTING/COMPARING LIFETIME STATISTICS       # 
######################################################################
def get_lifetime_dists(assign, include_ends=False):
    """
    Get the distribution of state lifetimes from a collection of
    trajectories. 

    The trajectories here should be of states, not of RFI values. 

    If `include_ends` is True, then the lifetimes of the starting and
    ending states are included. Note that these lifetimes are not
    necessarily reflective of the "true" lifetimes of these two visits,
    as the trajectories may be truncated at one or both ends.

    Parameters
    ----------
    assign : `numpy.ndarray`
        Array of state assignments along each trajectory.
    include_ends : bool
        If True, include the lifetimes of the starting and ending states
        in the statistics. 

    Returns
    -------
    Distributions of state lifetimes in hours. 
    """
    nsample = len(assign)

    lifetimes1 = []
    lifetimes2 = []
    # For each trajectory ... 
    for i in range(nsample):
        # Get the lifetime of each visit to each state
        nframes = assign[i].shape[0]
        lifetimes_traj = []
        state_curr = assign[i][0]
        lifetime_curr = 1
        for j in range(1, nframes):
            # Switched from state 0 to state 1 from frame j - 1 to frame j
            if state_curr == 0 and assign[i][j] == 1:
                lifetimes_traj.append((0, lifetime_curr))
                lifetime_curr = 1
                state_curr = 1
            # Switched from state 1 to state 0 from frame j - 1 to frame j
            elif state_curr == 1 and assign[i][j] == 0:
                lifetimes_traj.append((1, lifetime_curr))
                lifetime_curr = 1
                state_curr = 0
            # Did not switch from frame j - 1 to frame j
            else:
                lifetime_curr += 1
        
        # Omit the starting and ending states if desired 
        if not include_ends:
            lifetimes_traj = lifetimes_traj[1:-1]

        # Sort out the lifetimes of the two states
        for state, lifetime in lifetimes_traj:
            if state == 0:
                lifetimes1.append(lifetime)
            else:
                lifetimes2.append(lifetime)

    # Return lifetimes in hours 
    return (
        5 * np.array(lifetimes1, dtype=np.float64) / 60,
        5 * np.array(lifetimes2, dtype=np.float64) / 60
    )

######################################################################
def simulate(model, n, tmax, rng, start=None):
    """
    Generate a collection of simulated trajectories from the given HMM.

    Parameters
    ----------
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    n : int
        Number of trajectories. 
    tmax : float
        Maximum time for each trajectory.
    rng : `numpy.random.Generator`
        Random number generator. 
    start : int
        Starting state; if None, set to a random state. 

    Returns
    -------
    Generated trajectories. 
    """
    # Get the mean RFI values for each state and the transition matrix
    mean1 = model.distributions[0].means[0].item()
    mean2 = model.distributions[1].means[0].item()
    transition_matrix = np.exp(model.edges)
    
    # Sample n trajectories ... 
    trajectories = np.zeros((n, tmax + 1, 3))   # Add a dummy column at index 1
    for i in range(n):
        trajectories[i, :, 0] = np.arange(tmax + 1)
        
        # Choose a starting state for each trajectory 
        t = 0
        if start is None:
            prob1, prob2 = np.exp(model.starts)
            state = rng.choice(2, p=[prob1, prob2]) 
        else:
            state = start

        # Run the trajectory until it reaches the desired timepoint
        trajectory = [state]
        while t < tmax:
            prob_stay = transition_matrix[state, state]
            r = rng.random()
            if r > prob_stay:
                state = (1 if state == 0 else 0)
            trajectory.append(state)
            t += 1
        trajectories[i, :, 2] = mean1 + (mean2 - mean1) * np.array(trajectory)

    return trajectories

######################################################################
def get_mean_lifetimes_exp_vs_sim(data, model, tmin, tmax, nsim, rng,
                                  include_ends=True):
    """
    Plot the empirical lifetime distributions of a set of trajectories, along
    with the lifetime distributions of a set of simulated trajectories of 
    the given HMM. 

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    tmin : int
        Minimum time for each trajectory.
    tmax : int
        Maximum time for each trajectory.
    nsim : int
        Number of simulated trajectories. 
    rng : `numpy.random.Generator`
        Random number generator.
    include_ends : bool
        If True, include the lifetimes of the starting and ending states
        in the statistics. 
    """
    # Get state assignments for each trajectory
    assign, values = predict_from_hmm(data, model)

    # Simulate more trajectories from the model and plot their lifetime
    # distributions between tmin and tmax
    trajectories = simulate(model, nsim, tmax, rng, start=0)
    subtrajectories = []
    for i in range(nsim):
        subtraj_i = find_subtrajectory(trajectories[i, :, :], tmin, tmax)
        subtrajectories.append(subtraj_i)
    subtrajectories = np.array(subtrajectories)
    mean1 = model.distributions[0].means[0].item()
    mean2 = model.distributions[1].means[0].item()
    assign_sim = (subtrajectories[:, :, 2] > mean1 + 1e-3)

    # Get mean lifetimes from the data ...
    emp_lifetimes1, emp_lifetimes2 = get_lifetime_dists(assign, include_ends=include_ends)
    emp_mean1 = np.mean(emp_lifetimes1)
    emp_mean2 = np.mean(emp_lifetimes2)

    # ... and from the simulated trajectories
    sim_lifetimes1, sim_lifetimes2 = get_lifetime_dists(assign_sim, include_ends=include_ends)
    sim_mean1 = np.mean(sim_lifetimes1)
    sim_mean2 = np.mean(sim_lifetimes2)

    return np.array([emp_mean1, emp_mean2]), np.array([sim_mean1, sim_mean2])

######################################################################
#                   BOOTSTRAPPING, VALIDATING, ETC.                  #
######################################################################
def fit_hmm_subsample(data, rng, nsubset, nruns):
    """
    Fit a two-state HMM to randomly chosen subsamples of the given
    trajectories.

    The input data array is assumed to be a *3-D array* with dimensions
    (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.

    This dataset may or may not contain overlapping trajectories, which 
    are excised appropriately in each subsampling run. 
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray` or list
        3-D array or list of 2-D arrays of trajectory data, as specified
        above.
    rng : `numpy.random.Generator`
        Random number generator. 
    nsubset : int
        Number of trajectories to include per subsample. 
    nruns : int
        Number of subsampling runs. 

    Returns
    -------
    Array of model parameters for the fitted HMM instances. 
    """
    nsample = data.shape[0]
    subset_params = np.zeros((nruns, 6), dtype=np.float64)

    # In each run ... 
    for i in range(nruns):
        # Generate a subsample of the data and fit an HMM 
        subset_idx = rng.choice(nsample, size=nsubset, replace=False)
        subdata = data[subset_idx, :, :]
        _, params = fit_hmm(make_data_unique(subdata), verbose=False)
        subset_params[i, :] = params

    return subset_params

######################################################################
def compare_exp_vs_sim_lifetimes_brute_force(data, model, nmesh, tmin, tmax,
                                             nsim, rng, include_ends=True):
    """
    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.

    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray` or list
        3-D array or list of 2-D arrays of trajectory data, as specified
        above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    nmesh : int
        Array size for varying model parameters. 
    tmin : int
        Minimum time for each trajectory.
    tmax : int
        Maximum time for each trajectory.
    nsim : int
        Number of simulated trajectories. 
    rng : `numpy.random.Generator`
        Random number generator.
    include_ends : bool
        If True, include the lifetimes of the starting and ending states
        in the statistics.

    Returns
    -------
    Absolute differences between mean lifetimes in empirical vs. simulated
    trajectories. 
    """
    emp_means, sim_means = get_mean_lifetimes_exp_vs_sim(
        data, model, tmin, tmax, nsim, rng, include_ends=include_ends
    )
    errors = np.abs(emp_means - sim_means)

    # Initialize an HMM with the same topology, emission probabilities, 
    # start probabilities, and end probabilities 
    base_model = DenseHMM(verbose=False)
    means = [
        model.distributions[0].means[0].item(), 
        model.distributions[1].means[0].item()
    ]
    covs = [
        model.distributions[0].covs[0].item(),
        model.distributions[1].covs[0].item()
    ]
    d1 = Normal(means=[means[0]], covs=[covs[0]], covariance_type='diag')
    d2 = Normal(means=[means[1]], covs=[covs[1]], covariance_type='diag')
    base_model.add_distributions([d1, d2])
    base_model.add_edge(base_model.start, d1, np.exp(model.starts[0]))
    base_model.add_edge(base_model.start, d2, np.exp(model.starts[1]))
    p1_end = np.exp(model.ends[0])
    p2_end = np.exp(model.ends[1])
    base_model.add_edge(d1, base_model.end, p1_end)
    base_model.add_edge(d2, base_model.end, p2_end)
    base_model.add_edge(d1, d1, 0.99 / (1 + p1_end))
    base_model.add_edge(d1, d2, 0.01 / (1 + p1_end))
    base_model.add_edge(d2, d1, 0.01 / (1 + p2_end))
    base_model.add_edge(d2, d2, 0.99 / (1 + p2_end))

    # Set up an array of transition probabilities 
    error_dist = np.zeros((nmesh, nmesh, 2))
    p1_mesh = np.logspace(-3, -1, nmesh)   # Prob of 1 -> 2 transition
    p2_mesh = np.logspace(-3, -1, nmesh)   # Prob of 2 -> 1 transition
    for i, p1 in enumerate(p1_mesh):
        for j, p2 in enumerate(p2_mesh):
            base_model.edges[0, 0] = np.log((1 - p1) / (1 + p1_end))
            base_model.edges[0, 1] = np.log(p1 / (1 + p1_end))
            base_model.edges[1, 0] = np.log(p2 / (1 + p2_end))
            base_model.edges[1, 1] = np.log((1 - p2) / (1 + p2_end))
            emp_means, sim_means = get_mean_lifetimes_exp_vs_sim(
                data, base_model, tmin, tmax, nsim, rng, include_ends=include_ends
            )
            error_dist[i, j, :] = np.abs(emp_means - sim_means)

    return errors, error_dist

######################################################################
#                              PLOTTING                              #
######################################################################
def plot_trajectory_fits(data, names, model, fig, axes, plot_hmm=True,
                         plot_cdg=True, plot_ribo=True):
    """
    Plot the maximum-likelihood assignment of hidden states along a given 
    trajectory, according to the given HMM. 

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    names : list
        List of corresponding filenames. 
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    fig : `matplotlib.pyplot.Figure`
        Figure instance. 
    axes : `numpy.ndarray` of `matplotlib.pyplot.Axes`
        Array of axes onto which to plot the trajectories and their state
        assignments.
    plot_hmm : bool
        If True, plot the inferred state assignments. 
    plot_cdg : bool
        If True, plot the cdGreen2 trajectories.
    plot_ribo : bool
        If True, plot the corresponding riboswitch trajectories. 

    Returns
    -------
    Array of state assignments, both as a binary vector and as a vector of
    idealized RFI values. 
    """
    nrows, ncols = axes.shape
    nplots = axes.size
    if not plot_hmm and not plot_cdg and not plot_ribo:
        raise RuntimeError('Must specify at least one variable to be plotted')
    
    # Get state assignments for each trajectory
    #
    # Here, we assume that the input data is a complete 3-D array
    _, values = predict_from_hmm(data, model)

    # Plot each trajectory with its state assignments 
    c_state = sns.color_palette('deep')[1]     # HMM state
    c_cdg = sns.color_palette('deep')[2]       # cdGreen2
    c_ribo = sns.color_palette('pastel')[0]    # Riboswitch 
    times = data[:, :, 0] * 5 / 60    # Timepoints in hours
    for i in range(nplots):
        j = i // ncols
        k = i % ncols
        print('Plotting in axes[{}, {}]:'.format(j, k), names[i])
        if plot_hmm:
            axes[j, k].plot(times[i, :], values[i, :], c=c_state, zorder=2)
        if plot_cdg:
            axes[j, k].plot(times[i, :], data[i, :, 2], c=c_cdg, zorder=1)
        if plot_ribo:
            axes[j, k].plot(times[i, :], data[i, :, 3], c=c_ribo, zorder=0)

    # Set axes limits
    ymin = data[:nplots, :, 2:4].min() - 0.02
    ymax = data[:nplots, :, 2:4].max() + 0.02
    for j in range(nrows):
        for k in range(ncols):
            axes[j, k].set_ylim([ymin, ymax])

    # Set axes labels 
    for j in range(nrows):
        axes[j, 0].set_ylabel('RFI')
    for k in range(ncols):
        axes[-1, k].set_xlabel('Time (h)')

    # Add legend
    labels = []
    if plot_hmm:
        labels.append('HMM')
    if plot_cdg:
        labels.append('cdGreen2')
    if plot_ribo:
        labels.append('Riboswitch')
    fig.legend(labels, loc='outside upper right', ncols=3, fontsize=10)

    return fig, axes

######################################################################
def plot_lifetimes(data, model, axes, include_ends=False, xlog=True, ylog=False,
                   bins1=None, bins2=None):
    """
    Plot the maximum-likelihood assignment of hidden states along a given
    trajectory, according to the given HMM.

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance. 
    fig : `matplotlib.pyplot.Figure`
        Figure instance. 
    axes : `numpy.ndarray` of `matplotlib.pyplot.Axes`
        Array of axes onto which to plot the trajectories and their state
        assignments.
    include_ends : bool
        If True, include the lifetimes of the starting and ending states
        in the statistics.
    xlog : bool
        If True, plot x-axis in log scale. 
    ylog : bool
        If True, plot y-axis in log scale. 
    bins1 : int or array-like
        Number of bins for state 1 histogram, or the array of bin edges.  
    bins2 : int
        Number of bins for state 2 histogram, or the array of bin edges.

    Returns
    -------
    Updated axes and lifetime distributions. 
    """
    # Get state assignments for each trajectory
    assign, values = predict_from_hmm(data, model)

    # Get lifetime distributions
    lifetimes1, lifetimes2 = get_lifetime_dists(assign, include_ends=include_ends)

    # Get mean lifetimes
    emp_mean1 = np.mean(lifetimes1)
    emp_mean2 = np.mean(lifetimes2)
    transition_matrix = np.exp(model.edges)
    hmm_mean1 = (5 / transition_matrix[0, 1]) / 60
    hmm_mean2 = (5 / transition_matrix[1, 0]) / 60

    # Transform to log-scale if desired
    if xlog:
        lifetimes1 = np.log10(lifetimes1)
        lifetimes2 = np.log10(lifetimes2)
        emp_mean1 = np.log10(emp_mean1)
        emp_mean2 = np.log10(emp_mean2)
        hmm_mean1 = np.log10(hmm_mean1)
        hmm_mean2 = np.log10(hmm_mean2)

    # Plot histograms
    if bins1 is None:
        bins1 = 10
    if bins2 is None:
        bins2 = 10
    axes[0].hist(lifetimes1, bins=bins1, density=True)
    axes[1].hist(lifetimes2, bins=bins2, density=True)

    # Re-scale y-axes if desired
    if ylog:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')

    # Plot vertical lines mapping (1) the empirical mean lifetimes and (2) the
    # mean lifetimes from the HMM
    ylim0 = axes[0].get_ylim()
    ylim1 = axes[1].get_ylim()
    axes[0].plot([emp_mean1, emp_mean1], ylim0, linestyle='--', linewidth=2)
    axes[1].plot([emp_mean2, emp_mean2], ylim1, linestyle='--', linewidth=2)
    axes[0].set_ylim(ylim0)
    axes[1].set_ylim(ylim1)
    axes[0].plot([hmm_mean1, hmm_mean1], ylim0, linestyle='--', linewidth=2)
    axes[1].plot([hmm_mean2, hmm_mean2], ylim1, linestyle='--', linewidth=2)
    axes[0].set_ylim(ylim0)
    axes[1].set_ylim(ylim1)

    return axes, lifetimes1, lifetimes2

######################################################################
def plot_lifetimes_exp_vs_sim(data, model, tmin, tmax, nsim, rng, outfilename,
                              xlog=True, ylog=False, nbins=15, include_ends=True):
    """
    Plot the empirical lifetime distributions of a set of trajectories, along
    with the lifetime distributions of a set of simulated trajectories of 
    the given HMM. 

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    nsim : int
        Number of simulated trajectories. 
    rng : `numpy.random.Generator`
        Random number generator.
    outfilename : str
        Path to output PDF file. 
    xlog : bool
        If True, plot x-axis in log scale. 
    ylog : bool
        If True, plot y-axis in log scale.
    nbins : int
        Number of bins in each histogram.
    include_ends : bool
        If True, include the lifetimes of the starting and ending states
        in the statistics. 
    """
    # Get state assignments for each trajectory
    assign, values = predict_from_hmm(data, model)

    # Simulate more trajectories from the model and plot their lifetime
    # distributions between tmin and tmax
    trajectories = simulate(model, nsim, tmax, rng, start=0)
    subtrajectories = []
    for i in range(nsim):
        subtraj_i = find_subtrajectory(trajectories[i, :, :], tmin, tmax)
        subtrajectories.append(subtraj_i)
    subtrajectories = np.array(subtrajectories)
    mean1 = model.distributions[0].means[0].item()
    mean2 = model.distributions[1].means[0].item()
    assign_sim = (subtrajectories[:, :, 2] > mean1 + 1e-3)

    # Get the minimum and maximum RFI values along all empirical and simulated
    # trajectories
    lifetimes1_exp, lifetimes2_exp = get_lifetime_dists(assign, include_ends=include_ends)
    lifetimes1_sim, lifetimes2_sim = get_lifetime_dists(assign_sim, include_ends=include_ends)
    lifetimes1_min = min([lifetimes1_exp.min(), lifetimes1_sim.min()])
    lifetimes2_min = min([lifetimes2_exp.min(), lifetimes2_sim.min()])
    lifetimes1_max = max([lifetimes1_exp.max(), lifetimes1_sim.max()])
    lifetimes2_max = max([lifetimes2_exp.max(), lifetimes2_sim.max()])
    if xlog:
        bins1 = np.linspace(np.log10(lifetimes1_min - 1e-4), np.log10(lifetimes1_max + 1e-4), nbins + 1)
        bins2 = np.linspace(np.log10(lifetimes2_min - 1e-4), np.log10(lifetimes2_max + 1e-4), nbins + 1)
    else:
        bins1 = np.linspace(lifetimes1_min - 1e-4, lifetimes1_max + 1e-4, nbins + 1)
        bins2 = np.linspace(lifetimes2_min - 1e-4, lifetimes2_max + 1e-4, nbins + 1)

    # Infer and plot hidden state assignments corresponding to trajectories
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
    plot_lifetimes(
        data, model, axes[0, :], include_ends=include_ends, xlog=xlog,
        ylog=ylog, bins1=bins1, bins2=bins2
    )

    # Simulate more trajectories from the model and plot their lifetime distributions
    # between 2h and 10h
    plot_lifetimes(
        subtrajectories, model, axes[1, :], include_ends=include_ends,
        xlog=xlog, ylog=ylog, bins1=bins1, bins2=bins2
    )

    # Add titles for the top-row plots
    axes[0, 0].set_title('Low-c-di-GMP state lifetimes')
    axes[0, 1].set_title('High-c-di-GMP state lifetimes')

    # Add axes labels for the top-row plots
    for j in range(2):
        axes[-1, j].set_xlabel('Time (h)')
    for i in range(2):
        axes[i, 0].set_ylabel('Density')

    # Add legend
    axes[0, 0].legend(['Empirical', 'HMM'], fontsize=8, loc='upper left')

    # Annotate the plot with sample sizes
    xy1 = (0.99, 0.95)
    xy2 = (0.015, 0.95)
    for i in range(2):
        if i == 0:
            n_traj = len(assign)
            n_lifetimes1 = lifetimes1_exp.size
            n_lifetimes2 = lifetimes2_exp.size
        else:
            n_traj = nsim
            n_lifetimes1 = lifetimes1_sim.size
            n_lifetimes2 = lifetimes2_sim.size
        if i == 0:
            text1 = '{} experimental\n{} lifetimes'.format(n_traj, n_lifetimes1)
            text2 = '{} experimental\n{} lifetimes'.format(n_traj, n_lifetimes2)
        else:
            text1 = '{} simulated\n{} lifetimes'.format(n_traj, n_lifetimes1)
            text2 = '{} simulated\n{} lifetimes'.format(n_traj, n_lifetimes2)
        axes[i, 0].annotate(
            text1, xy=xy1, xycoords='axes fraction', horizontalalignment='right',
            verticalalignment='top'
        )
        if xlog:
            axes[i, 1].annotate(
                text2, xy=xy2, xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top'
            )
        else:
            axes[i, 1].annotate(
                text2, xy=xy1, xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top'
            )

    # Enforce tight layout
    plt.tight_layout()

    # Update x-axis ticks and labels
    if xlog:
        for i in range(2):
            for j in range(2):
                xticks = axes[i, j].get_xticks()[1:-1]
                axes[i, j].set_xticks(xticks)
                axes[i, j].set_xticklabels([r'$10^{' + str(x) + '}$' for x in xticks])

    plt.savefig(outfilename)

######################################################################
#               LOCATING AND PLOTTING SWITCHING EVENTS               #
######################################################################
def find_switch_times(data, model, delta_backward, delta_forward, max_error_01=0.02,
                      max_error_10=0.02):
    """
    Find timepoints along each trajectory at which a switch from one state 
    to the other occurs.

    These timepoints t are chosen such that, in the window [t - delta_backward,
    t + delta_forward] (endpoints inclusive), there is only one switching
    event (from time t - 1 to time t).

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    delta_backward : int
        Backward window half-length.
    delta_forward : int 
        Forward window half-length.
    max_error : float
        Maximum error between HMM state assignments and cdGreen2 trajectories.

    Returns 
    -------
    Array of switching timepoints along each trajectory. 
    """
    nsample = len(data)

    # Get state assignments for each trajectory
    assign, values = predict_from_hmm(data, model)

    # For each trajectory ...
    switches = [{'01': [], '10': []} for i in range(nsample)]
    for i in range(nsample):
        # Start from the earliest timepoint at which the window is well-defined
        nframes = data[i].shape[0]
        for t_curr in range(delta_backward, nframes - delta_forward - 1):
            # Get the window at each timepoint (this window should have length
            # delta_backward + delta_forward + 1, including the current timepoint)
            window = range(t_curr - delta_backward, t_curr + delta_forward + 1)
            data_window = data[i][window, 2]
            assign_window = assign[i][window]
            values_window = values[i][window]
            t0, t1 = delta_backward - 1, delta_backward
            
            # Is there a switching event at the current timepoint? 
            if assign_window[t0] != assign_window[t1]:
                # If so, is that the only switching event within the window? 
                if (np.all(assign_window[:t0] == assign_window[t0]) and
                    np.all(assign_window[t1:] == assign_window[t1])):
                    # If so, is the error between the trajectory and the 
                    # state assignments within that window small enough? 
                    error = np.abs(values_window - data_window).mean()
                    if ((assign_window[t0] == 0 and error < max_error_01) or
                        (assign_window[t0] == 1 and error < max_error_10)):
                        # If so, collect this switching event
                        if assign_window[t0] == 0:
                            switches[i]['01'].append(t_curr)
                        else:
                            switches[i]['10'].append(t_curr)

    return switches
        
######################################################################
def plot_aligned_trajectories(data, model, times, delta_backward, delta_forward,
                              ax, alpha=1.0, truncate_at_further_switches=True):
    """
    Plot the given trajectories, aligned at the given timepoints. 

    The input data array is assumed to have dimensions (ns, nt, 8), where:
    - ns is the number of samples, and
    - nt is the number of timepoints.
    
    Each timepoint has values for the following 8 quantities:
    0) Frame index (the time between frames is 5 minutes)
    1) Constitutive reporter fluorescence intensity
    2) cdGreen2 relative fluorescence intensity
    3) Riboswitch relative fluorescence intensity
    4) Minimum cdGreen2 RFI among all cells at the given timepoint
    5) Maximum cdGreen2 RFI among all cells at the given timepoint
    6) Minimum riboswitch RFI among all cells at the given timepoint
    7) Maximum riboswitch RFI among all cells at the given timepoint

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array of trajectory data, as specified above.
    model : `pomegranate.hmm.DenseHMM`
        Fitted HMM instance.
    delta_backward : int
        Backward window half-length for plotting.
    delta_forward : int 
        Forward window half-length for plotting.
    ax : `matplotlib.pyplot.Axes`

    Returns 
    -------
    Updated axes. 
    """
    nsample = len(data)

    # Get state assignments for each trajectory
    assign, values = predict_from_hmm(data, model)

    # For each timepoint ...
    for i, (idx, t_curr) in enumerate(times):
        # Get the window at each timepoint (this window should have length
        # 2 * delta + 1, including the current timepoint)
        t0 = t_curr - delta_backward
        t1 = t_curr + delta_forward + 1
        if truncate_at_further_switches:
            # Assuming there is a switch at [t_curr - 1, t_curr], check if
            # there are switches between t_curr and t_curr + delta_forward
            #
            # If there is a switch between t_curr + j and t_curr + j + 1, 
            # truncate the trajectory at t_curr + j
            for j in range(delta_forward):
                # If we have reached the end of the trajectory, break
                if t_curr + j >= values[idx].shape[0] or t_curr + j + 1 >= values[idx].shape[0]:
                    t1 = values[idx].shape[0]
                    break
                # If we have reached a switching event, break 
                if values[idx][t_curr + j] != values[idx][t_curr + j + 1]:
                    t1 = t_curr + j
                    break
        data_window = data[idx][t0:t1, 3]
        assign_window = values[idx][t0:t1]
        window_length = data_window.shape[0]

        # Plot the riboswitch RFIs along the trajectory
        ax.plot(
            np.arange(window_length), data_window,
            color=sns.color_palette()[0],
            zorder=3,
            alpha=alpha
        )

    # Shade the time increment during which the switch occurs
    ymin, ymax = ax.get_ylim()
    rect = Rectangle(
        (delta_backward - 1, ymin), 1, ymax - ymin,
        linewidth=0, facecolor=sns.color_palette('pastel')[7], zorder=0
    )
    ax.add_patch(rect)
    ax.set_ylim([ymin, ymax])

    return ax

######################################################################
def fit_to_ord2_ode(trajectories, ypinit, rng):
    """
    Fit the given trajectories to a second-order ODE of the form, 

    a*y''(t) + b*y'(t) + c*y(t) = 0,

    given estimates of their derivatives at t = 0. 
    """
    # Define an array of timepoints
    #
    # Note that each trajectory may have a different length due to truncation
    t = [np.arange(len(trajectories[i])) for i in range(len(trajectories))]

    def objective_composite_ord2(params):
        """
        Objective function to be minimized.

        The solution should have the form c1*exp(r1*t) + c2*exp(r2*t), or,
        if r1 == r2, c1*exp(r1*t) + c2*t*exp(r1*t).
        """
        r1, r2 = params
        ypred = []
        if r1 == r2:
            for i in range(len(trajectories)):
                yinit = trajectories[i][0]
                c1 = yinit
                c2 = ypinit[i] - r1 * yinit
                ypred.append(c1 * np.exp(r1 * t[i]) + c2 * t[i] * np.exp(r1 * t[i]))
        else:
            for i in range(len(trajectories)):
                yinit = trajectories[i][0]
                c1 = (ypinit[i] - yinit * r2) / (r1 - r2)
                c2 = yinit - c1
                ypred.append(c1 * np.exp(r1 * t[i]) + c2 * np.exp(r2 * t[i]))

        # Unravel the data/predictions so that the error can be passed as 
        # a 1-D array 
        dev = []
        for i in range(len(trajectories)):
            dev += [pred - traj for pred, traj in zip(ypred[i], trajectories[i])]
        return np.array(dev)

    # Perform least-squares minimization, changing the initial conditions
    # as necessary 
    fit = least_squares(
        objective_composite_ord2, [-0.01, -0.01], bounds=[[-10, -10], [10, 10]],
        verbose=1
    )
    while not fit.success or fit.status == 3:
        fit = least_squares(
            objective_composite_ord2, -1 + 2 * rng.random((2,)),
            bounds=[[-10, -10], [10, 10]], verbose=1
        )
    r1, r2 = fit.x

    # Evaluate the fitted function for each trajectory
    #
    # Again, note that each fit may have a different length 
    fits = []
    if r1 == r2:
        for i in range(len(trajectories)):
            yinit = trajectories[i][0]
            c1 = yinit
            c2 = ypinit[i] - r1 * yinit
            fits.append(c1 * np.exp(r1 * t[i]) + c2 * t[i] * np.exp(r1 * t[i]))
    else:
        for i in range(len(trajectories)):
            yinit = trajectories[i][0]
            c1 = (ypinit[i] - yinit * r2) / (r1 - r2)
            c2 = trajectories[i][0] - c1
            fits.append(c1 * np.exp(r1 * t[i]) + c2 * np.exp(r2 * t[i]))

    return fits
