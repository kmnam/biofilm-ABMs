"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/11/2025
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

#######################################################################
#                           MIXTURE MODELS                            #
#######################################################################
def fit_cdGMP_gaussian_mixture(data, n_components, seed=42):
    """
    Fit a Gaussian mixture model with the given number of components to the
    c-di-GMP levels in the given biofilm at the given timepoint.

    Parameters
    ----------
    data : `numpy.ndarray`
        1-D array of c-di-GMP levels. 
    n_components : int
        Number of Gaussian components. 
    seed : int
        Seed for random number generation.

    Returns
    -------
    A fitted `GaussianMixture` instance, alongside its BIC score. 
    """
    model = GaussianMixture(
        n_components=n_components, covariance_type='full', random_state=seed
    )
    model.fit(data.reshape(-1, 1))
    return model, model.bic(data.reshape(-1, 1))

#######################################################################
def compare_gaussian_vs_mixture(data, seed=42):
    """
    Fit both a 1-component (i.e., pure Gaussian) and 2-component Gaussian
    mixture model to the given distribution of c-di-GMP levels, and compare
    them according to their BICs. 

    Parameters
    ----------
    data : `numpy.ndarray`
        1-D array of c-di-GMP levels. 
    seed : int
        Seed for random number generation.

    Returns
    -------
    The mean(s), standard deviations(s), and weight(s) of the preferred
    model. 
    """
    # Fit the two models 
    model1, bic1 = fit_cdGMP_gaussian_mixture(data, 1, seed=seed)
    model2, bic2 = fit_cdGMP_gaussian_mixture(data, 2, seed=seed)
    params = np.nan * np.ones((6,))

    # If the pure Gaussian scores better than the mixture ... 
    if bic1 < bic2:
        # In this case, only update the second Gaussian (with the "higher"
        # mean)
        params[1] = model1.means_[0, 0]
        params[3] = np.sqrt(model1.covariances_[0, 0, 0])
        params[4] = 0.0
        params[5] = 1.0
    else:    # Otherwise ... 
        # Update the means and SDs such that mean1 < mean2
        idxmin = 0 if model2.means_[0, 0] < model2.means_[1, 0] else 1
        idxmax = int(not idxmin)
        params = np.array([
            model2.means_[idxmin, 0],
            model2.means_[idxmax, 0],
            np.sqrt(model2.covariances_[idxmin, 0, 0]),
            np.sqrt(model2.covariances_[idxmax, 0, 0]),
            model2.weights_[idxmin],
            model2.weights_[idxmax]
        ])

    return params

#######################################################################
def fit_cdGMP_gaussian_mixtures_per_biofilm(filenames, col, seed=42):
    """
    Fit a sequence of Gaussian mixture models to the c-di-GMP levels in
    the given biofilm at the given timepoints.

    Parameters
    ----------
    filenames : list of str
        Paths to files containing data pertaining to the given biofilm.
    col : str
        Column name for c-di-GMP level.
    seed : int
        Seed for random number generation.

    Returns
    -------
    The mean(s), standard deviations(s), and weight(s) of the 1- or 2-
    component Gaussian mixture models fitted for each timepoint. 
    """
    means = np.zeros((len(filenames), 2), dtype=np.float64)
    stds = np.zeros((len(filenames), 2), dtype=np.float64)
    weights = np.zeros((len(filenames), 2), dtype=np.float64)
    
    # Assume that, once a biofilm begins to exhibit c-di-GMP bimodality,
    # it continues to do so during the given time window 
    is_bimodal = False

    # For each timepoint ... 
    for i, filename in enumerate(filenames):
        data = pd.read_csv(filename, sep=',')[col].to_numpy()

        # If bimodality has not yet been encountered, try fitting two models,
        # one with one component and the other with two components, and choose
        # one model over the other based on the BIC
        if not is_bimodal:
            params = compare_gaussian_vs_mixture(data, seed=seed)
            means[i, :] = params[:2]
            stds[i, :] = params[2:4]
            weights[i, :] = params[4:]
            # Is the preferred model a two-component mixture?
            if not np.isnan(params[0]):
                is_bimodal = True
        # Otherwise, assume bimodality and simply fit with a two-component mixture 
        else:
            model, _ = fit_cdGMP_gaussian_mixture(data, 2, seed=seed)
            idxmin = 0 if model.means_[0, 0] < model.means_[1, 0] else 1
            idxmax = int(not idxmin)
            means[i, 0] = model.means_[idxmin, 0]
            means[i, 1] = model.means_[idxmax, 0]
            stds[i, 0] = np.sqrt(model.covariances_[idxmin, 0, 0])
            stds[i, 1] = np.sqrt(model.covariances_[idxmax, 0, 0])
            weights[i, 0] = model.weights_[idxmin]
            weights[i, 1] = model.weights_[idxmax]
            
    return means, stds, weights

#######################################################################
def run_fit_cdGMP_gaussian_mixtures(brep_prefixes, col, tstart, tend, axes,
                                    nbins=30, plot_weights=True):
    """
    Fit Gaussian mixture models to the c-di-GMP levels in all biofilms within
    a given biological replicate in the given time window.

    Parameters
    ----------
    brep_prefixes : list of str
        Prefixes for paths to all files containing data for the given
        biological replicate.
    col : str
        Column name for c-di-GMP level.
    tstart : int
        Start time. 
    tend : int
        End time (inclusive). 
    axes : `numpy.ndarray` of `matplotlib.pyplot.Axes`
        Axes onto which to plot the histograms.
    nbins : int
        Number of bins.

    Returns
    -------
    Array of weights for the higher-mean component in each mixture model. 
    """
    # Initialize figures and weights
    n = len(brep_prefixes)
    times = list(range(tstart, tend + 1))
    weights_all = np.zeros((tend - tstart + 1, n), dtype=np.float64)

    # For each biofilm in the given biological replicate ...
    for i, prefix in enumerate(brep_prefixes):
        name = 'MP' + prefix.split('/')[-1][-1]
        filenames = ['{}_total_time{:04d}.csv'.format(prefix, t) for t in times]

        # Get the maximum fluorescence intensity in each biofilm
        xmax = max(
            pd.read_csv(filename, sep=',')[col].max() for filename in filenames
        )

        # Plot the fluorescence intensity histogram for each biofilm
        bins = np.linspace(0, xmax, nbins + 1)
        for j, (t, filename) in enumerate(zip(times, filenames)):
            data = pd.read_csv(filename, sep=',')[col]
            axes[j, i].hist(
                data, bins=bins, density=True, color=sns.color_palette()[0]
            )
            axes[j, i].set_title(
                r'{}, $t = {}$h, $n = {}$'.format(name, t, data.shape[0]),
                size=10
            )
            axes[j, i].set_xlim([-0.1, xmax])

        # Fit Gaussian mixture models
        means, stds, weights = fit_cdGMP_gaussian_mixtures_per_biofilm(
            filenames, col, seed=42
        )
        weights_all[:, i] = weights[:, 1]

        # For each model, if the model is a two-component mixture, plot its
        # density
        for j in range(means.shape[0]):
            if not np.isnan(means[j, 0]):
                x = np.linspace(0, xmax, 100)
                y1 = weights[j, 0] * norm.pdf(x, means[j, 0], stds[j, 0])
                y2 = weights[j, 1] * norm.pdf(x, means[j, 1], stds[j, 1])
                axes[j, i].plot(x, y1, color=sns.color_palette()[1])
                axes[j, i].plot(x, y2, color=sns.color_palette()[2])

        # Plot the Gaussian weights
        if plot_weights:
            axes[-1, i].scatter(times, weights[:, 0], color=sns.color_palette()[1])
            axes[-1, i].scatter(times, weights[:, 1], color=sns.color_palette()[2])
            axes[-1, i].set_ylim([0, 1])
            axes[-1, i].set_title('{}, weights'.format(name), size=10)

    return weights_all

#######################################################################
#                              PLOTTING                               #
#######################################################################
def plot_cdGMP_histograms_per_replicate(brep_prefixes, col, tstart, tend, axes,
                                        nbins=30):
    """
    Plot the c-di-GMP histograms for all biofilms within a given biological
    replicate within the given time window.

    Parameters
    ----------
    brep_prefixes : list of str
        Prefixes for paths to all files containing data for the given
        biological replicate.
    col : str
        Column name for c-di-GMP level.
    tstart : int
        Start time. 
    tend : int
        End time (inclusive). 
    axes : `numpy.ndarray` of `matplotlib.pyplot.Axes`
        Axes onto which to plot the histograms. 
    nbins : int
        Number of bins. 
    """
    times = list(range(tstart, tend + 1))

    # For each biofilm in the biological replicate ...
    for i, prefix in enumerate(brep_prefixes):
        # Get the subset of populations corresponding to the given time window
        filenames = ['{}_total_time{:04d}.csv'.format(prefix, t) for t in times]

        # Get the maximum fluorescence intensity in each biofilm
        xmax = max(
            pd.read_csv(filename, sep=',')[col].max() for filename in filenames
        )

        # Plot the c-di-GMP histogram for each biofilm
        bins = np.linspace(0, xmax, nbins + 1)
        name = 'MP' + prefix.split('/')[-1][-1]
        for j, (t, filename) in enumerate(zip(times, filenames)):
            data = pd.read_csv(filename, sep=',')[col]
            axes[j, i].hist(data, bins=bins, color=sns.color_palette()[0])
            axes[j, i].set_title(
                r'{}, $t = {}$h, $n = {}$'.format(name, t, data.shape[0]),
                size=10
            )

    return axes

