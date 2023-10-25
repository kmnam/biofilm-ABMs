"""
Various biofilm-related metrics. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/25/2023
"""
import numpy as np
from scipy.stats import spearmanr, kendalltau

#######################################################################
def radial_sortedness(cells, scores, rng):
    """
    Compute the radial sortedness of the given biofilm with respect to the 
    given array of scores.

    Adapted from code by Jung-Shen Benny Tai. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    scores : `numpy.ndarray`
        1-D array of scores (one for each cell). 
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Radial sortedness of the biofilm. 
    """
    # Compute radial distance from biofilm center
    center = cells[:, :2].sum(axis=0) / cells.shape[0]
    distances = np.linalg.norm(cells[:, 2] - center, axis=1)

    # Get the scores in ascending and descending order
    scores_ascend = np.sort(scores)
    scores_descend = scores_ascend[::-1]

    # Get the indices of the cells in order of ascending radial distance
    dists_ascend_idx = np.argsort(distances)

    # Get the absolute difference between the scores in descending order 
    # (perfect sorting) and random permutations of the scores
    diffs = np.zeros((100,), dtype=np.float64)
    for i in range(100):
        diffs[i] = np.linalg.norm(scores_descend - rng.permutation(scores), ord=1)
    diff_mean = np.mean(diffs)

    # Get the absolute difference between the scores in descending order 
    # (perfect sorting) and the scores in ascending order (inverse perfect
    # sorting)
    diff_ascend = np.linalg.norm(scores_descend - scores_ascend, ord=1)

    # Get the scores in order of ascending radial distance
    scores_by_dist = scores[dists_ascend_idx]

    # Get the absolute difference between the scores in order of ascending 
    # radial distance ("the data") and the scores in descending order
    # (perfect sorting)
    diff_data = np.linalg.norm(scores_by_dist - scores_descend, ord=1)

    # Compute the radial sortedness index
    if diff_data < diff_mean:
        return 1 - diff_data / diff_mean
    elif diff_data > diff_mean:
        return -(diff_data - diff_mean) / (diff_ascend - diff_mean)
    else:
        return 0.0

#######################################################################
def radial_spearman_coeff(cells, scores):
    """
    Compute the Spearman correlation coefficient between the cells'
    radial distances and their given scores.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    scores : `numpy.ndarray`
        1-D array of scores (one for each cell). 
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Spearman correlation coefficient between radial distance and score.
    """
    # Compute radial distance from biofilm center
    center = cells[:, :2].sum(axis=0) / cells.shape[0]
    distances = np.linalg.norm(cells[:, 2] - center, axis=1)

    return spearmanr(distances, scores).statistic

#######################################################################
def radial_kendall_tau(cells, scores):
    """
    Compute the Kendall's tau correlation coefficient between the cells'
    radial distances and their given scores.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    scores : `numpy.ndarray`
        1-D array of scores (one for each cell). 
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Kendall's tau correlation coefficient between radial distance and score.
    """
    # Compute radial distance from biofilm center
    center = cells[:, :2].sum(axis=0) / cells.shape[0]
    distances = np.linalg.norm(cells[:, 2] - center, axis=1)

    return kendalltau(distances, scores).statistic

#######################################################################
def radial_group_distribution(cells, delta, nradii=100):
    """
    Given a biofilm of cells existing in two groups, get the distribution
    of the two groups as a function of radius from the biofilm's center.

    For each radius, this function locates the cells whose centers lie
    within an annulus with width `delta` and the given radius as its
    outer radius.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells. 
    delta : float
        The width of the annulus used for scanning the biofilm.
    nradii : int
        The number of different annuli for which to evaluate this 
        distribution.

    Returns
    -------
    The outer radius of each annulus and the corresponding fraction of
    cells within each annulus that belong to group 1. 
    """
    # Identify the biofilm center and the radial distance of the furthest
    # cell from the center
    center = cells[:, :2].sum(axis=0) / cells.shape[0]
    distances = np.linalg.norm(cells[:, :2] - center, axis=1)
    maxdist = distances.max()

    # Starting from an outer radius of delta, identify the cells whose
    # centers lie within an annulus of given outer radius and width delta
    radii = np.linspace(delta, maxdist + 1e-8, nradii)
    in_group_1 = np.zeros((nradii,), dtype=np.float64)
    for i, r in enumerate(radii):
        cells_within = (distances >= r - delta) & (distances < r)
        in_group_1[i] = (cells[cells_within, 9] == 1).sum() / cells_within.sum()

    return radii, in_group_1
    
