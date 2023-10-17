"""
Authors:
    Kee-Myoung Nam

Last updated:
    10/16/2023
"""
import numpy as np

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
    distances = np.sqrt(
        np.power(cells[:, 0] - center[0], 2) + np.power(cells[:, 1] - center[1], 2)
    )

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

