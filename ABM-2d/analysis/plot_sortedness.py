"""
Authors:
    Kee-Myoung Nam

Last updated:
    6/30/2025
"""
import glob
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_ind
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
import seaborn as sns
from utils import read_cells, parse_dir

#########################################################################
# Column indices in simulation data files 
__colidx_id = 0
__colseq_r = [1, 2]
__colidx_growth = 12
__colidx_group = 16
__ncols_required = 17

# Colors for plotting
color_mean = sns.color_palette()[0]
color_scatter = sns.color_palette('pastel')[0]

#########################################################################
def adjust_axes_size(ax, w, h):
    """
    Adjust the given axes to have the given width and height. 

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Input axes. 
    w : float
        Width (in inches). 
    h : float
        Height (in inches).
    """
    fig_width = ax.figure.get_size_inches()[0]
    fig_height = ax.figure.get_size_inches()[1]
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    ax_width = fig_width * (r - l)      # Note that r - l is a fraction of figure width
    ax_height = fig_height * (t - b)    # Note that t - b is a fraction of figure height
    #print(w, h, r - l, t - b)
    ax.figure.set_size_inches(w / (r - l), h / (t - b))
    #print(ax.figure.get_size_inches())

#########################################################################
def setup_fig(ax_width, ax_height, margins):
    """
    """
    # Extrapolate the figure dimensions and instantiate 
    left = margins[0]
    right = 1 - margins[1]
    bottom = margins[2]
    top = 1 - margins[3]
    fig_width = ax_width / (right - left)
    fig_height = ax_height / (top - bottom) 
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Position the axes with the figure
    ax = fig.add_axes([left, bottom, right - left, top - bottom])

    return fig, ax

#########################################################################
def radial_spearman_coeff(cells, scores):
    """
    Compute the Spearman correlation between the radial positions of the 
    cells (with respect to the center of mass) and the given array of 
    "scores," which may represent any cell-related quantity. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Cell population data. 
    scores : `numpy.ndarray`
        Cell scores. 

    Returns
    -------
    Spearman correlation coefficient, along with a normalized coefficient
    obtained by dividing the Spearman coefficient by its theoretical maximum.
    """
    # Get the radial position of each cell 
    center = cells[:, __colseq_r].mean(axis=0)
    distances = np.linalg.norm(cells[:, __colseq_r] - center, axis=1)

    # Get the Spearman correlation and its normalization 
    rho = spearmanr(distances, scores).statistic
    scores_sorted = np.sort(scores)
    norm = spearmanr(np.arange(cells.shape[0]), scores_sorted).statistic

    # Negate so that the resulting values are positive 
    return -rho, -rho / norm

#########################################################################
def plot_velocities_at_timepoint(cells_t1, cells_t2, dt, ax, rmin, rmax, 
                                 **violinplot_kwargs):
    """
    Plot the radial velocity distributions of group-1 and group-2 cells
    at the two timepoints.

    This function assumes that there are only two groups. 

    Each of the two input data arrays, `cells_t1` and `cells_t2`, are *lists*
    of arrays, each of which pertains to a different simulation; the frames
    should be matched so that `cells_t1[i]` and `cells_t2[i]` refer to frames
    from the same simulation at the two timepoints. 

    Parameters
    ----------
    cells_t1 : list of `numpy.ndarray`
        Cell population data at timepoint 1 pertaining to one or more
        simulations. 
    cells_t2 : list of `numpy.ndarray`
        Cell population data at timepoint 2 pertaining to one or more
        simulations.
    dt : float
        Difference between timepoints 1 and 2 (in minutes). Assumed to be
        the same for every simulation. 
    ax : `matplotlib.pyplot.Axes`
        Input axes. 
    rmin : float
        Minimum radial position.
    rmax : float
        Maximum radial position. 
    violinplot_kwargs : dict
        Optional arguments to pass to `sns.violinplot()`.
    """
    df = pd.DataFrame()

    # For each simulation ... 
    nsim = len(cells_t1)
    if nsim != len(cells_t2):
        raise RuntimeError(
            'Two input arrays pertain to different numbers of simulations'
        )
    for i in range(nsim):
        # Identify the cells that are common to the two frames from the
        # i-th simulation 
        cells_t1_ids = set(cells_t1[i][:, __colidx_id])
        cells_t2_ids = set(cells_t2[i][:, __colidx_id])
        intersect = cells_t1_ids.intersection(cells_t2_ids)
        cells_t1_idx = [(j in intersect) for j in cells_t1[i][:, __colidx_id]]
        cells_t2_idx = [(j in intersect) for j in cells_t2[i][:, __colidx_id]]
        cells_t1_i = cells_t1[i][cells_t1_idx, :]
        cells_t2_i = cells_t2[i][cells_t2_idx, :]

        # Get the radial position of each cell in the first frame 
        origin = cells_t1_i[:, __colseq_r].mean(axis=0)
        rdist = np.linalg.norm(cells_t1_i[:, __colseq_r] - origin, axis=1)
        rfrac = rdist / np.max(rdist)

        # Filter out all cells such that:
        # - their radial positions fall outside the desired range
        # - they have switched groups within the timestep
        # - they are non-growing
        growing = (cells_t1_i[:, __colidx_growth] > 0)
        switched = (cells_t1_i[:, __colidx_group] != cells_t2_i[:, __colidx_group])
        cells_t1_i = cells_t1_i[
            (growing & ~switched & (rfrac >= rmin[i]) & (rfrac <= rmax[i])), :
        ]
        cells_t2_i = cells_t2_i[
            (growing & ~switched & (rfrac >= rmin[i]) & (rfrac <= rmax[i])), :
        ]

        # Get the radial direction of each cell from the origin 
        rdir = cells_t1_i[:, __colseq_r] - origin
        rdir /= np.linalg.norm(rdir, axis=1).reshape(-1, 1)

        # Get the velocities of the cells (in microns per minute)
        dr = (cells_t2_i[:, __colseq_r] - cells_t1_i[:, __colseq_r]) / dt

        # Project the velocities along the radial direction 
        rvel = np.zeros((dr.shape[0],), dtype=np.float64)
        for j in range(dr.shape[0]):
            rvel[j] = np.dot(dr[j, :], rdir[j, :])

        # Re-organize and append onto the DataFrame 
        df = pd.concat((
            df,
            pd.DataFrame({
                'sim': i,
                'rvel': rvel,
                'group': cells_t1_i[:, __colidx_group]
            })
        ))

    print(df.groupby(['sim', 'group']).mean())

    # Plot each violinplot
    sns.violinplot(
        data=df, y='sim', x='rvel', hue='group', ax=ax,
        orient='h',
        split=True,
        inner=None,
        density_norm='area',
        linecolor='black',
        palette={1: colormaps['managua'](255), 2: sns.color_palette()[3]},
        **violinplot_kwargs
    )

    # For each (half-)violin, mark the mean with a vertical line
    patches = ax.get_children()[:4]
    for i, patch in enumerate(patches):
        sim = i // 2
        group = (i % 2) + 1
        vertices = patch.get_paths()[0].vertices
        vertices = vertices[np.abs(vertices[:, 1] - sim) > 1e-8, :]
        mean = np.mean(df.loc[((df['sim'] == sim) & (df['group'] == group)), 'rvel'])
        mean_vertex = vertices[np.argmin(np.abs(vertices[:, 0] - mean)), :]
        ymax = mean_vertex[1]
        ax.plot(
            [mean, mean],
            [sim, sim + 0.98 * (ymax - sim)],
            color='black',
            **violinplot_kwargs
        )

    return ax, df

#########################################################################
if __name__ == '__main__':
    # ----------------------------------------------------------------- #
    # Plot radial velocity distributions from simulations with and without
    # the growth void
    # ----------------------------------------------------------------- #
    width = 305.0 * 7.24 / 1257.0
    height = 181.0 * 7.24 / 1257.0
    fig, ax = setup_fig(width, height, (0.15, 0.01, 0.25, 0.1))
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(
        axis='both', which='major', labelsize=10, direction='out'
    )

    # For each set of simulations, get the penultimate and final frames, using
    # a time difference of 3 min
    dt = 3.0
    idx1 = -7    # Frame 3 min before the penultimate frame, assuming dt_write = 0.01 hr = 36 sec
    idx2 = -2    # Penultimate frame
    cells_void1 = np.zeros((0, __ncols_required), dtype=np.float64)
    cells_void2 = np.zeros((0, __ncols_required), dtype=np.float64)
    cells_novoid1 = np.zeros((0, __ncols_required), dtype=np.float64)
    cells_novoid2 = np.zeros((0, __ncols_required), dtype=np.float64)
    for i in range(20):
        # Get the filenames for each pair of simulations, with and without
        # the growth void
        filenames_void = parse_dir(
            'frames/cells_5000/'
            'dual_friction_20fold_void_annulus_0.5_start100_switch_8_8_{}/*.txt'.format(i)
        )
        filenames_novoid = parse_dir(
            'frames/cells_5000/'
            'dual_friction_20fold_switch_8_8_{}/*.txt'.format(i)
        )
        cells_void1_, params_void1 = read_cells(filenames_void[idx1])
        cells_void2_, params_void2 = read_cells(filenames_void[idx2])
        assert cells_void1_.shape[0] <= cells_void2_.shape[0]
        assert np.isclose(params_void2['t_curr'], params_void1['t_curr'] + dt / 60.0, rtol=0, atol=1e-5)
        cells_novoid1_, params_novoid1 = read_cells(filenames_novoid[idx1])
        cells_novoid2_, params_novoid2 = read_cells(filenames_novoid[idx2])
        assert cells_novoid1_.shape[0] <= cells_novoid2_.shape[0]
        assert np.isclose(params_novoid2['t_curr'],  params_novoid1['t_curr'] + dt / 60.0, rtol=0, atol=1e-5)

        # Edit the cell IDs so that they do not overlap among simulations
        # without the growth void ... 
        cells_void1_[:, __colidx_id] += i * 10000
        cells_void2_[:, __colidx_id] += i * 10000

        # ... or among simulations with the growth void 
        cells_novoid1_[:, __colidx_id] += i * 10000
        cells_novoid2_[:, __colidx_id] += i * 10000

        # Collect the cells in the two frames  
        cells_void1 = np.vstack((cells_void1, cells_void1_[:, :__ncols_required]))
        cells_void2 = np.vstack((cells_void2, cells_void2_[:, :__ncols_required]))
        cells_novoid1 = np.vstack((cells_novoid1, cells_novoid1_))
        cells_novoid2 = np.vstack((cells_novoid2, cells_novoid2_))

    assert cells_void1.shape[0] <= cells_void2.shape[0]
    assert cells_novoid1.shape[0] <= cells_novoid2.shape[0]
    assert len(set(cells_void1[:, __colidx_id])) == cells_void1.shape[0]
    assert len(set(cells_void2[:, __colidx_id])) == cells_void2.shape[0]
    assert len(set(cells_novoid1[:, __colidx_id])) == cells_novoid1.shape[0]
    assert len(set(cells_novoid2[:, __colidx_id])) == cells_novoid2.shape[0]

    # Plot the radial velocity distributions 
    _, df = plot_velocities_at_timepoint(
        [cells_novoid1, cells_void1],
        [cells_novoid2, cells_void2],
        dt, ax, [0.05, 0.55], [0.25, 0.75], linewidth=0.5
    )

    # Determine whether differences in mean radial velocity between the 
    # void and no-void simulations are statistically significant 
    rvel_novoid_group1 = df.loc[((df['sim'] == 0) & (df['group'] == 1)), 'rvel']
    rvel_void_group1 = df.loc[((df['sim'] == 1) & (df['group'] == 1)), 'rvel']
    rvel_novoid_group2 = df.loc[((df['sim'] == 0) & (df['group'] == 2)), 'rvel']
    rvel_void_group2 = df.loc[((df['sim'] == 1) & (df['group'] == 2)), 'rvel']
    print(ttest_ind(rvel_novoid_group1, rvel_void_group1, equal_var=False))
    print(ttest_ind(rvel_novoid_group2, rvel_void_group2, equal_var=False))

    # Add a legend
    ax.legend(
        handles=[
            Patch(
                facecolor=colormaps['managua'](255),
                edgecolor='black', linewidth=0.5, label='High'
            ),
            Patch(
                facecolor=sns.color_palette()[3],
                edgecolor='black', linewidth=0.5, label='Low'
            )
        ],
        loc='center right',
        handlelength=1,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.4,
        ncols=1,
        fontsize=10
    )

    # Configure axes labels, size, etc.
    ax.set_xlabel(r'Radial velocity ($\mu$m/min)', size=10, labelpad=2)
    ax.set_ylabel('')
    dx = 0.3
    xmin, xmax = ax.get_xlim()
    if xmin < 0:
        xmin = xmin + (np.abs(xmin) % dx)
    else:
        xmin = xmin + (xmin % dx)
    xmax = xmax - (xmax % dx)
    ax.set_xticks(np.arange(xmin, xmax + dx, dx))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No void', 'Void'], rotation=90, va='center', size=10)
    #adjust_axes_size(ax, 1.2, 0.666)
    #plt.tight_layout(pad=0.1)
    plt.savefig(
        'radial_velocities_void_vs_novoid_combined.pdf',
        transparent=True
    )

    # ----------------------------------------------------------------- #
    # Plot sortedness values for simulations with differential growth
    # ----------------------------------------------------------------- #
    sortedness = {}
    growth_rate_folds = ['1.1', '1.2', '1.5', '2']
    switch_rates = ['2', '4', '8']
    sortedness['growth'] = {
        (fold, rate): [] for fold in growth_rate_folds for rate in switch_rates
    }

    # For each growth rate ratio and switching rate ... 
    for fold in growth_rate_folds:
        for rate in switch_rates:
            # ... parse the final frames of the corresponding group of simulations
            # and compute their sortedness values 
            filenames = glob.glob(
                'frames_final/cells_5000/'
                'dual_growth_{}fold_switch_{}_{}_*_final.txt'.format(fold, rate, rate)
            )
            assert len(filenames) == 20
            for filename in filenames:
                cells, _ = read_cells(filename)
                scores = 2 - cells[:, __colidx_group]
                _, rho = radial_spearman_coeff(cells, scores)
                sortedness['growth'][(fold, rate)].append(rho)

    # Test for significant difference from zero
    for fold in growth_rate_folds:
        for rate in switch_rates:
            print(fold, rate, ttest_1samp(sortedness['growth'][(fold, rate)], 0))

    # Plot sortedness values 
    width = 0.8245
    height = 0.9157
    fig = plt.figure(figsize=(1.1 * width, 1.1 * height))
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(
        axis='both', which='major', labelsize=6, direction='in', length=2,
        width=0.5
    )
    for i, fold in enumerate(growth_rate_folds):
        ax = sns.swarmplot(
            x=[i for _ in sortedness['growth'][(fold, '8')]],
            y=sortedness['growth'][(fold, '8')],
            zorder=0,
            color=color_scatter,
            s=2
        )
        ax.errorbar(
            x=i,
            y=np.mean(sortedness['growth'][(fold, '8')]),
            yerr=[np.std(sortedness['growth'][(fold, '8')])],
            fmt='_', markersize=10, markeredgecolor='black',
            markerfacecolor='black', capsize=5, capthick=0.5, elinewidth=0.5,
            color='black', zorder=1
        )

    # Plot a horizontal line to indicate zero sortedness 
    ax.plot(
        [-0.5, 0.5 + len(growth_rate_folds) - 1], [0, 0],
        color='red', linestyle='--', linewidth=0.8, zorder=0
    )

    # Configure axes labels, etc.
    ax.set_xticks(range(len(growth_rate_folds)))
    ax.set_xticklabels(
        ['{}'.format(fold) for fold in growth_rate_folds]
    )
    ax.set_xticklabels(
        ['{}'.format(fold) for fold in growth_rate_folds]
    )
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim([-0.2, ax.get_ylim()[1]])
    ax.set_xlabel('High-/low-c-di-GMP growth rate ratio', size=6)
    ax.set_ylabel('Sortedness', size=6)
    plt.tight_layout()
    adjust_axes_size(ax, width, height)
    plt.savefig('sortedness_growth_paper.pdf', transparent=True)

    # ----------------------------------------------------------------- #
    # Plot sortedness values for simulations with differential cell-
    # surface friction
    # ----------------------------------------------------------------- #
    friction_folds = ['1', '2', '5', '10', '20', '50', '100']
    sortedness['friction'] = {
        (fold, rate): [] for fold in friction_folds for rate in switch_rates
    }

    # For each friction coefficient ratio and switching rate ... 
    for fold in friction_folds:
        for rate in switch_rates:
            # ... parse the final frames of the corresponding group of simulations
            # and compute their sortedness values 
            filenames = glob.glob(
                'frames_final/cells_5000/'
                'dual_friction_{}fold_switch_{}_{}_*_final.txt'.format(fold, rate, rate)
            )
            assert len(filenames) == 20
            for filename in filenames:
                cells, _ = read_cells(filename)
                scores = 2 - cells[:, __colidx_group]
                _, rho = radial_spearman_coeff(cells, scores)
                sortedness['friction'][(fold, rate)].append(rho)

    # Test for significant difference from zero 
    for fold in friction_folds:
        for rate in switch_rates:
            print(fold, rate, ttest_1samp(sortedness['friction'][(fold, rate)], 0))

    # Plot sortedness values 
    width = 1.2672
    height = 0.9157
    fig = plt.figure(figsize=(1.1 * width, 1.1 * height))
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=2, width=0.5)
    for i, fold in enumerate(friction_folds):
        ax = sns.swarmplot(
            x=[i for _ in sortedness['friction'][(fold, '8')]],
            y=sortedness['friction'][(fold, '8')],
            zorder=0,
            color=color_scatter,
            s=1.6
        )
        ax.errorbar(
            x=i,
            y=np.mean(sortedness['friction'][(fold, '8')]),
            yerr=[np.std(sortedness['friction'][(fold, '8')])],
            fmt='_', markersize=10, markeredgecolor='black',
            markerfacecolor='black', capsize=5, capthick=0.5, elinewidth=0.5,
            color='black', zorder=1
        )

    # Plot a horizontal line to indicate zero sortedness 
    ax.plot(
        [-0.4, len(friction_folds) - 1 + 0.4], [0, 0], color='red', linestyle='--',
        linewidth=0.8, zorder=0
    )

    # Configure axes labels, etc. 
    ax.set_xlim([-0.6, len(friction_folds) - 1 + 0.6])
    ax.set_xticks(range(len(friction_folds)))
    ax.set_xticklabels(['{}'.format(fold) for fold in friction_folds])
    ax.set_xlabel('High-/low-c-di-GMP friction coefficient ratio', size=6)
    ax.set_ylabel('Sortedness', size=6)
    plt.tight_layout()
    adjust_axes_size(ax, width, height)
    plt.savefig('sortedness_friction_paper.pdf', transparent=True)

    # ----------------------------------------------------------------- #
    # Plot sortedness values for simulations with differential growth,
    # for different combinations of switching rates 
    # ----------------------------------------------------------------- #
    switch_pairs = list(itertools.product(switch_rates, switch_rates))
    sortedness['growth_1.2fold_switch'] = {pair: [] for pair in switch_pairs}

    # For each pair of switching rates (fixing growth rate ratio to 1.2) ... 
    for pair in switch_pairs:
        # ... parse the final frames of the corresponding group of simulations
        # and compute their sortedness values 
        filenames = glob.glob(
            'frames_final/cells_5000/'
            'dual_growth_1.2fold_switch_{}_{}_*_final.txt'.format(pair[0], pair[1])
        )
        assert len(filenames) == 20
        for filename in filenames:
            cells, _ = read_cells(filename)
            scores = 2 - cells[:, __colidx_group]
            _, rho = radial_spearman_coeff(cells, scores)
            sortedness['growth_1.2fold_switch'][pair].append(rho)

    # Test for significant difference from zero
    for pair in switch_pairs:
        print(1.2, pair[0], pair[1], ttest_1samp(sortedness['growth_1.2fold_switch'][pair], 0))

    # Plot sortedness values in two plots:
    #
    # 1) Plot sortedness values for different growth rate ratios and
    #    switching rates, with switching rates fixed to the same value 
    # 2) Plot sortedness values for growth rate ratio = 1.2 and different
    #    combinations of switching rates
    #
    # Start with sortedness values for different growth rate ratios and
    # symmetric switching rates 
    width1 = 505.0 * 7.24 / 1257.0
    height1 = 151.0 * 7.24 / 1257.0
    fig1, ax1 = setup_fig(width1, height1, (0.2, 0.01, 0.35, 0.05))
    idx = 0
    xlabels = [(fold, rate) for fold in growth_rate_folds for rate in switch_rates]
    for fold, rate in xlabels:
        ax1 = sns.swarmplot(
            x=[idx for _ in sortedness['growth'][(fold, rate)]],
            y=sortedness['growth'][(fold, rate)],
            s=1.5,
            zorder=1,
            color=color_scatter,
            ax=ax1
        )
        ax1.errorbar(
            x=idx,
            y=np.mean(sortedness['growth'][(fold, rate)]),
            yerr=[np.std(sortedness['growth'][(fold, rate)])],
            fmt='_', markersize=15, markeredgecolor='black',
            markerfacecolor='black', capsize=5, capthick=1.0,
            elinewidth=1.0, color='black', zorder=2
        )
        idx += 1

    # Then plot sortedness values for different pairs of switching rates
    # and with growth rate ratio = 1.2
    width2 = 505.0 * 7.24 / 1257.0
    height2 = 139.0 * 7.24 / 1257.0
    fig2, ax2 = setup_fig(width2, height2, (0.2, 0.01, 0.35, 0.05))
    for j, pair in enumerate(switch_pairs):
        ax2 = sns.swarmplot(
            x=[j for _ in sortedness['growth_1.2fold_switch'][pair]],
            y=sortedness['growth_1.2fold_switch'][pair],
            s=1.5,
            zorder=1,
            color=color_scatter,
            ax=ax2
        )
        ax2.errorbar(
            x=j,
            y=np.mean(sortedness['growth_1.2fold_switch'][pair]),
            yerr=[np.std(sortedness['growth_1.2fold_switch'][pair])],
            fmt='_', markersize=15, markeredgecolor='black',
            markerfacecolor='black', capsize=5, capthick=1.0, elinewidth=1.0,
            color='black', zorder=2
        )

    # Plot horizontal lines for zero sortedness
    ax1.plot(
        [-0.4, len(growth_rate_folds) * len(switch_rates) - 1 + 0.4],
        [0, 0], color='red', linestyle='--', linewidth=1, zorder=0
    )
    ax2.plot(
        [-0.4, len(switch_pairs) - 1 + 0.4], [0, 0],
        color='red', linestyle='--', linewidth=1, zorder=0
    )

    # Configure labels and annotate for ax1
    ax1.set_ylabel('Sortedness', size=10)
    ax1.set_xticks(range(len(growth_rate_folds) * len(switch_rates)))
    ax1.set_xticklabels([rate for fold, rate in xlabels], size=10)
    ax1.set_xlabel(r'$\tau_{\text{H}} = \tau_{\text{L}}$', size=10)
    for i, fold in enumerate(growth_rate_folds):
        ax1.annotate(
            r'$\gamma_{{\text{{L}}}} / \gamma_{{\text{{H}}}} = {}$'.format(fold),
            xy=(i * 3 + 1, 0.19),
            xycoords='data',
            horizontalalignment='center',
            size=9
        )
        ax1.plot([i * 3 - 0.4, i * 3 + 2.4], [0.17, 0.17], color='black')
    ax1.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax1.set_xlim([-0.6, len(growth_rate_folds) * len(switch_rates) - 1 + 0.6])
    ax1.set_ylim([-0.2, 0.25])

    # Configure labels and annotate for axes[1]
    ax2.set_ylabel('Sortedness', size=10)
    ax2.set_xticks(range(len(switch_pairs)))
    ax2.annotate(
        r'$\gamma_{\text{L}} / \gamma_{\text{H}} = 1.2$',
        xy=(0.01, 0.95),
        size=9,
        xycoords='axes fraction',
        horizontalalignment='left',
        verticalalignment='top'
    )
    ax2.set_xticklabels(
        [
            (
                r'$\tau_{{\text{{H}}}} = {}$,'.format(pair[0]) + '\n' +
                r'$\tau_{{\text{{L}}}} = {}$'.format(pair[1])
            )
            for pair in switch_pairs
        ],
        size=7
    )
    ax2.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax2.set_xlim([-0.6, len(switch_pairs) - 1 + 0.6])
    ax2.set_ylim([-0.2, 0.25])
    fig1.savefig('sortedness_growth_switch_tLtH.pdf', transparent=True)
    fig2.savefig('sortedness_growth_switch_g1.2.pdf', transparent=True)

    # ----------------------------------------------------------------- #
    # Plot sortedness values for simulations with differential cell-
    # surface friction, for different choices of switching rates 
    # ----------------------------------------------------------------- #
    width = 1044.0 * 7.24 / 1257.0
    height = 240.0 * 7.24 / 1257.0
    fig, ax = setup_fig(width, height, (0.2, 0.01, 0.35, 0.05))
    idx = 0

    # Plot sortedness values for different friction coefficient ratios 
    # and different switching rates, assuming symmetric switching 
    xlabels = [(fold, rate) for fold in friction_folds for rate in switch_rates]
    maxval = -np.inf
    for fold, rate in xlabels:
        ax = sns.swarmplot(
            x=[idx for _ in sortedness['friction'][(fold, rate)]],
            y=sortedness['friction'][(fold, rate)],
            s=1.5,
            zorder=1,
            color=color_scatter,
            ax=ax,
        )
        ax.errorbar(
            x=idx,
            y=np.mean(sortedness['friction'][(fold, rate)]),
            yerr=[np.std(sortedness['friction'][(fold, rate)])],
            fmt='_', markersize=15, markeredgecolor='black',
            markerfacecolor='black', capsize=5, capthick=1.0, elinewidth=1.0,
            color='black', zorder=2
        )
        if max(sortedness['friction'][(fold, rate)]) > maxval:
            maxval = max(sortedness['friction'][(fold, rate)])
        idx += 1

    # Plot horizontal line to indicate zero sortedness
    ax.plot(
        [-0.4, len(friction_folds) * len(switch_rates) - 1 + 0.4],
        [0, 0], color='red', linestyle='--', linewidth=1, zorder=0
    )

    # Configure labels and annotate for ax
    ax.set_xlabel(r'$\tau_{\text{H}} = \tau_{\text{L}}$', size=10)
    ax.set_ylabel('Sortedness', size=10)
    ax.set_xticks(range(len(friction_folds) * len(switch_rates)))
    ax.set_xticklabels([rate for fold, rate in xlabels], size=10)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    line_ycoord = maxval + 0.01
    text_ycoord = line_ycoord + 0.01
    for i, fold in enumerate(friction_folds):
        ax.annotate(
            r'$\eta_{{1,\text{{H}}}} / \eta_{{1,\text{{L}}}} = {}$'.format(fold),
            xy=(i * 3 + 1, text_ycoord),
            xycoords='data',
            horizontalalignment='center',
            verticalalignment='bottom',
            size=8
        )
        ax.plot([i * 3 - 0.4, i * 3 + 2.4], [line_ycoord, line_ycoord], color='black')
    ax.set_xlim([-0.6, len(friction_folds) * len(switch_rates) - 1 + 0.6])
    ax.set_ylim([ax.get_ylim()[0], text_ycoord + 0.1])
    
    plt.savefig('sortedness_friction_switch.pdf', transparent=True)

