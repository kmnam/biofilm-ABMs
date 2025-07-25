"""
Authors:
    Kee-Myoung Nam

Last updated:
    3/30/2025
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from utils import read_cells

#########################################################################
_colidx_rz = 3
_colseq_rxy = [1, 2]
_colidx_group = 21

#########################################################################
def compute_sortedness(cells):
    """
    """
    # Get radial distances and directions 
    center = cells[:, _colseq_rxy].mean(axis=0)
    rdists = np.linalg.norm(cells[:, _colseq_rxy] - center, axis=1)
    rmax = np.max(rdists)
    rdirs = (cells[:, _colseq_rxy] - center) / rdists.reshape(-1, 1)

    # For each cell ...
    rdists_scaled = np.zeros((cells.shape[0],), dtype=np.float64)
    for i in range(cells.shape[0]):
        # Find the cells that have nearly the same radial direction 
        nearly_parallel = (rdirs.dot(rdirs[i].reshape(-1, 1)) > 0.999)
        idx_parallel = np.where(nearly_parallel)[0]

        # Get the cell with the maximal radial distance
        max_idx = idx_parallel[0]
        for j in range(1, len(idx_parallel)):
            if rdists[idx_parallel[j]] > rdists[max_idx]:
                max_idx = idx_parallel[j]

        # Transform the cell's radial distance
        rdists_scaled[i] = rdists[i] * (rmax / rdists[max_idx])
    
    # Compute the normalized Spearman correlation
    print('p =', spearmanr(rdists_scaled, cells[:, _colidx_group]).pvalue)
    corr = spearmanr(rdists_scaled, cells[:, _colidx_group]).statistic
    norm = -spearmanr(np.arange(cells.shape[0]), np.sort(cells[:, _colidx_group])).statistic
    print(norm)
    
    return corr / norm

#########################################################################
def plot_histogram(cells, ax, rmax=None, zmax=None, binsize=1.0, no_surface=False):
    """
    """
    # Determine bins in the radial and z-directions 
    center = cells[:, _colseq_rxy].mean(axis=0)
    rdists = np.linalg.norm(cells[:, _colseq_rxy] - center, axis=1)
    if rmax is None or rmax <= rdists.max():
        rmax = np.ceil(rdists.max())
    if zmax is None or zmax <= cells[:, _colidx_rz].max():
        zmax = np.ceil(np.abs(cells[:, _colidx_rz]).max())
    rbins = np.arange(0, rmax + binsize, binsize)
    zbins = np.arange(0, zmax + binsize, binsize)
    frac1 = (cells[:, _colidx_group] == 1).sum() / cells.shape[0]

    # Assign each cell to a block in the histogram
    hist_total = np.zeros((rbins.shape[0], zbins.shape[0]), dtype=np.float64)
    hist_group1 = np.zeros((rbins.shape[0], zbins.shape[0]), dtype=np.float64)
    for i in range(cells.shape[0]):
        rbin = np.where(rbins <= rdists[i])[0][-1]
        zbin = np.where(zbins <= np.abs(cells[i, _colidx_rz]))[0][-1]
        hist_total[rbin, zbin] += 1
        if cells[i, _colidx_group] == 1:
            hist_group1[rbin, zbin] += 1

    # Define colormap, which should be neutral at 1
    def cmap_func(x):
        # x should range between 0 and 1 / frac1
        cmap = colormaps['coolwarm_r']
        if x <= 1.0:
            return cmap(x / 2.0)
        else:
            return cmap(0.5 + (x - 1) / ((2.0 / frac1) - 2))
    cmap_modified = LinearSegmentedColormap.from_list(
        'coolwarm_r_modified',
        [cmap_func(x) for x in np.linspace(0, 1.0 / frac1, 256)]
    )

    # Get the proportion of group 1 cells in each block in the histogram 
    hist = (hist_group1 / hist_total) / frac1
    mappable = ax.imshow(
        hist.T, cmap=cmap_modified, vmin=0, vmax=(1./frac1), origin='lower'
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.1)
    cbar = plt.colorbar(mappable, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.tick_params(axis='x', direction='in')
    cbar.set_label(r'$c_{\mathrm{H}} / c_{\mathrm{H}}^*$', size=16)
    cbar.set_ticks(np.linspace(0, 1. / frac1, 9))
    cbar.set_ticklabels(
        ['{:.2f}'.format(x) for x in np.linspace(0, 1. / frac1, 9)], size=12
    )

    # Update the histogram axes labels, by shifting the tick positions
    # down by 0.5 while keeping their values
    xticks = np.arange(0, 5 * np.ceil((rmax + binsize) / 5), 5)
    yticks = np.arange(0, 5 * np.ceil((zmax + binsize) / 5), 5)
    ax.set_xticks(xticks - 0.5)
    ax.set_yticks(yticks - 0.5)
    ax.set_xticklabels([str(int(x)) for x in xticks], size=16)
    ax.set_yticklabels([str(int(y)) for y in yticks], size=16)
    ax.set_xlabel(r'$r$', size=16)
    ax.set_ylabel(r'$z$', size=16)

    return ax

#########################################################################
def plot_histogram_avg(filenames, ax, rmax=None, zmax=None, binsize=1.0,
                       no_surface=False):
    """
    """
    # For each file ...
    cells = []
    rmax_per_file = []
    zmax_per_file = []
    sortedness = []
    for filename in filenames:
        # Parse the cells 
        cells_, _ = read_cells(filename)
        sortedness.append(compute_sortedness(cells_))
        cells.append(cells_)
        print(
            'Parsed {}: {} in group 1 ({:.8f}%), {} in group 2 ({:.8f}%)'.format(
                filename,
                (cells_[:, _colidx_group] == 1).sum(),
                100 * (cells_[:, _colidx_group] == 1).sum() / cells_.shape[0],
                (cells_[:, _colidx_group] == 2).sum(),
                100 * (cells_[:, _colidx_group] == 2).sum() / cells_.shape[0]
            )
        )

        # Get the maximum radial distance and z-coordinate
        center = cells_[:, _colseq_rxy].mean(axis=0)
        rdists = np.linalg.norm(cells_[:, _colseq_rxy] - center, axis=1)
        rmax_per_file.append(rdists.max())
        zmax_per_file.append(np.abs(cells_[:, _colidx_rz]).max())

    # Rescale the cell coordinates by the maximum such distance/coordinate 
    rmax_overall = max(rmax_per_file)
    zmax_overall = max(zmax_per_file)
    for i in range(len(cells)):
        cells[i][:, _colseq_rxy] *= (rmax_overall / rmax_per_file[i])
        cells[i][:, _colidx_rz] *= (zmax_overall / zmax_per_file[i])
    cells = np.vstack(cells)

    # Plot all the cells
    plot_histogram(cells, ax, rmax=rmax, zmax=zmax, binsize=binsize)
    #ax.annotate(
    #    r'$\overline{{\rho}} = {:.6f}$'.format(np.mean(sortedness)),
    #    xy=(0.99, 0.99),
    #    xycoords='axes fraction',
    #    size=16,
    #    horizontalalignment='right',
    #    verticalalignment='top'
    #)

    return ax

#########################################################################
if __name__ == '__main__':
    fmtstrs = [
        'frames_final/dual_adhesion_JKR_400nm_switch_4_2_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_switch_4_4_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_switch_4_8_{}_final.txt',
        'frames_final/dual_growth_1.2fold_adhesion_JKR_400nm_switch_4_4_{}_final.txt',
        'frames_final/dual_growth_1.5fold_adhesion_JKR_400nm_switch_4_4_{}_final.txt',
        'frames_final/dual_growth_2fold_adhesion_JKR_400nm_switch_4_4_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_switch_4_4_start16_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_switch_4_4_start32_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_switch_4_4_start64_{}_final.txt',
        'frames_final/dual_adhesion_JKR_400nm_nosurface_switch_4_4_{}_final.txt'
    ]

    for fmtstr in fmtstrs:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        filenames = [fmtstr.format(i) for i in range(20)]
        outfilename = os.path.join(
            'histograms',
            '_'.join(os.path.basename(fmtstr).split('_')[:-2]) + '.pdf'
        )
        print('Plotting:', outfilename)
        plot_histogram_avg(filenames, ax)
        plt.savefig(outfilename)

