"""
Authors:
    Kee-Myoung Nam

Last updated:
    2/19/2025
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2
import seaborn as sns
import networkx as nx
from utils import read_cells, parse_dir

__colidx_rx = 1
__colidx_ry = 2
__colidx_rz = 3
__colseq_r = [1, 2, 3]
__colidx_group = 21

##########################################################################
def read_graph(filename):
    """
    Read the given file, specifying a pre-computed cell-cell neighbor graph.

    This function outputs five objects:
    - the graph itself, as a `networkx.Graph` object 
    - the array of vertex degrees 
    - the array of component indices for each vertex 
    - the array of component sizes 
    - the array of triangles in the graph
    - the degree distribution 
    """
    graph = nx.Graph()
    degrees = []
    components = []
    component_sizes = []
    triangles = []
    degree_dist = {}
    cluster_coefs = []
    with open(filename) as f:
        for line in f:
            if line.startswith('DEGREE_DIST'):
                _, d, n = line.split()
                degree_dist[int(d)] = int(n)
            elif line.startswith('VERTEX'):
                args = line.split()
                graph.add_node(int(args[1]))
                for arg in args[2:]:
                    if arg.startswith('COMPONENT:'):
                        components.append(int(arg[10:]))
                    elif arg.startswith('DEGREE:'):
                        degrees.append(int(arg[7:]))
                    else:    # arg.startswith('CLUSTER:')
                        x = float(arg[8:])
                        if x == -1:
                            cluster_coefs.append(0.0)
                        else:
                            cluster_coefs.append(x)
            elif line.startswith('EDGE'):
                _, v, w, _ = line.split()
                graph.add_edge(int(v), int(w))
            elif line.startswith('COMPONENT'):
                component_sizes.append(int(line.split()[2]))
            elif line.startswith('TRIANGLE'):
                triangles.append([int(x) for x in line.split()[1:]])
    degree_dist = np.array(
        [degree_dist[d] for d in sorted(degree_dist.keys())], dtype=np.float64
    )
    degree_dist /= degree_dist.sum()

    return (
        graph, np.array(degrees), np.array(components), np.array(component_sizes),
        np.array(triangles), degree_dist, np.array(cluster_coefs)
    )

##########################################################################
def plot_graph_3d(cells, graph, components, component_sizes, ax):
    """
    Plot the given cell-cell neighbor graph, with each vertex represented 
    by its center in three dimensions. 
    """
    n = cells.shape[0]
    n_groups = int(np.max(cells[:, __colidx_group]))
    color_main = sns.color_palette()[0]
    colors_misc = [
        sns.color_palette('pastel')[0],
        sns.color_palette()[3]
    ]

    # Plot the vertices, color-coded by group
    #
    # First plot the group 1 vertices in the main component
    component_main = np.argmax(component_sizes)
    idx_main = np.where(components == component_main)[0]
    idx_misc = np.where(components != component_main)[0]
    coords_main = cells[np.ix_(idx_main, __colseq_r)]
    ax.scatter(
        coords_main[:, 0], coords_main[:, 1], coords_main[:, 2], marker='.',
        s=10, color=color_main
    )

    # Then plot the remaining cells
    cells_misc = cells[idx_misc, :]
    for i in range(n_groups):
        idx = np.where(cells_misc[:, __colidx_group] == i + 1)[0]
        coords = cells_misc[np.ix_(idx, __colseq_r)]
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], marker='.', s=10,
            color=colors_misc[i]
        )

    # Plot the edges 
    for i, j in graph.edges:
        ax.plot(
            [cells[i, __colidx_rx], cells[j, __colidx_rx]],
            [cells[i, __colidx_ry], cells[j, __colidx_ry]],
            [cells[i, __colidx_rz], cells[j, __colidx_rz]],
            c=(color_main if i in idx_main else colors_misc[0])
        )

    return ax

##########################################################################
if __name__ == '__main__':
    cells_filenames = parse_dir(os.path.join(sys.argv[1], '*'))
    graph_filenames = [
        os.path.join(
            os.path.dirname(cells_filename),
            'graphs',
            os.path.basename(cells_filename)[:-4] + '_graph.txt'
        )
        for cells_filename in cells_filenames
    ]
    plot_filenames = []

    # Set axes limits for the cell-cell neighbor graph 
    cells, params = read_cells(cells_filenames[-1])
    xmin = cells[:, __colidx_rx].min() * 1.05
    xmax = cells[:, __colidx_rx].max() * 1.05
    ymin = cells[:, __colidx_ry].min() * 1.05
    ymax = cells[:, __colidx_ry].max() * 1.05
    zmin = cells[:, __colidx_rz].min() * 1.05
    zmax = cells[:, __colidx_rz].max() * 1.05
    n_total = cells.shape[0]
    t_final = params['t_curr']

    # Maintain arrays for:
    # - the fraction of group 1 cells at each timepoint
    # - the largest connected component size at each timepoint 
    # - the number of cells at each timepoint
    # - the number of triangles at each timepoint
    # - the timepoints themselves
    fraction_group1 = []
    fraction_max_components = []
    avg_degree = []
    avg_degree_group1 = []
    n_cells = []
    avg_cluster_group1 = []
    n_triangles = []
    timepoints = []

    # For each frame ... 
    for cells_filename, graph_filename in zip(cells_filenames, graph_filenames):
        # Only plot if there are > 10 cells
        cells, params = read_cells(cells_filename)
        if cells.shape[0] > 10:
            print('Plotting: {} ...'.format(cells_filename))
            (
                graph, degrees, components, component_sizes, triangles,
                degree_dist, cluster_coefs
            ) = read_graph(graph_filename)

            # Get the fraction of cells in group 1
            n_cells.append(cells.shape[0])
            timepoints.append(params['t_curr'])
            in_group1 = (cells[:, __colidx_group] == 1)
            n_group1 = np.sum(in_group1)
            fraction_group1.append(n_group1 / cells.shape[0])

            # Get the average degree of all cells  
            avg_degree.append(np.sum([i * d for i, d in enumerate(degree_dist)]))

            # Get the average degree of all group 1 cells
            degrees_group1 = degrees[in_group1]
            avg_degree_group1.append(np.mean(degrees_group1))

            # Get the average local clustering coefficient of all group 1 cells
            avg_cluster_group1.append(np.mean(cluster_coefs[in_group1]))

            # Get the number of triangles
            n_triangles.append(triangles.shape[0])

            # Get the fraction of group 1 cells that fall within each of the 
            # top three components 
            top_three = sorted(component_sizes, reverse=True)[:3]
            if top_three[0] == n_group1:    # All group 1 cells in one component
                top_three[1] = 0
                top_three[2] = 0
            elif top_three[0] + top_three[1] == n_group1:   # All group 1 cells in top two components
                top_three[2] = 0
            fraction_max_components.append([x / n_group1 for x in top_three])
            
            # Set up five plots 
            fig = plt.figure(figsize=(10, 7))
            gs = GridSpec(4, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 1])
            ax5 = fig.add_subplot(gs[3, 1])

            # Draw the cell-cell neighbor graph in the left axes
            plot_graph_3d(cells, graph, components, component_sizes, ax1)
            ax1.set_xlim([xmin, xmax])
            ax1.set_ylim([ymin, ymax])
            ax1.set_zlim([zmin, zmax])
            ax1.set_aspect('equal')

            # Annotate with number of cells and timepoint
            ax1.set_title('{} cells'.format(cells.shape[0]))

            # Plot the top 3 largest connected component sizes over time 
            for i in range(3):
                ax2.plot(
                    n_cells, [x[i] for x in fraction_max_components], 
                    color=sns.color_palette('pastel')[7], linewidth=2, zorder=0
                )
                ax2.scatter(
                    n_cells[:-1], [x[i] for x in fraction_max_components[:-1]], 
                    color=sns.color_palette('pastel')[7], s=10, zorder=0
                )
            crosses = ax2.scatter(
                [n_cells[-1], n_cells[-1], n_cells[-1]],
                fraction_max_components[-1],
                c=sns.color_palette()[:3], marker='X', s=20, zorder=1
            )
            ax2.set_xlim([10, n_total * 1.1])
            ax2.set_ylim([0.0, 1.0])
            ax2.set_xscale('log')
            #ax2.set_xlabel('Number of cells')
            ax2.set_ylabel('High-c-di-GMP fraction\nin top 3 components')
            ax2.legend(
                handles=[
                    Patch(
                        facecolor=sns.color_palette()[0],
                        edgecolor=sns.color_palette()[0], label='1st'
                    ),
                    Patch(
                        facecolor=sns.color_palette()[1],
                        edgecolor=sns.color_palette()[1], label='2nd'
                    ),
                    Patch(
                        facecolor=sns.color_palette()[2],
                        edgecolor=sns.color_palette()[2], label='3rd'
                    )
                ],
                loc='upper left',
                fontsize='x-small',
            )

            # Plot the average degree of all cells and of the group 1 cells
            # over time 
            ax3.plot(
                n_cells, avg_degree, c=sns.color_palette('pastel')[7],
                linewidth=2, zorder=0
            )
            ax3.scatter(
                n_cells[:-1], avg_degree[:-1], color=sns.color_palette('pastel')[7],
                s=10, zorder=0
            )
            ax3.scatter(
                [n_cells[-1]], [avg_degree[-1]], color=sns.color_palette()[4],
                marker='X', s=20, zorder=1
            )
            ax3.plot(
                n_cells, avg_degree_group1, c=sns.color_palette('pastel')[7],
                linewidth=2, zorder=0
            )
            ax3.scatter(
                n_cells[:-1], avg_degree_group1[:-1], color=sns.color_palette('pastel')[7],
                s=10, zorder=0
            )
            ax3.scatter(
                [n_cells[-1]], [avg_degree_group1[-1]], color=sns.color_palette()[0],
                marker='X', s=20, zorder=1
            )
            ax3.set_xlim([10, n_total * 1.1])
            ax3.set_xscale('log')
            #ax3.set_xlabel('Number of cells')
            ax3.set_ylabel('Average degree')
            ax3.legend(
                handles=[
                    Patch(
                        facecolor=sns.color_palette()[4],
                        edgecolor=sns.color_palette()[4], label='All cells'
                    ),
                    Patch(
                        facecolor=sns.color_palette()[0],
                        edgecolor=sns.color_palette()[0], label='High c-di-GMP'
                    ),
                ],
                loc='upper left',
                fontsize='x-small',
            )

            # Plot the average local clustering coefficient over all group 1 
            # cells (with the coefficient set to 0 for degree-0 or degree-1
            # cells)
            ax4.plot(
                n_cells, avg_cluster_group1, c=sns.color_palette('pastel')[7],
                linewidth=2, zorder=0
            )
            ax4.scatter(
                n_cells[:-1], avg_cluster_group1[:-1], color=sns.color_palette('pastel')[7],
                s=10, zorder=0
            )
            ax4.scatter(
                [n_cells[-1]], [avg_cluster_group1[-1]], color=sns.color_palette()[0],
                marker='X', s=20, zorder=1
            )
            ax4.set_xlim([10, n_total * 1.1])
            ax4.set_xscale('log')
            #ax4.set_xlabel('Number of cells')
            ax4.set_ylabel('Average LCC')

            # Plot the number of triangles in each graph as a function of 
            # the number of cells
            ax5.plot(
                n_cells, n_triangles, c=sns.color_palette('pastel')[7],
                linewidth=2, zorder=0
            )
            ax5.scatter(
                n_cells[:-1], n_triangles[:-1], color=sns.color_palette('pastel')[7],
                s=10, zorder=0
            )
            ax5.scatter(
                [n_cells[-1]], [n_triangles[-1]], color=sns.color_palette()[0],
                marker='X', s=20, zorder=1
            )
            ax5.set_xlim([10, n_total * 1.1])
            ax5.set_xscale('log')
            ax5.set_xlabel('Number of cells')
            ax5.set_ylabel('Number of triangles')

            # Save to file 
            plot_filename = graph_filename[:-4] + '_plot.jpg'
            plot_filenames.append(plot_filename)
            plt.tight_layout()
            plt.savefig(plot_filename, transparent=True, dpi=300)
            plt.close(fig)

    # Stitch together frames into a video
    width = None
    height = None
    with Image.open(plot_filenames[0]) as image:
        width, height = image.size
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    outfilename = os.path.join(
        os.path.dirname(plot_filenames[0]),
        os.path.basename(cells_filenames[-1])[:-4] + '_combined.avi'
    )
    video = cv2.VideoWriter(
        outfilename, fourcc, 20, (width, height), isColor=True
    )
    for plot_filename in plot_filenames:
        with Image.open(plot_filename) as image:
            video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

