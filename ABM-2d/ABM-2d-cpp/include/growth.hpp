/**
 * Functions for modeling cell growth and division. 
 * 
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 10+), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps) 
 * 5) timepoint at which the cell was formed
 * 6) cell growth rate
 * 7) cell's ambient viscosity with respect to surrounding fluid
 * 8) cell-surface friction coefficient
 * 9) cell group identifier (integer, optional)
 *
 * Additional features may be included in the array but these are not 
 * relevant for the computations implemented here
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/21/2023
 */

#include <cmath>
#include <Eigen/Dense>

using namespace Eigen; 

/**
 * Grow the cells in the given population according to the exponential 
 * volume growth law.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells. 
 * @param dt Timestep. 
 * @param R Cell radius.
 */
template <typename T>
void growCells(Ref<Array<T, Dynamic, Dynamic> > cells, const T dt, const T R)
{
    // Each cell grows in length according to an exponential growth law 
    cells.col(4) += (cells.col(6) * (4 * R / 3 + cells.col(4)) * dt).eval(); 
}

/**
 * Identify the cells that exceed the given division length. 
 *
 * @param cells Existing population of cells. 
 * @param Ldiv Cell division length.
 * @returns Boolean index indicating which cells are to divide. 
 */
template <typename T>
Array<int, Dynamic, 1> divideMaxLength(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                       const T Ldiv)
{
    return (cells.col(4) > Ldiv).template cast<int>(); 
}

/**
 * Divide the indicated cells at the given time. 
 *
 * If i is the index of a dividing cell, row i is updated to store one 
 * daughter cell, and a new row is appended to store the other daughter
 * cell.
 *
 * A newly allocated array of cell data is returned. 
 *
 * The growth rate of each daughter cell is chosen using the given
 * distribution function, which must take a random number generator
 * instance as its single input. 
 *
 * The function daughter_length_dist() controls the degree of asymmetry
 * in cell division: the daughter cells are determined to have lengths
 * M * (L - 2 * R) and (1 - M) * (L - 2 * R), where L is the length of
 * the dividing cell, R is the cell radius, and M is sampled using
 * daughter_length_dist(). 
 *
 * The function daughter_angle_dist() controls the orientations of the 
 * daughter cells: each daughter cell's orientation is obtained by rotating
 * the dividing cell's orientation by theta, which is sampled using 
 * daughter_angle_dist(). 
 *
 * @param cells Existing population of cells.
 * @param t Current time.
 * @param R Cell radius. 
 * @param to_divide Boolean index indicating which cells are to divide.
 * @param growth_dist Function instance specifying the growth rate
 *                    distribution. Must take boost::random::mt19937&
 *                    as its single argument.
 * @param rng Random number generator. 
 * @param daughter_length_dist Function instance specifying the daughter 
 *                             cell length ratio distribution. 
 * @param daughter_angle_dist Function instance specifying the daughter 
 *                            cell orientation distribution.
 * @returns Updated population of cells. 
 */
template <typename T>
Array<T, Dynamic, Dynamic> divideCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                       const T t, const T R,
                                       const Ref<const Array<int, Dynamic, 1> >& to_divide,
                                       std::function<T(boost::random::mt19937&)>& growth_dist,
                                       boost::random::mt19937& rng,
                                       std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                                       std::function<T(boost::random::mt19937&)>& daughter_angle_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();
    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        int n = cells.rows(); 
        std::vector<int> idx_divide;
        for (int i = 0; i < n; ++i)
        {
            if (to_divide(i))
                idx_divide.push_back(i);
        }

        // Get an extended copy of the existing population
        Array<T, Dynamic, Dynamic> cells_total(n + n_divide, cells.cols()); 
        cells_total(Eigen::seqN(0, n), Eigen::all) = cells; 

        // Get a copy of the corresponding sub-array
        Array<T, Dynamic, Dynamic> new_cells(cells(idx_divide, Eigen::all)); 

        // Update cell orientations ...
        //
        // Each daughter cell has an orientation that is obtained by 
        // rotating the dividing cell's orientation counterclockwise by
        // a angle theta, which is sampled using daughter_angle_dist()
        Array<T, Dynamic, 2> dividing_orientations(new_cells(Eigen::all, Eigen::seq(2, 3))); 
        Array<T, Dynamic, 1> theta1(n_divide), theta2(n_divide); 
        for (int i = 0; i < n_divide; ++i)
        {
            theta1(i) = daughter_angle_dist(rng); 
            theta2(i) = daughter_angle_dist(rng);
        }
        Array<T, Dynamic, 1> cos_theta1 = theta1.cos(); 
        Array<T, Dynamic, 1> sin_theta1 = theta1.sin(); 
        Array<T, Dynamic, 1> cos_theta2 = theta2.cos(); 
        Array<T, Dynamic, 1> sin_theta2 = theta2.sin(); 
        for (int i = 0; i < n_divide; ++i)
        {
            Matrix<T, 2, 2> rot(2, 2); 
            rot << cos_theta1(i), -sin_theta1(i),
                   sin_theta1(i),  cos_theta1(i); 
            cells_total(i, Eigen::seq(2, 3)) = (rot * dividing_orientations.matrix().row(i).transpose()).array();
            rot << cos_theta2(i), -sin_theta2(i),
                   sin_theta2(i),  cos_theta2(i); 
            new_cells(i, Eigen::seq(2, 3)) = (rot * dividing_orientations.matrix().row(i).transpose()).array();  
        }

        // Update cell lengths and positions ... 
        //
        // If cell division is symmetric, then the daughter cells both have
        // length L / 2 - R, where L is the length of the dividing cell, 
        // the point of division occurs at the center of the dividing cell,
        // and their x- and y-coordinates are perturbed by L / 4 + R / 2
        // along the daughter cells' orientation vectors 
        //
        // If not, then the daughter cells have lengths L1 = M * (L - 2 * R)
        // and L2 = (1 - M) * (L - 2 * R), where M is a random variable between
        // 0 and 1. The point of division occurs at cell body coordinate
        // L1 + R - Lh with Lh = L / 2, which has x- and y-coordinates given
        // by rx + (L1 + R - Lh) * nx and ry + (L1 + R - Lh) * ny, where
        // (rx, ry) and (nx, ny) are the position and orientation of the
        // dividing cell. The corresponding perturbations of the cell centers
        // from the point of division are given by R + M * (L - 2 * R) / 2 and
        // R + (1 - M) * (L - 2 * R) / 2
        //
        // Sample a normally distributed daughter cell length for each cell
        // to be divided 
        Array<T, Dynamic, 1> M(n_divide); 
        for (int i = 0; i < n_divide; ++i)
            M(i) = daughter_length_dist(rng);
        // Locate point of division along dividing cell centerline 
        Array<T, Dynamic, 1> div = (
            M * (cells_total(idx_divide, 4) - 2 * R) - (cells_total(idx_divide, 4)) / 2 + R
        );
        // Get perturbations from point of division along cell centerline
        // for the daughter cell centers
        Array<T, Dynamic, 1> delta1 = M * (cells_total(idx_divide, 4) - 2 * R) / 2 + R; 
        Array<T, Dynamic, 1> delta2 = (1 - M) * (new_cells.col(4) - 2 * R) / 2 + R;
        // Define daughter cell lengths and locate daughter cell centers
        cells_total(idx_divide, 4) = M * (cells_total(idx_divide, 4) - 2 * R); 
        new_cells.col(4) = (1 - M) * (new_cells.col(4) - 2 * R); 
        cells_total(idx_divide, 0) = (
            cells_total(idx_divide, 0) + (div - delta1) * cells_total(idx_divide, 2)
        );
        new_cells.col(0) = new_cells.col(0) + (div + delta2) * new_cells.col(2);
        cells_total(idx_divide, 1) = (
            cells_total(idx_divide, 1) + (div - delta1) * cells_total(idx_divide, 3)
        );
        new_cells.col(1) = new_cells.col(1) + (div + delta2) * new_cells.col(3);

        // Update cell birth times
        cells_total(idx_divide, 5) = t;
        new_cells.col(5) = t;

        // Update cell growth rates (sample from specified distribution)
        for (auto it = idx_divide.begin(); it != idx_divide.end(); ++it)
            cells_total(*it, 6) = growth_dist(rng); 
        for (int i = 0; i < n_divide; ++i)
            new_cells(i, 6) = growth_dist(rng);

        // Copy over daughter cell data
        //
        // Note that each daughter cell inherits its mother cell's viscosity
        // and friction coefficient
        cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;  
        
        return cells_total; 
    }
    else 
    {
        return cells; 
    }
} 
