/**
 * Implementations of potential-based cell-cell interaction forces.
 *
 * In what follows, a population of N cells is represented as a 2-D array of
 * size (N, 10+), where each row represents a cell and stores the following 
 * data: 
 *
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps)
 * 5) half of cell length (excluding caps)
 * 6) timepoint at which cell was formed
 * 7) cell growth rate
 * 8) cell's ambient viscosity with respect to surrounding fluid
 * 9) cell-surface friction coefficient
 *
 * Additional features may be included in the array but these are not
 * relevant for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     11/10/2023
 */

#ifndef BIOFILM_CELL_CELL_POTENTIAL_FORCES_HPP
#define BIOFILM_CELL_CELL_POTENTIAL_FORCES_HPP

#include <assert>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

/**
 * Compute the derivatives of the repulsive component of the Kihara cell-cell
 * potential for each cell with respect to the cell's position and orientation
 * coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed.
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population.
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param prefactor_12 The value `12 * eps0 * std::pow(dmin, 12)`, where
 *                     `eps0` is the strength parameter and `dmin` is the
 *                     cell-cell distance at which the Kihara potential 
 *                     is minimized (in the presence of an attractive 
 *                     component).
 */
template <typename T>
Array<T, Dynamic, 4> cellCellForcesKihara(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                          const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                          const T R, const T Rcell,
                                          const T prefactor_12)
{
    int n = cells.rows();    // Number of cells
    
    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4);

    // Maintain array of partial derivatives of the interaction energies
    // with respect to x-position, y-position, x-orientation, y-orientation
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4);

    // Compute distance vector magnitude and direction for every pair of
    // neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array();
    Array<T, Dynamic, 2> directions = neighbors(Eigen::all, Eigen::seq(2, 3)).colwise() / magnitudes;
    
    // Compute cell-cell distances between the cells themselves (excluding EPS)
    Array<T, Dynamic, 1> distances = magnitudes - 2 * Rcell;

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        T si = neighbors(k, 4);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 5);                         // Cell-body coordinate along cell j
        Array<T, 2, 1> dir_ij = directions.row(k);      // Normalized distance vector 
        T dist = distances(k);                          // Cell-cell distance

        // If the distance between cells is at most 2 * R ... 
        if (dist <= 2 * R)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 2, 1> vij = (-prefactor_12 / std::pow(dist, 13)) * dir_ij;
            dEdq(i, Eigen::seq(0, 1)) += vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell i
            dEdq(i, Eigen::seq(2, 3)) += vij * si; 
            // Derivative of cell-cell interaction energy w.r.t position of cell j
            dEdq(j, Eigen::seq(0, 1)) -= vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell j
            dEdq(j, Eigen::seq(2, 3)) -= vij * sj; 
        } 
    }

    return dEdq;  
}

/**
 * Compute the derivatives of the Kihara cell-cell potential for each 
 * cell with respect to the cell's position and orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population.
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param prefactor_12 The value `12 * eps0 * std::pow(dmin, 12)`, where
 *                     `eps0` is the strength parameter and `dmin` is the
 *                     cell-cell distance at which the Kihara potential 
 *                     is minimized.
 * @param prefactor_6 The value `12 * eps0 * std::pow(dmin, 6)`.
 */
template <typename T>
Array<T, Dynamic, 4> cellCellForcesKihara(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                          const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                          const T R, const T Rcell,
                                          const T prefactor_12,
                                          const T prefactor_6)
{
    int n = cells.rows();    // Number of cells
    
    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4);

    // Maintain array of partial derivatives of the interaction energies
    // with respect to x-position, y-position, x-orientation, y-orientation
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4);

    // Compute distance vector magnitude and direction for every pair of
    // neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array();
    Array<T, Dynamic, 2> directions = neighbors(Eigen::all, Eigen::seq(2, 3)).colwise() / magnitudes;
    
    // Compute cell-cell distances between the cells themselves (excluding EPS)
    Array<T, Dynamic, 1> distances = magnitudes - 2 * Rcell;

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        T si = neighbors(k, 4);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 5);                         // Cell-body coordinate along cell j
        Array<T, 2, 1> dir_ij = directions.row(k);      // Normalized distance vector 
        T dist = distances(k);                          // Cell-cell distance

        // If the distance between cells is at most 2 * R ... 
        if (dist <= 2 * R)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 2, 1> vij = (-prefactor_12 / std::pow(dist, 13) + prefactor_6 / std::pow(dist, 7)) * dir_ij;
            dEdq(i, Eigen::seq(0, 1)) += vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell i
            dEdq(i, Eigen::seq(2, 3)) += vij * si; 
            // Derivative of cell-cell interaction energy w.r.t position of cell j
            dEdq(j, Eigen::seq(0, 1)) -= vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell j
            dEdq(j, Eigen::seq(2, 3)) -= vij * sj; 
        } 
    }

    return dEdq;  
}

#endif
