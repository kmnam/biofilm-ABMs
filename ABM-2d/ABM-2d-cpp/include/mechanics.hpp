/**
 * Implementations of cell-cell and cell-surface interaction forces.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/21/2023
 */

#ifndef BIOFILM_MECHANICS_HPP
#define BIOFILM_MECHANICS_HPP

#include <vector>
#include <Eigen/Dense>
#include "cell.hpp"

using namespace Eigen; 

/**
 * Compute the derivatives of the dissipation due to bulk viscosity and 
 * surface friction for each cell with respect to the cell's translational
 * and orientational velocities, *divided by* the cell's translational and
 * orientational velocities. 
 *
 * Namely, if drx, dry, dnx, dny are the translational and orientational
 * velocities of cell i, then the derivative of the total dissipation, P,
 * due to bulk viscosity and surface friction with respect to these velocities
 * are given by 
 *
 * dP/d(drx) = K * drx
 * dP/d(dry) = K * dry
 * dP/d(dnx) = L * dnx
 * dP/d(dny) = L * dny
 *
 * where K and L are prefactors. This function returns these prefactors for
 * each cell in the given population.
 *
 * @param cells Existing population of cells. 
 * @param R Cell radius.
 * @param surface_contact_density Cell-surface contact area density.
 * @returns The prefactors K and L defined above for each cell. 
 */
template <typename T>
Matrix<T, Dynamic, 2> compositeViscosityForcePrefactors(const Ref<const Matrix<T, Dynamic, Dynamic> >& cells,
                                                        const T R,
                                                        const T surface_contact_density)
{
    Matrix<T, Dynamic, 2> KL(cells.rows(), 2);
    Matrix<T, Dynamic, 1> composite_drag = cells.col(7) + cells.col(8) * surface_contact_density / R; 
    KL.col(0) = cells.col(4) * composite_drag; 
    KL.col(1) = cells.col(4).array().pow(3) * composite_drag / 12; 

    return KL; 
}

/**
 * Get all pairs of cells whose distances are within the given threshold. 
 *
 * @param cells Existing population of cells.
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param R Cell radius. 
 * @param Ldiv Cell division length.
 * @returns Array of indices of pairs of neighboring cells.
 */
template <typename T>
Matrix<T, Dynamic, 6> getCellNeighbors(const Ref<const Matrix<T, Dynamic, Dynamic> >& cells,
                                       const T neighbor_threshold,
                                       const T R, const T Ldiv)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return an empty array
    if (n == 1)
        return Matrix<T, Dynamic, 6>::Zero(0, 6); 

    // Maintain array of neighboring cells 
    //
    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) Cell-body coordinate of contact point along centerline of cell i
    // 5) Cell-body coordinate of contact point along centerline of cell j
    Matrix<T, Dynamic, 6> neighbors(n * (n - 1) / 2, 6);
    int idx = 0; 

    // For each pair of cells in the population ... 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)   // Note that j < i
        {
            // For two cells to be within neighbor_threshold of each other,
            // their centers must be within neighbor_threshold + Ldiv + 2 * R
            T dist_rij = (cells(i, Eigen::seq(0, 1)) - cells(j, Eigen::seq(0, 1))).norm();  
            if (dist_rij < neighbor_threshold + Ldiv + 2 * R)
            {
                // In this case, compute their actual distance and check that 
                // they lie within neighbor_threshold of each other 
                auto result = distBetweenCells(
                    cells(i, Eigen::seq(0, 1)), cells(i, Eigen::seq(2, 3)), cells(i, 4), 
                    cells(j, Eigen::seq(0, 1)), cells(j, Eigen::seq(2, 3)), cells(j, 4)
                );
                Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
                T si = std::get<1>(result);
                T sj = std::get<2>(result);
                if (dist_ij.norm() < neighbor_threshold)
                {
                    neighbors(idx, 0) = i; 
                    neighbors(idx, 1) = j; 
                    neighbors(idx, Eigen::seq(2, 3)) = dist_ij;
                    neighbors(idx, 4) = si; 
                    neighbors(idx, 5) = sj;
                    idx++;  
                }
            }
        }
    }

    // Discard remaining rows and return
    return neighbors(Eigen::seq(0, idx - 1), Eigen::all);
}

/**
 * Update the cell-cell distances in the given array of neighboring pairs
 * of cells. 
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array of neighboring pairs of cells. 
 * @returns Updated array of neighboring pairs of cells.  
 */
template <typename T>
void updateNeighborDistances(const Ref<const Matrix<T, Dynamic, Dynamic> >& cells, 
                             Ref<Matrix<T, Dynamic, 6> > neighbors)
{
    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) Cell-body coordinate of contact point along centerline of cell i
    // 5) Cell-body coordinate of contact point along centerline of cell j
    //
    // Columns 2, 3, 4, 5 are updated here
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        auto result = distBetweenCells(
            cells(i, Eigen::seq(0, 1)), cells(i, Eigen::seq(2, 3)), cells(i, 4), 
            cells(j, Eigen::seq(0, 1)), cells(j, Eigen::seq(2, 3)), cells(j, 4)
        ); 
        Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
        T si = std::get<1>(result);
        T sj = std::get<2>(result); 
        neighbors(k, Eigen::seq(2, 3)) = dist_ij; 
        neighbors(k, 4) = si; 
        neighbors(k, 5) = sj;
    }
} 

/**
 * Compute the derivatives of the cell-cell interaction energies for each 
 * cell with respect to the cell's position and orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell.
 * @returns Derivatives of the cell-cell interaction energies with respect 
 *          to cell positions and orientations.   
 */
template <typename T>
Matrix<T, Dynamic, 4> cellCellForcesFromNeighbors(const Ref<const Matrix<T, Dynamic, Dynamic> >& cells,
                                                  const Ref<const Matrix<T, Dynamic, 6> >& neighbors,
                                                  const T R, const T Rcell,
                                                  const T E0, const T Ecell)
{
    int n = cells.size();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Matrix<T, Dynamic, 4>::Zero(n, 4); 

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, x-orientation, y-orientation
    Matrix<T, Dynamic, 4> dEdq = Matrix<T, Dynamic, 4>::Zero(n, 4); 

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        Matrix<T, 2, 1> dist_ij = neighbors(k, Eigen::seq(2, 3));   // Distance vector from i to j
        T si = neighbors(k, 4);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 5);                         // Cell-body coordinate along cell j
        T delta_ij = dist_ij.norm();                    // Magnitude of distance vector 
        Matrix<T, 2, 1> dir_ij = dist_ij / delta_ij;    // Normalized distance vector

        // Get the overlapping distance between cells i and j (this distance
        // is negative if the cells are not overlapping)
        T overlap = 2 * R - delta_ij; 

        // Define prefactors that determine the magnitudes of the interaction
        // forces, depending on the size of the overlap 
        //
        // Case 1: the overlap is positive but less than R - Rcell (i.e., it 
        // is limited to within the EPS coating)
        T prefactor;  
        if (overlap > 0 && overlap < R - Rcell)
        {
            prefactor = 2.5 * E0 * std::sqrt(R) * std::pow(overlap, 1.5); 
        }
        // Case 2: the overlap is instead greater than R - Rcell (i.e., it 
        // encroaches into the bodies of the two cells)
        else if (overlap >= R - Rcell)
        {
            T prefactor1 = E0 * std::pow(R - Rcell, 1.5);
            T prefactor2 = Ecell * std::pow(overlap - R + Rcell, 1.5);
            prefactor = 2.5 * std::sqrt(R) * (prefactor1 + prefactor2); 
        }

        if (overlap > 0)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Matrix<T, 2, 1> vij = prefactor * dir_ij; 
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
 * Given the current positions, orientations, lengths, viscosity coefficients,
 * and surface friction coefficients for the given population of cells,  
 */

#endif
