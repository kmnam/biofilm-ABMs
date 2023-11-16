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
 *     11/16/2023
 */

#ifndef BIOFILM_CELL_CELL_POTENTIAL_FORCES_HPP
#define BIOFILM_CELL_CELL_POTENTIAL_FORCES_HPP

#include <cmath>
#include <Eigen/Dense>
#include "mechanics.hpp"

using namespace Eigen;

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
 * @param repulsive_only A vector that indicates, for each pair of neighboring
 *                       cells, whether to use only the repulsive part of the
 *                       potential.  
 */
template <typename T>
Array<T, Dynamic, 4> cellCellForcesKihara(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                          const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                          const T R, const T Rcell,
                                          const T prefactor_12,
                                          const T prefactor_6,
                                          const Ref<const Array<int, Dynamic, 1> >& repulsive_only)
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

    // Maximum cell-cell distance at which the force is nonzero
    T maxdist = 2 * (R - Rcell); 

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
        if (dist <= maxdist)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 2, 1> vij;
            if (!repulsive_only(k)) 
                vij = (prefactor_12 / std::pow(dist, 13) - prefactor_6 / std::pow(dist, 7)) * dir_ij;
            else
                vij = (prefactor_12 / std::pow(dist, 13)) * dir_ij;
            dEdq(i, Eigen::seq(0, 1)) -= vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell i
            dEdq(i, Eigen::seq(2, 3)) -= vij * si; 
            // Derivative of cell-cell interaction energy w.r.t position of cell j
            dEdq(j, Eigen::seq(0, 1)) += vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell j
            dEdq(j, Eigen::seq(2, 3)) += vij * sj; 
        } 
    }

    return dEdq;  
}

/**
 * Given the current positions, orientations, lengths, viscosity coefficients,
 * and surface friction coefficients for the given population of cells, compute
 * their translational and orientational velocities.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * This function uses forces calculated with the Kihara potential. 
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
 * @param surface_contact_density Cell-surface contact area density.
 * @param repulsive_only A vector that indicates, for each pair of neighboring
 *                       cells, whether to use only the repulsive part of the
 *                       potential.  
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 4> getVelocitiesKihara(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                         const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                         const T R, const T Rcell,
                                         const T prefactor_12,
                                         const T prefactor_6,
                                         const T surface_contact_density,
                                         const Ref<const Array<int, Dynamic, 1> >& repulsive_only)
{
    // For each cell, the relevant Lagrangian mechanics are given by 
    // 
    // dP/d(dq) = -dE/dq + lambda * d(nx^2 + ny^2 - 1)/dq, 
    //
    // where:
    // - P is the total dissipation due to bulk viscosity and surface friction,
    // - E is the total cell-cell interaction energy involving the given cell,
    // - q is a generalized coordinate (x-position, y-position, x-orientation, 
    //   y-orientation),
    // - nx and ny are the x-orientation and y-orientation, respectively,
    // - dq is the corresponding velocity, and 
    // - lambda is a Lagrange multiplier. 
    //
    // Note that dP/d(dq) = K * dq for some prefactor K, which is given in 
    // `composite_viscosity_force_prefactors()`. Moreover, the constraint 
    // 
    // nx^2 + ny^2 - 1 == 0
    // 
    // implies the constraint 
    //
    // 2 * nx * dnx + 2 * ny * dny == 0
    //
    // where dnx and dny are the orientational velocities. This yields the 
    // following value of the Lagrange multiplier:
    //
    // lambda = -0.5 * (nx * dE/dnx + ny * dE/dny)
    //
    int n = cells.rows(); 
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 2> prefactors = compositeViscosityForcePrefactors<T>(
        cells, R, surface_contact_density
    );
    Array<T, Dynamic, 1> K = prefactors.col(0);
    Array<T, Dynamic, 1> L = prefactors.col(1);
    assert((K != 0).all() && "Composite viscosity force prefactors for positions have zero values"); 
    assert((L != 0).all() && "Composite viscosity force prefactors for orientations have zero values");
    Array<T, Dynamic, 4> dEdq = cellCellForcesKihara<T>(
        cells, neighbors, R, Rcell, prefactor_12, prefactor_6, repulsive_only
    );
    Array<T, Dynamic, 1> mult = cells.col(2) * dEdq.col(2) + cells.col(3) * dEdq.col(3);
    Array<T, Dynamic, 2> dEdn_constrained = (
        dEdq(Eigen::all, Eigen::seq(2, 3)) +
        cells(Eigen::all, Eigen::seq(2, 3)).colwise() * mult
    );
    velocities.col(0) = -dEdq.col(0) / K;
    velocities.col(1) = -dEdq.col(1) / K; 
    velocities.col(2) = -dEdn_constrained.col(0) / L;
    velocities.col(3) = -dEdn_constrained.col(1) / L;

    return velocities;  
}

/**
 * Run one step of an adaptive Runge-Kutta method with the given Butcher 
 * tableau for the given timestep.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * This function uses forces (and velocities) calculated with the Kihara
 * potential. 
 *
 * Since the differential equations are time-autonomous, the nodes of the 
 * Butcher tableau are not required. 
 *
 * @param A Runge-Kutta matrix of Butcher tableau. Should be lower triangular
 *          with zero diagonal. 
 * @param b Weights of Butcher tableau. Entries should sum to one. 
 * @param bs Error weights of Butcher tableau. Entries should sum to one. 
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population. 
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param prefactor_12 The value `12 * eps0 * std::pow(dmin, 12)`, where
 *                     `eps0` is the strength parameter and `dmin` is the
 *                     cell-cell distance at which the Kihara potential 
 *                     is minimized.
 * @param prefactor_6 The value `12 * eps0 * std::pow(dmin, 6)`.
 * @param surface_contact_density Cell-surface contact area density.
 * @param repulsive_only A vector that indicates, for each pair of neighboring
 *                       cells, whether to use only the repulsive part of the
 *                       potential.  
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 4> >
    stepRungeKuttaAdaptiveKihara(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                                 const Ref<const Array<T, Dynamic, 1> >& b,
                                 const Ref<const Array<T, Dynamic, 1> >& bs, 
                                 const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                                 const Ref<const Array<T, Dynamic, 6> >& neighbors, 
                                 const T dt, const T R, const T Rcell,
                                 const T prefactor_12, const T prefactor_6,
                                 const T surface_contact_density,
                                 const Ref<const Array<int, Dynamic, 1> >& repulsive_only)
{
    // Compute velocities at given partial timesteps 
    int n = cells.rows(); 
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 4> > velocities; 
    velocities.push_back(
        getVelocitiesKihara<T>(
            cells, neighbors, R, Rcell, prefactor_12, prefactor_6,
            surface_contact_density, repulsive_only
        )
    );
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 4> multipliers = Array<T, Dynamic, 4>::Zero(n, 4);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, Eigen::seq(0, 3)) += multipliers * dt; 
        velocities.push_back(
            getVelocitiesKihara<T>(
                cells_i, neighbors, R, Rcell, prefactor_12, prefactor_6,
                surface_contact_density, repulsive_only
            )
        );
    }

    // Compute Runge-Kutta update from computed velocities
    Array<T, Dynamic, Dynamic> cells_new(cells); 
    Array<T, Dynamic, 4> velocities_final1 = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 4> velocities_final2 = Array<T, Dynamic, 4>::Zero(n, 4); 
    for (int i = 0; i < s; ++i)
    {
        velocities_final1 += velocities[i] * b(i); 
        velocities_final2 += velocities[i] * bs(i); 
    }
    Array<T, Dynamic, 4> delta1 = velocities_final1 * dt; 
    Array<T, Dynamic, 4> delta2 = velocities_final2 * dt; 
    cells_new(Eigen::all, Eigen::seq(0, 3)) += delta1;
    Array<T, Dynamic, 4> errors = delta1 - delta2; 
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells_new); 

    return std::make_pair(cells_new, errors); 
}

#endif
