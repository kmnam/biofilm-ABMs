/**
 * Implementations of cell-cell and cell-surface interaction forces.
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
 *     6/28/2024
 */

#ifndef BIOFILM_MECHANICS_2D_HPP
#define BIOFILM_MECHANICS_2D_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "distances.hpp"
#include "kiharaGBK.hpp"

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * An enum that enumerates the different adhesion force types. 
 */
enum AdhesionMode
{
    NONE,
    KIHARA,
    GBK
}; 

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
Array<T, Dynamic, 2> compositeViscosityForcePrefactors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                       const T R,
                                                       const T surface_contact_density)
{
    Array<T, Dynamic, 2> KL(cells.rows(), 2);
    Array<T, Dynamic, 1> composite_drag = cells.col(8) + cells.col(9) * surface_contact_density / R; 
    KL.col(0) = cells.col(4) * composite_drag; 
    KL.col(1) = cells.col(4) * cells.col(4) * cells.col(4) * composite_drag / 12;

    return KL; 
}

/**
 * Get all pairs of cells whose distances are within the given threshold.
 *
 * The returned array is given in row-major format.  
 *
 * @param cells Existing population of cells.
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param R Cell radius. 
 * @param Ldiv Cell division length.
 * @returns Array of indices of pairs of neighboring cells.
 */
template <typename T>
Array<T, Dynamic, 6> getCellNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                      const T neighbor_threshold, const T R, const T Ldiv)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return an empty array
    if (n == 1)
        return Array<T, Dynamic, 6>::Zero(0, 6);

    // Generate Segment_3 instances for each cell 
    std::vector<Segment_3> segments = generateSegments(cells);

    // Instantiate kernel to be passed into distBetweenCells()
    K kernel;

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
    int npairs = n * (n - 1) / 2;
    Array<T, Dynamic, 6> neighbors(npairs, 6);
    int idx = 0; 

    // For each pair of cells in the population ... 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)   // Note that j < i
        {
            // For two cells to be within neighbor_threshold of each other,
            // their centers must be within neighbor_threshold + Ldiv + 2 * R
            T dist_rij = (cells(i, Eigen::seq(0, 1)) - cells(j, Eigen::seq(0, 1))).matrix().norm();  
            if (dist_rij < neighbor_threshold + Ldiv + 2 * R)
            {
                // In this case, compute their actual distance and check that 
                // they lie within neighbor_threshold of each other 
                auto result = distBetweenCells<T>(
                    segments[i], segments[j], cells(i, Eigen::seq(0, 1)).matrix(),
                    cells(i, Eigen::seq(2, 3)).matrix(), cells(i, 5),
                    cells(j, Eigen::seq(0, 1)).matrix(),
                    cells(j, Eigen::seq(2, 3)).matrix(), cells(j, 5), kernel
                );
                Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
                T si = std::get<1>(result);
                T sj = std::get<2>(result);
                if (dist_ij.norm() < neighbor_threshold)
                {
                    neighbors(idx, 0) = i; 
                    neighbors(idx, 1) = j; 
                    neighbors(idx, Eigen::seq(2, 3)) = dist_ij.array();
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
 * The given array of neighboring pairs of cells is updated in place.  
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array of neighboring pairs of cells. 
 */
template <typename T>
void updateNeighborDistances(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                             Ref<Array<T, Dynamic, 6> > neighbors)
{
    // Generate Segment_3 instances for each cell 
    std::vector<Segment_3> segments = generateSegments(cells);

    // Instantiate kernel to be passed into distBetweenCells()
    K kernel;

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
        auto result = distBetweenCells<T>(
            segments[i], segments[j], cells(i, Eigen::seq(0, 1)).matrix(),
            cells(i, Eigen::seq(2, 3)).matrix(), cells(i, 5),
            cells(j, Eigen::seq(0, 1)).matrix(),
            cells(j, Eigen::seq(2, 3)).matrix(), cells(j, 5), kernel
        ); 
        Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
        T si = std::get<1>(result);
        T sj = std::get<2>(result); 
        neighbors(k, Eigen::seq(2, 3)) = dist_ij.array(); 
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
 * @param prefactors Array of four pre-computed prefactors, namely `2.5 * sqrt(R)`,
 *                   `2.5 * E0 * sqrt(R)`, `E0 * pow(R - Rcell, 1.5)`, and `Ecell`.
 * @returns Derivatives of the cell-cell interaction energies with respect 
 *          to cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 4> cellCellRepulsiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                             const T R, const T Rcell,
                                             const Ref<const Array<T, 4, 1> >& prefactors)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4); 

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, x-orientation, y-orientation
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4);

    // Compute distance vector magnitude, direction, and corresponding
    // cell-cell overlap for every pair of neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array(); 
    Array<T, Dynamic, 2> directions = neighbors(Eigen::all, Eigen::seq(2, 3)).colwise() / magnitudes;
    Array<T, Dynamic, 1> overlaps = 2 * R - magnitudes;  

    // Note that:
    //     prefactors(0) = 2.5 * std::sqrt(R)
    //     prefactors(1) = 2.5 * E0 * std::sqrt(R)
    //     prefactors(2) = E0 * std::pow(R - Rcell, 1.5)
    //     prefactors(3) = Ecell

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        T si = neighbors(k, 4);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 5);                         // Cell-body coordinate along cell j
        Array<T, 2, 1> dir_ij = directions.row(k);      // Normalized distance vector 
        T overlap = overlaps(k);                        // Cell-cell overlap 

        // Define prefactors that determine the magnitudes of the interaction
        // forces, depending on the size of the overlap 
        //
        // Case 1: the overlap is positive but less than R - Rcell (i.e., it 
        // is limited to within the EPS coating)
        T prefactor = 0; 
        if (overlap > 0 && overlap < R - Rcell)
        {
            prefactor = prefactors(1) * std::pow(overlap, 1.5); 
        }
        // Case 2: the overlap is instead greater than R - Rcell (i.e., it 
        // encroaches into the bodies of the two cells)
        else if (overlap >= R - Rcell)
        {
            T term = prefactors(3) * std::pow(overlap - R + Rcell, 1.5);
            prefactor = prefactors(0) * (prefactors(2) + term);
        }

        if (overlap > 0)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 2, 1> vij = prefactor * dir_ij;
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
 * TODO Update
 *
 * Compute the derivatives of the cell-cell adhesion energies, modeled as 
 * Lennard-Jones attractive interactive energies, for each cell with respect
 * to the cell's position and orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param mode
 * @param params
 * @returns Derivatives of the cell-cell adhesion energies with respect to  
 *          cell positions and orientations.   
 */
template <typename T,
          typename PreciseType
              = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<30> > >
Array<T, Dynamic, 4> cellCellAdhesiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                            const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                            const T R, const T Rcell, 
                                            const AdhesionMode mode, 
                                            std::unordered_map<std::string, T>& params)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4); 

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, x-orientation, y-orientation
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4);

    // Compute distance vector magnitude, direction, and corresponding
    // cell-cell overlap for every pair of neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array(); 
    Array<T, Dynamic, 1> overlaps = 2 * R - magnitudes;  

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));

        // Check that the two cells adhere and their overlap is nonzero 
        if (to_adhere(k) && overlaps(k) > 0)
        {
            // Extract the cell position and orientation vectors 
            Matrix<T, 2, 1> ri = cells(i, Eigen::seq(0, 1)).matrix();
            Matrix<T, 2, 1> ni = cells(i, Eigen::seq(2, 3)).matrix();
            Matrix<T, 2, 1> rj = cells(j, Eigen::seq(0, 1)).matrix();
            Matrix<T, 2, 1> nj = cells(j, Eigen::seq(2, 3)).matrix();
            T half_li = cells(i, 5);
            T half_lj = cells(j, 5);
            Matrix<T, 2, 1> dij = neighbors(k, Eigen::seq(2, 3)).matrix();
            T si = neighbors(k, 4); 
            T sj = neighbors(k, 5); 

            // Get the corresponding forces
            Matrix<T, 2, 4> forces; 
            if (mode == KIHARA) 
            {
                const T strength = params["strength"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                forces = strength * forcesKihara2D<T>(
                    ri, ni, half_li, rj, nj, half_lj, R, dij, si, sj, expd,
                    dmin
                );
            }
            else if (mode == GBK)
            {
                const T strength = params["strength"];
                const T exp1 = params["anisotropy_exp1"];
                const T exp2 = params["anisotropy_exp2"];
                const T expd = params["distance_exp"]; 
                const T kappa0 = params["well_depth_delta"];
                const T dmin = params["mindist"];
                forces = strength * forcesGBK2D<T>(
                    ri, ni, half_li, rj, nj, half_lj, R, Rcell, dij, si, sj,
                    expd, exp1, exp2, kappa0, dmin
                ); 
            }
            dEdq.row(i) += forces.row(0).array(); 
            dEdq.row(j) += forces.row(1).array();
        }
    }

    return dEdq;
}

/**
 * TODO Update
 *
 * Given the current positions, orientations, lengths, viscosity coefficients,
 * and surface friction coefficients for the given population of cells, compute
 * their translational and orientational velocities.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param adhesion_mode
 * @param adhesion_params
 * @returns Array of translational and orientational velocities.   
 */
template <typename T,
          typename PreciseType
              = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<30> > >
Array<T, Dynamic, 4> getVelocities(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                   const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                   const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                   const T R, const T Rcell,
                                   const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                                   const T surface_contact_density,
                                   const AdhesionMode adhesion_mode,
                                   std::unordered_map<std::string, T>& adhesion_params)
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
    // lambda = 0.5 * (nx * dE/dnx + ny * dE/dny)
    //
    int n = cells.rows(); 
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 2> prefactors = compositeViscosityForcePrefactors<T>(
        cells, R, surface_contact_density
    );
    Array<T, Dynamic, 1> K = prefactors.col(0);
    Array<T, Dynamic, 1> L = prefactors.col(1);
    Array<T, Dynamic, 4> dEdq_repulsion = cellCellRepulsiveForces<T>(
        cells, neighbors, R, Rcell, cell_cell_prefactors
    );
    Array<T, Dynamic, 4> dEdq_adhesion = Array<T, Dynamic, 4>::Zero(n, 4); 
    if (adhesion_mode != NONE)
    {
        dEdq_adhesion = cellCellAdhesiveForces<T, PreciseType>(
            cells, neighbors, to_adhere, R, Rcell, adhesion_mode, adhesion_params
        );
    }
    Array<T, Dynamic, 4> dEdq = dEdq_repulsion + dEdq_adhesion; 

    // Set mult = 2 * lambda
    Array<T, Dynamic, 1> mult = cells.col(2) * dEdq.col(2) + cells.col(3) * dEdq.col(3);

    // Solve the Lagrangian equations of motion
    Array<T, Dynamic, 2> dEdn_constrained = (
        dEdq(Eigen::all, Eigen::seq(2, 3)) -
        cells(Eigen::all, Eigen::seq(2, 3)).colwise() * mult
    );
    velocities.col(0) = -dEdq.col(0) / K;
    velocities.col(1) = -dEdq.col(1) / K; 
    velocities.col(2) = -dEdn_constrained.col(0) / L;
    velocities.col(3) = -dEdn_constrained.col(1) / L;

    return velocities;  
}

/**
 * Normalize the orientation vectors of all cells in the given population.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells. 
 */
template <typename T>
void normalizeOrientations(Ref<Array<T, Dynamic, Dynamic> > cells)
{
    Array<T, Dynamic, 1> norms = cells(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array();
    assert((norms > 0).all() && "Zero norms encountered during orientation normalization");
    cells.col(2) /= norms; 
    cells.col(3) /= norms;
}

/**
 * TODO Update
 *
 * Run one step of an adaptive Runge-Kutta method with the given Butcher 
 * tableau for the given timestep.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
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
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero. 
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param adhesion_mode
 * @param adhesion_params
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T,
          typename PreciseType
              = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<30> > >
std::tuple<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 4>, Array<T, Dynamic, 4> >
    stepRungeKuttaAdaptive(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                           const Ref<const Array<T, Dynamic, 1> >& b,
                           const Ref<const Array<T, Dynamic, 1> >& bs, 
                           const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                           const Ref<const Array<T, Dynamic, 6> >& neighbors,
                           const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                           const T dt, const T R, const T Rcell,
                           const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                           const T surface_contact_density,
                           const AdhesionMode adhesion_mode, 
                           std::unordered_map<std::string, T>& adhesion_params)
{
    // Compute velocities at given partial timesteps 
    int n = cells.rows(); 
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 4> > velocities; 
    velocities.push_back(
        getVelocities<T, PreciseType>(
            cells, neighbors, to_adhere, R, Rcell, cell_cell_prefactors,
            surface_contact_density, adhesion_mode, adhesion_params
        )
    );
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 4> multipliers = Array<T, Dynamic, 4>::Zero(n, 4);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, Eigen::seq(0, 3)) += multipliers * dt;
        normalizeOrientations<T>(cells_i);    // Renormalize orientations after each modification
        velocities.push_back(
            getVelocities<T, PreciseType>(
                cells_i, neighbors, to_adhere, R, Rcell, cell_cell_prefactors,
                surface_contact_density, adhesion_mode, adhesion_params
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

    return std::make_tuple(cells_new, errors, velocities_final1); 
}

#endif
