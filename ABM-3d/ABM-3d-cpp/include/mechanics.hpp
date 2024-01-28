/**
 * Implementations of cell-cell and cell-surface interaction forces.
 *
 * In what follows, a population of N cells is represented as a 2-D array of
 * size (N, 12+), where each row represents a cell and stores the following 
 * data: 
 *
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) z-coordinate of cell center
 * 3) x-coordinate of cell orientation vector
 * 4) y-coordinate of cell orientation vector
 * 5) z-coordinate of cell orientation vector
 * 6) cell length (excluding caps)
 * 7) half of cell length (excluding caps)
 * 8) timepoint at which cell was formed
 * 9) cell growth rate
 * 10) cell's ambient viscosity with respect to surrounding fluid
 * 11) cell-surface friction coefficient
 *
 * Additional features may be included in the array but these are not
 * relevant for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/27/2024
 */

#ifndef BIOFILM_MECHANICS_3D_HPP
#define BIOFILM_MECHANICS_3D_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <boost/math/constants/constants.hpp>
#include "integrals.hpp"
#include "distances.hpp"

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * Compute the derivatives of the cell-surface repulsion energy for each cell
 * with respect to the cell's z-position and z-orientation.
 *
 * @param cells Existing population of cells.
 * @param ss Cell-body coordinates at which each cell-surface overlap is zero. 
 * @param R Cell radius.
 * @param E0 Elastic modulus of EPS. 
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero. 
 */
template <typename T>
Array<T, Dynamic, 2> cellSurfaceRepulsionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                const Ref<const Array<T, Dynamic, 1> >& ss,
                                                const T R, const T E0,
                                                const T nz_threshold)
{
    Array<T, Dynamic, 1> abs_nz = cells(Eigen::all, 5).abs();
    Array<T, Dynamic, 2> dEdq = Array<T, Dynamic, 2>::Zero(cells.rows(), 2); 

    // For each cell ...
    const T prefactor0 = 2 * E0;
    const T prefactor1 = (8. / 3.) * E0 * std::pow(R, 0.5); 
    const T prefactor2 = 2 * E0 * std::pow(R, 0.5);
    for (int i = 0; i < cells.rows(); ++i)
    {
        // If the z-coordinate of the cell's orientation is zero ... 
        if (abs_nz(i) < nz_threshold)
        {
            T phi = R - cells(i, 2);
            // dEdq(i, 0) is nonzero if phi > 0
            if (phi > 0)
                dEdq(i, 0) = -prefactor0 * phi * cells(i, 6);
            // dEdq(i, 1) is zero 
        }
        // Otherwise ...
        else
        {
            // Compute the derivative of the cell-surface repulsion energy 
            // with respect to z-position
            T nz2 = cells(i, 5) * cells(i, 5);
            T int1 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.0, ss(i));
            T int2 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            dEdq(i, 0) = -prefactor0 * ((1 - nz2) * int1 + std::pow(R, 0.5) * nz2 * int2);

            // Compute the derivative of the cell-surface repulsion energy 
            // with respect to z-orientation
            T int3 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 2.0, ss(i));
            T int4 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.0, ss(i));
            T int5 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.5, ss(i));
            T int6 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            //dEdq(i, 1) -= prefactor0 * cells(i, 5) * int3;
            //dEdq(i, 1) -= prefactor0 * (1 - nz2) * int4; 
            //dEdq(i, 1) += prefactor1 * cells(i, 5) * int5;
            //dEdq(i, 1) -= prefactor2 * nz2 * int6;
            dEdq(i, 1) -= prefactor0 * (-cells(i, 5)) * int3;
            dEdq(i, 1) -= prefactor0 * (1 - nz2) * int4;
            dEdq(i, 1) += prefactor1 * (-cells(i, 5)) * int5;
            dEdq(i, 1) -= prefactor2 * nz2 * int6; 
        }
    }

    return dEdq;
}

/**
 * Compute the derivatives of the cell-surface adhesion energy for each cell 
 * with respect to the cell's z-position and z-orientation. 
 *
 * @param cells Existing population of cells.
 * @param ss Cell-body coordinates at which each cell-surface overlap is zero. 
 * @param R Cell radius.
 * @param adhesion_energy_density Cell-surface adhesion energy density.
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero. 
 */
template <typename T>
Array<T, Dynamic, 2> cellSurfaceAdhesionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const Ref<const Array<T, Dynamic, 1> >& ss,
                                               const T R,
                                               const T adhesion_energy_density, 
                                               const T nz_threshold)
{
    Array<T, Dynamic, 1> abs_nz = cells(Eigen::all, 5).abs(); 
    Array<T, Dynamic, 2> dEdq = Array<T, Dynamic, 2>::Zero(cells.rows(), 2);

    // For each cell ...
    const T prefactor0 = adhesion_energy_density * std::pow(R, 0.5) / 2;
    const T prefactor1 = 2 * adhesion_energy_density * boost::math::constants::pi<T>() * R;
    const T prefactor2 = 2 * adhesion_energy_density * std::pow(R, 0.5);
    for (int i = 0; i < cells.rows(); ++i)
    {
        // If the z-coordinate of the cell's orientation is zero ... 
        if (abs_nz(i) < nz_threshold)
        {
            T phi = R - cells(i, 2);
            // dEdq(i, 0) is nonzero if phi > 0
            if (phi > 0)
                dEdq(i, 0) = prefactor0 * cells(i, 6) / std::pow(phi, 0.5);
            // dEdq(i, 1) is zero 
        }
        // Otherwise ... 
        else
        {
            // Compute the derivative of the cell-surface adhesion energy 
            // with respect to z-position
            T nz2 = cells(i, 5) * cells(i, 5);
            T int1 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), -0.5, ss(i));
            T term2 = 0;
            if (ss(i) >= -cells(i, 7) && ss(i) < cells(i, 7)) 
                term2 = (prefactor1 / 2) * cells(i, 5);
            dEdq(i, 0) = prefactor0 * (1 - nz2) * int1 - term2;

            // Compute the derivative of the cell-surface adhesion energy
            // with respect to z-orientation
            T int2 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            T int3 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), -0.5, ss(i));
            T int4 = integral4(cells(i, 2), cells(i, 5), R, cells(i, 7), ss(i));
            T term4 = 0;
            if (ss(i) >= -cells(i, 7) && ss(i) < cells(i, 7))
                term4 = (prefactor1 / 2) * (R - cells(i, 2));
            //dEdq(i, 1) += prefactor2 * cells(i, 5) * int2;
            //dEdq(i, 1) += prefactor0 * (1 - nz2) * int3;
            //dEdq(i, 1) -= prefactor1 * cells(i, 5) * int4;
            //dEdq(i, 1) -= term4;
            dEdq(i, 1) += prefactor2 * (-cells(i, 5)) * int2; 
            dEdq(i, 1) += prefactor0 * (1 - nz2) * int3;
            dEdq(i, 1) -= prefactor1 * (-cells(i, 5)) * int4;
            dEdq(i, 1) -= term4;
        }
    }

    return dEdq;
}

/**
 * Compute the derivatives of the dissipation due to bulk viscosity and
 * surface friction for the given cell with respect to the cell's
 * translational and orientational velocities.
 *
 * Namely, if drx, dry, drz, dnx, dny, dnz are the translational and 
 * orientational velocities of cell i, then the derivative of the total 
 * dissipation, P, with respect to these velocities can be encoded as a
 * 6x6 matrix, M, where
 *
 * [ dP/d(drx) ]       [ drx ]
 * [ dP/d(dry) ]       [ dry ]
 * [ dP/d(drz) ] = M * [ drz ]
 * [ dP/d(dnx) ]       [ dnx ]
 * [ dP/d(dny) ]       [ dny ]
 * [ dP/d(dnz) ]       [ dnz ]
 *
 * This function returns the matrix M for the given cell. 
 *
 * @param rz z-position of given cell.
 * @param nz z-orientation of given cell.
 * @param l Length of given cell.
 * @param half_l Half-length of given cell.
 * @param ss Cell-body coordinate at which cell-surface overlap is zero.
 * @param eta0 Ambient viscosity of given cell.
 * @param eta1 Surface friction coefficient of given cell.
 * @param R Cell radius.
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero. 
 * @returns The 6x6 matrix defined above (flattened) for each cell. 
 */
template <typename T>
Array<T, 6, 6> compositeViscosityForceMatrix(const T rz, const T nz,
                                             const T l, const T half_l,
                                             const T ss, const T eta0,
                                             const T eta1, const T R,
                                             const T nz_threshold)
{
    Array<T, 6, 6> M = Array<T, 6, 6>::Zero(6, 6);
    
    T abs_nz = std::abs(nz);
    T term1 = eta0 * l;
    T term2 = eta0 * l * l * l / 12;
    T term3, term4, term5;
    if (abs_nz < nz_threshold)
    {
        T phi = R - rz; 
        if (phi > R - rz)
        {
            T prefactor = std::pow(R * phi, 0.5);
            term3 = prefactor * l; 
            term4 = 0;
            term5 = prefactor * l * l * l / 12; 
        }
        else
        {
            term3 = 0;
            term4 = 0;
            term5 = 0;
        }
    }
    else
    {
        term3 = (eta1 / R) * areaIntegral1(rz, nz, R, half_l, ss); 
        term4 = (eta1 / R) * areaIntegral2(rz, nz, R, half_l, ss);
        term5 = (eta1 / R) * areaIntegral3(rz, nz, R, half_l, ss);
    }
    M(0, 0) = term1 + term3;
    M(0, 3) = term4;
    M(1, 1) = term1 + term3;
    M(1, 4) = term4;
    M(2, 2) = term1;
    M(3, 0) = term4;
    M(3, 3) = term2 + term5;
    M(4, 1) = term4;
    M(4, 4) = term2 + term5;
    M(5, 5) = term2;

    return M;
}

/**
 * Generate a vector of Segment_3 instances for the given population of cells.
 *
 * Note that Segment_3 is not mutable, and therefore a new vector must be 
 * generated every time the population is updated in some way. 
 *
 * @param cells Existing population of cells.
 * @returns Vector of Segment_3 instances for each cell 
 */
template <typename T>
std::vector<Segment_3> generateSegments(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    std::vector<Segment_3> segments;
    for (int i = 0; i < cells.rows(); ++i)
    {
        Point_3 p(
            cells(i, 0) - cells(i, 7) * cells(i, 3),
            cells(i, 1) - cells(i, 7) * cells(i, 4),
            cells(i, 2) - cells(i, 7) * cells(i, 5)
        );
        Point_3 q(
            cells(i, 0) + cells(i, 7) * cells(i, 3),
            cells(i, 1) + cells(i, 7) * cells(i, 4),
            cells(i, 2) + cells(i, 7) * cells(i, 5)
        );
        segments.push_back(Segment_3(p, q));
    }

    return segments;
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
Array<T, Dynamic, 7> getCellNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                      const T neighbor_threshold, const T R, const T Ldiv)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return an empty array
    if (n == 1)
        return Array<T, Dynamic, 7>::Zero(0, 7);

    // Generate Segment_3 instances for each cell 
    std::vector<Segment_3> segments = generateSegments(cells);

    // Maintain array of neighboring cells 
    //
    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) z-coordinate of distance vector from cell i to cell j 
    // 5) Cell-body coordinate of contact point along centerline of cell i
    // 6) Cell-body coordinate of contact point along centerline of cell j
    int npairs = n * (n - 1) / 2;
    Array<T, Dynamic, 7> neighbors(npairs, 7);
    int idx = 0; 

    // For each pair of cells in the population ... 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)   // Note that j < i
        {
            // For two cells to be within neighbor_threshold of each other,
            // their centers must be within neighbor_threshold + Ldiv + 2 * R
            T dist_rij = (cells(i, Eigen::seq(0, 2)) - cells(j, Eigen::seq(0, 2))).matrix().norm();  
            if (dist_rij < neighbor_threshold + Ldiv + 2 * R)
            {
                // In this case, compute their actual distance and check that 
                // they lie within neighbor_threshold of each other
                auto result = distBetweenCells(
                    segments[i], segments[j], cells(i, Eigen::seq(0, 2)).matrix(),
                    cells(i, Eigen::seq(3, 5)).matrix(), cells(i, 7),
                    cells(j, Eigen::seq(0, 2)).matrix(),
                    cells(j, Eigen::seq(3, 5)).matrix(), cells(j, 7), K()
                );
                Matrix<T, 3, 1> dist_ij = std::get<0>(result); 
                T si = std::get<1>(result);
                T sj = std::get<2>(result);
                if (dist_ij.norm() < neighbor_threshold)
                {
                    neighbors(idx, 0) = i; 
                    neighbors(idx, 1) = j; 
                    neighbors(idx, Eigen::seq(2, 4)) = dist_ij.array();
                    neighbors(idx, 5) = si; 
                    neighbors(idx, 6) = sj;
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
                             Ref<Array<T, Dynamic, 7> > neighbors)
{ 
    // Generate Segment_3 instances for each cell 
    std::vector<Segment_3> segments = generateSegments(cells);

    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) z-coordinate of distance vector from cell i to cell j
    // 5) Cell-body coordinate of contact point along centerline of cell i
    // 6) Cell-body coordinate of contact point along centerline of cell j
    //
    // Columns 2, 3, 4, 5, 6 are updated here
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));
        auto result = distBetweenCells(
            segments[i], segments[j], cells(i, Eigen::seq(0, 2)).matrix(),
            cells(i, Eigen::seq(3, 5)).matrix(), cells(i, 7),
            cells(j, Eigen::seq(0, 2)).matrix(),
            cells(j, Eigen::seq(3, 5)).matrix(), cells(j, 7), K()
        );
        Matrix<T, 3, 1> dist_ij = std::get<0>(result); 
        T si = std::get<1>(result);
        T sj = std::get<2>(result);
        neighbors(k, Eigen::seq(2, 4)) = dist_ij.array(); 
        neighbors(k, 5) = si; 
        neighbors(k, 6) = sj;
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
 * @param prefactors Array of four pre-computed prefactors, namely `2.5 * sqrt(R)`,
 *                   `2.5 * E0 * sqrt(R)`, `E0 * pow(R - Rcell, 1.5)`, and `Ecell`.
 * @returns Derivatives of the cell-cell interaction energies with respect 
 *          to cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellCellForcesFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                 const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                                 const T R, const T Rcell,
                                                 const Ref<const Array<T, 4, 1> >& prefactors)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 6>::Zero(n, 6); 

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, z-position, x-orientation,
    // y-orientation, z-orientation
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(n, 6);

    // Compute distance vector magnitude, direction, and corresponding
    // cell-cell overlap for every pair of neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 4)).matrix().rowwise().norm().array(); 
    Array<T, Dynamic, 3> directions = neighbors(Eigen::all, Eigen::seq(2, 4)).colwise() / magnitudes;
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
        T si = neighbors(k, 5);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 6);                         // Cell-body coordinate along cell j
        Array<T, 3, 1> dir_ij = directions.row(k);      // Normalized distance vector 
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
        //
        // TODO Does this change the verticalization trajectory?
        else if (overlap >= R - Rcell)
        {
            //prefactor = prefactors(1) * std::pow(overlap, 1.5); 
            T term = prefactors(3) * std::pow(overlap - R + Rcell, 1.5);
            prefactor = prefactors(0) * (prefactors(2) + term);
        }

        if (overlap > 0)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 3, 1> vij = prefactor * dir_ij;
            dEdq(i, Eigen::seq(0, 2)) += vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell i
            dEdq(i, Eigen::seq(3, 5)) += vij * si; 
            // Derivative of cell-cell interaction energy w.r.t position of cell j
            dEdq(j, Eigen::seq(0, 2)) -= vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell j
            dEdq(j, Eigen::seq(3, 5)) -= vij * sj;
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
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for
 *                             cell-cell interaction forces.
 * @param E0 Elastic modulus of EPS.
 * @param adhesion_energy_density Cell-surface adhesion energy density.
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero.
 * @param noise Pre-determined vector of noise values; should range between
 *              -1 and 1.
 * @param noise_strength Noise strength.
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 6> getVelocitiesFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                                const T R, const T Rcell,
                                                const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                                                const T E0,
                                                const T adhesion_energy_density,
                                                const T nz_threshold, 
                                                const Ref<const Array<T, 6, 1> >& noise,
                                                const T noise_strength)
{
    // For each cell, the relevant Lagrangian mechanics are given by 
    // 
    // dP/d(dq) = -dE/dq + lambda * d(nx^2 + ny^2 + nz^2 - 1)/dq, 
    //
    // where:
    // - P is the total dissipation due to bulk viscosity and surface friction,
    // - E is the total cell-cell interaction energy involving the given cell,
    // - q is a generalized coordinate (x-position, y-position, z-position, 
    //   x-orientation, y-orientation, z-orientation),
    // - nx, ny, nz are the x-, y-, z-orientations, respectively,
    // - dq is the corresponding velocity, and 
    // - lambda is a Lagrange multiplier. 
    //
    // This yields a system of seven equations in seven variables (drx, dry,
    // drz, dnx, dny, dnz, lambda), which this function solves for each cell
    int n = cells.rows(); 
    Array<T, Dynamic, 6> velocities = Array<T, Dynamic, 6>::Zero(n, 6);

    // Get cell-body coordinates at which cell-surface overlap is zero for 
    // each cell
    Array<T, Dynamic, 1> abs_nz = cells(Eigen::all, 5).abs();
    Array<T, Dynamic, 1> ss(n); 
    for (int i = 0; i < n; ++i)
    {
        if (abs_nz(i) < nz_threshold)
            ss(i) = std::numeric_limits<T>::quiet_NaN();
        else
            ss(i) = sstar(cells(i, 2), cells(i, 5), R); 
    }
    
    // Get the derivatives of the cell-cell interaction energy, cell-surface
    // repulsion energy, and cell-surface adhesion energy for each cell 
    Array<T, Dynamic, 6> dEdq_cell = cellCellForcesFromNeighbors<T>(
        cells, neighbors, R, Rcell, cell_cell_prefactors 
    );
    Array<T, Dynamic, 2> dEdq_surface_repulsion = cellSurfaceRepulsionForces<T>(
        cells, ss, R, E0, nz_threshold
    );
    Array<T, Dynamic, 2> dEdq_surface_adhesion = cellSurfaceAdhesionForces<T>(
        cells, ss, R, adhesion_energy_density, nz_threshold
    );

    // For each cell ... 
    for (int i = 0; i < n; ++i)
    {
        Array<T, 7, 7> A = Array<T, 7, 7>::Zero();
        Array<T, 7, 1> b = Array<T, 7, 1>::Zero();

        // Compute the derivatives of the dissipation with respect to the 
        // cell's translational and orientational velocities 
        A(Eigen::seq(0, 5), Eigen::seq(0, 5)) = compositeViscosityForceMatrix<T>(
            cells(i, 2), cells(i, 5), cells(i, 6), cells(i, 7), ss(i),
            cells(i, 10), cells(i, 11), R, nz_threshold
        );
        A(3, 6) = -2 * cells(i, 3); 
        A(4, 6) = -2 * cells(i, 4);
        A(5, 6) = -2 * cells(i, 5);
        A(6, 3) = cells(i, 3); 
        A(6, 4) = cells(i, 4);
        A(6, 5) = cells(i, 5);

        // Extract the derivatives of the cell-cell interaction energy, 
        // cell-surface repulsion energy, and cell-surface adhesion energy
        b(0) = -dEdq_cell(i, 0);
        b(1) = -dEdq_cell(i, 1);
        b(2) = -dEdq_cell(i, 2) - dEdq_surface_repulsion(i, 0) - dEdq_surface_adhesion(i, 0);
        b(3) = -dEdq_cell(i, 3);
        b(4) = -dEdq_cell(i, 4);
        b(5) = -dEdq_cell(i, 5) - dEdq_surface_repulsion(i, 1) - dEdq_surface_adhesion(i, 1);

        // Solve the corresponding linear system without noise
        auto QR = A.matrix().colPivHouseholderQr(); 
        Array<T, 7, 1> x = QR.solve(b.matrix()).array();

        // If a nonzero noise vector has been specified ... 
        if ((noise != 0).any())
        {
            // Solve the linear system corresponding to the given noise vector
            Array<T, 7, 1> r = Array<T, 7, 1>::Zero();
            r(Eigen::seq(0, 5)) = noise;
            Array<T, 7, 1> y = QR.solve(r.matrix()).array();

            // Determine noise magnitude by comparing magnitudes of x and y
            T eps = noise_strength * x.head(6).matrix().norm() / y.head(6).matrix().norm();

            // Compute resulting velocities 
            velocities.row(i) = (x + eps * y).head(6);
        }
        else    // Otherwise ... 
        {
            velocities.row(i) = x.head(6);
        }
    }

    return velocities;  
}

/**
 * Normalize the orientation vectors of all cells in the given population,
 * and redirect all orientation vectors with positive z-coordinate.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells. 
 */
template <typename T>
void normalizeOrientations(Ref<Array<T, Dynamic, Dynamic> > cells)
{
    Array<T, Dynamic, 1> norms = cells(Eigen::all, Eigen::seq(3, 5)).matrix().rowwise().norm().array();
    assert((norms > 0).all() && "Zero norms encountered during orientation normalization");
    cells.col(3) /= norms; 
    cells.col(4) /= norms;
    cells.col(5) /= norms;

    for (int i = 0; i < cells.rows(); ++i)
    {
        if (cells(i, 5) > 0)
        {
            cells(i, 3) *= -1;
            cells(i, 4) *= -1;
            cells(i, 5) *= -1;
        }
    }
}

/**
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
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for
 *                             cell-cell interaction forces.
 * @param E0 Elastic modulus of EPS. 
 * @param adhesion_energy_density Cell-surface adhesion energy density.
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero.
 * @param rng Random number generator.
 * @param noise_dist Noise distribution.
 * @param noise_strength Noise strength.  
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T>
std::tuple<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 6>, Array<T, Dynamic, 6> >
    stepRungeKuttaAdaptiveFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                                        const Ref<const Array<T, Dynamic, 1> >& b,
                                        const Ref<const Array<T, Dynamic, 1> >& bs, 
                                        const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                                        const Ref<const Array<T, Dynamic, 7> >& neighbors, 
                                        const T dt, const T R, const T Rcell,
                                        const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                                        const T E0,
                                        const T adhesion_energy_density,
                                        const T nz_threshold,
                                        boost::random::mt19937& rng,
                                        std::function<T(boost::random::mt19937&)>& noise_dist,
                                        const T noise_strength)
{
    // Determine noise to add to each generalized force at each timestep
    Array<T, 6, 1> noise = Array<T, 6, 1>::Zero();
    if (cells.rows() > 1)
    {
        for (int i = 0; i < 6; ++i)
            noise(i) = noise_dist(rng);
    }

    // Compute velocities at given partial timesteps 
    int n = cells.rows(); 
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 6> > velocities; 
    velocities.push_back(
        getVelocitiesFromNeighbors<T>(
            cells, neighbors, R, Rcell, cell_cell_prefactors, E0,
            adhesion_energy_density, nz_threshold, noise, noise_strength
        )
    );
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 6> multipliers = Array<T, Dynamic, 6>::Zero(n, 6);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, Eigen::seq(0, 5)) += multipliers * dt;
        normalizeOrientations<T>(cells_i);    // Renormalize orientations after each modification
        velocities.push_back(
            getVelocitiesFromNeighbors<T>(
                cells_i, neighbors, R, Rcell, cell_cell_prefactors, E0,
                adhesion_energy_density, nz_threshold, noise, noise_strength
            )
        );
    }

    // Compute Runge-Kutta update from computed velocities
    Array<T, Dynamic, Dynamic> cells_new(cells); 
    Array<T, Dynamic, 6> velocities_final1 = Array<T, Dynamic, 6>::Zero(n, 6); 
    Array<T, Dynamic, 6> velocities_final2 = Array<T, Dynamic, 6>::Zero(n, 6); 
    for (int i = 0; i < s; ++i)
    {
        velocities_final1 += velocities[i] * b(i); 
        velocities_final2 += velocities[i] * bs(i); 
    }
    Array<T, Dynamic, 6> delta1 = velocities_final1 * dt; 
    Array<T, Dynamic, 6> delta2 = velocities_final2 * dt; 
    cells_new(Eigen::all, Eigen::seq(0, 5)) += delta1;
    Array<T, Dynamic, 6> errors = delta1 - delta2; 
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells_new); 

    return std::make_tuple(cells_new, errors, velocities_final1); 
}

#endif
