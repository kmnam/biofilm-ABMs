/**
 * Implementations of cell-cell and cell-surface interaction forces.
 *
 * In what follows, a population of N cells is represented as a 2-D array
 * with N rows, whose columns are as specified in `indices.hpp`.
 *
 * Additional columns may be included in the array but these are not relevant
 * for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/7/2025
 */

#ifndef BIOFILM_MECHANICS_2D_HPP
#define BIOFILM_MECHANICS_2D_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "indices.hpp"
#include "distances.hpp"
#include "kiharaGBK.hpp"
#include "confinement.hpp"

using namespace Eigen;

using std::min; 
using boost::multiprecision::min;
using std::sqrt; 
using boost::multiprecision::sqrt; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * An enum that enumerates the different adhesion force types. 
 */
enum class AdhesionMode
{
    NONE = 0,
    KIHARA = 1,
    GBK = 2
};

/**
 * An enum that enumerates the different confinement modes.
 */
enum class ConfinementMode
{
    NONE = 0,
    RADIAL = 1,
    CHANNEL = 2
};

/**
 * Output an error message pertaining to the given cell and the generalized 
 * forces exerted upon it.  
 *
 * @param r Cell center.
 * @param n Cell orientation.
 * @param half_l Cell half-length.
 * @param dEdq Array of generalized forces. 
 */
template <typename T>
void cellForcesSummary(const Ref<const Array<T, 2, 1> >& r,
                       const Ref<const Array<T, 2, 1> >& n, const T half_l,
                       const Ref<const Array<T, 4, 1> >& dEdq)
{
    std::cerr << std::setprecision(10)
              << "Cell center = (" << r(0) << ", " << r(1) << ")" << std::endl
              << "Cell orientation = (" << n(0) << ", " << n(1) << ")" << std::endl
              << "Cell half-length = " << half_l << std::endl
              << "Generalized forces: " << std::endl
              << " - w.r.t r(0) = " << dEdq(0) << std::endl
              << " - w.r.t r(1) = " << dEdq(1) << std::endl 
              << " - w.r.t n(0) = " << dEdq(2) << std::endl
              << " - w.r.t n(1) = " << dEdq(3) << std::endl;
}

/**
 * Output an error message pertaining to the given cell-cell configuration
 * and the generalized forces between them. 
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param dr1 Translational velocity of cell 1. 
 * @param dn1 Orientational velocity of cell 1. 
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param dr2 Translational velocity of cell 2. 
 * @param dn2 Orientational velocity of cell 2. 
 * @param half_l2 Half of length of cell 2.
 * @param d12 Distance vector from cell 1 to cell 2. 
 * @param s Cell-body coordinate of contact point along cell 1.
 * @param t Cell-body coordinate of contact point along cell 2.
 * @param dEdq Array of generalized forces. 
 */
template <typename T>
void pairForcesSummary(const Ref<const Array<T, 2, 1> >& r1,
                       const Ref<const Array<T, 2, 1> >& n1,
                       const Ref<const Array<T, 2, 1> >& dr1, 
                       const Ref<const Array<T, 2, 1> >& dn1, const T half_l1, 
                       const Ref<const Array<T, 2, 1> >& r2,
                       const Ref<const Array<T, 2, 1> >& n2,
                       const Ref<const Array<T, 2, 1> >& dr2, 
                       const Ref<const Array<T, 2, 1> >& dn2, const T half_l2,  
                       const Ref<const Array<T, 2, 1> >& d12, const T s, const T t,
                       const Ref<const Array<T, 2, 4> >& dEdq)
{
    std::cerr << std::setprecision(10)
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ")" << std::endl
              << "Cell 1 translational velocity = (" << dr1(0) << ", " << dr1(1) << ")" << std::endl
              << "Cell 1 orientational velocity = (" << dn1(0) << ", " << dn1(1) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ")" << std::endl
              << "Cell 2 translational velocity = (" << dr2(0) << ", " << dr2(1) << ")" << std::endl
              << "Cell 2 orientational velocity = (" << dn2(0) << ", " << dn2(1) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl
              << "Distance vector = (" << d12(0) << ", " << d12(1) << ")" << std::endl 
              << "Cell-body coordinate of contact point along cell 1 = " << s << std::endl
              << "Cell-body coordinate of contact point along cell 2 = " << t << std::endl 
              << "Generalized forces: " << std::endl
              << " - w.r.t r1(0) = " << dEdq(0, 0) << std::endl
              << " - w.r.t r1(1) = " << dEdq(0, 1) << std::endl 
              << " - w.r.t n1(0) = " << dEdq(0, 2) << std::endl
              << " - w.r.t n1(1) = " << dEdq(0, 3) << std::endl
              << " - w.r.t r2(0) = " << dEdq(1, 0) << std::endl
              << " - w.r.t r2(1) = " << dEdq(1, 1) << std::endl 
              << " - w.r.t n2(0) = " << dEdq(1, 2) << std::endl
              << " - w.r.t n2(1) = " << dEdq(1, 3) << std::endl;
}

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
 * @param cells Current population of cells. 
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
    Array<T, Dynamic, 1> composite_drag = (
        cells.col(__colidx_eta0) + cells.col(__colidx_eta1) * surface_contact_density / R
    ); 
    KL.col(0) = cells.col(__colidx_l) * composite_drag; 
    KL.col(1) = (
        cells.col(__colidx_l) * cells.col(__colidx_l) * cells.col(__colidx_l) * composite_drag / 12
    );

    return KL; 
}

/**
 * Get all pairs of cells whose distances are within the given threshold.
 *
 * The returned array is given in row-major format.  
 *
 * @param cells Current population of cells.
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
            T dist_rij = (cells(i, __colseq_r) - cells(j, __colseq_r)).matrix().norm(); 
            if (dist_rij < neighbor_threshold + Ldiv + 2 * R)
            {
                // In this case, compute their actual distance and check that 
                // they lie within neighbor_threshold of each other 
                auto result = distBetweenCells<T>(
                    segments[i], segments[j],
                    static_cast<int>(cells(i, __colidx_id)),
                    cells(i, __colseq_r).matrix(), 
                    cells(i, __colseq_n).matrix(),
                    cells(i, __colidx_half_l),
                    cells(i, __colseq_dr).matrix(), 
                    static_cast<int>(cells(j, __colidx_id)),
                    cells(j, __colseq_r).matrix(),
                    cells(j, __colseq_n).matrix(),
                    cells(j, __colidx_half_l),
                    cells(j, __colseq_dr).matrix(), 
                    kernel
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
 * @param cells Current population of cells. 
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
            segments[i], segments[j],
            static_cast<int>(cells(i, __colidx_id)), 
            cells(i, __colseq_r).matrix(),
            cells(i, __colseq_n).matrix(),
            cells(i, __colidx_half_l),
            cells(i, __colseq_dr).matrix(),
            static_cast<int>(cells(j, __colidx_id)), 
            cells(j, __colseq_r).matrix(),
            cells(j, __colseq_n).matrix(),
            cells(j, __colidx_half_l),
            cells(j, __colseq_dr).matrix(),
            kernel
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
 * Compute the derivatives of the cell-cell repulsion energy for each pair
 * of neighboring cells, with respect to each cell's position and orientation
 * coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param iter Iteration number. Only used for debugging output. 
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
                                             const int iter, const T R, const T Rcell,
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
            Array<T, 2, 1> vij = prefactor * dir_ij;
            Array<T, 2, 4> forces;
            forces << vij(0),       vij(1),        // Derivatives w.r.t position of cell i
                      vij(0) * si,  vij(1) * si,   // Derivatives w.r.t orientation of cell i
                      -vij(0),      -vij(1),       // Derivatives w.r.t position of cell j
                      -vij(0) * sj, -vij(1) * sj;  // Derivatives w.r.t orientation of cell j
            #ifdef DEBUG_CHECK_REPULSIVE_FORCES_NAN
                if (forces.isNaN().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in repulsive forces between cells " 
                              << i << " and " << j << std::endl;
                    pairForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colseq_dr), cells(i, __colseq_dn), 
                        cells(i, __colidx_half_l),
                        cells(j, __colseq_r), cells(j, __colseq_n),
                        cells(j, __colseq_dr), cells(j, __colseq_dn), 
                        cells(j, __colidx_half_l),
                        neighbors(k, Eigen::seq(2, 3)), si, sj, 
                        forces
                    );
                    throw std::runtime_error("Found nan in repulsive forces"); 
                }
            #endif
            dEdq.row(i) += forces.row(0); 
            dEdq.row(j) += forces.row(1); 
        }
    }

    return dEdq;  
}

/**
 * Compute the derivatives of the cell-cell adhesion potential energy for
 * each pair of neighboring cells, with respect to each cell's position and
 * orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param mode Choice of potential used to model cell-cell adhesion. Can be
 *             NONE (0), KIHARA (1), or GBK (2).
 * @param params Parameters required to compute cell-cell adhesion forces. 
 * @returns Derivatives of the cell-cell adhesion energies with respect to  
 *          cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 4> cellCellAdhesiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                            const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                            const int iter, const T R, const T Rcell, 
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
            Matrix<T, 2, 1> ri = cells(i, __colseq_r).matrix();
            Matrix<T, 2, 1> ni = cells(i, __colseq_n).matrix();
            Matrix<T, 2, 1> rj = cells(j, __colseq_r).matrix();
            Matrix<T, 2, 1> nj = cells(j, __colseq_n).matrix();
            T half_li = cells(i, __colidx_half_l);
            T half_lj = cells(j, __colidx_half_l);
            Matrix<T, 2, 1> dij = neighbors(k, Eigen::seq(2, 3)).matrix();
            T si = neighbors(k, 4); 
            T sj = neighbors(k, 5); 

            // Get the corresponding forces
            Array<T, 2, 4> forces; 
            if (mode == AdhesionMode::KIHARA) 
            {
                const T strength = params["strength"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                forces = strength * forcesKiharaLagrange<T, 2>(dij, R, si, sj, expd, dmin);
            }
            else if (mode == AdhesionMode::GBK)
            {
                const T strength = params["strength"];
                const T exp1 = params["anisotropy_exp1"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                forces = strength * forcesGBKLagrange<T, 2>(
                    ri, ni, half_li, rj, nj, half_lj, R, Rcell, dij, si, sj,
                    expd, exp1, dmin
                ); 
            }
            #ifdef DEBUG_CHECK_ADHESIVE_FORCES_NAN
                if (forces.isNaN().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in adhesive forces between cells " 
                              << i << " and " << j << std::endl;
                    pairForcesSummary<T>(
                        ri.array(), ni.array(), cells(i, __colseq_dr), cells(i, __colseq_dn), half_li,
                        rj.array(), nj.array(), cells(j, __colseq_dr), cells(j, __colseq_dn), half_lj,  
                        dij.array(), si, sj, forces
                    );
                    throw std::runtime_error("Found nan in adhesive forces"); 
                }
            #endif
            dEdq.row(i) += forces.row(0); 
            dEdq.row(j) += forces.row(1);
        }
    }

    return dEdq;
}

/**
 * Compute the derivatives of the dissipation due to cell-cell tangential 
 * friction for each pair of neighboring cells, with respect to each cell's
 * translational and orientational velocities.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed.
 *
 * TODO This function is being evaluated for correctness. It may not be 
 * correct, and may be removed in the future (in favor of a Newtonian 
 * implementation). 
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param eta Cell-cell friction coefficient. 
 * @returns Derivatives of the dissipation due to cell-cell tangential friction
 *          for each pair of neighboring cells. 
 */
template <typename T>
Array<T, Dynamic, 4> cellCellFrictionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                            const int iter, const T R, const T eta)
{
    int n = cells.rows();       // Number of cells
    int m = neighbors.rows();   // Number of neighboring pairs

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4);

    // Initialize array of derivatives 
    Array<T, Dynamic, 4> dPdq = Array<T, Dynamic, 4>::Zero(n, 4);
    for (int k = 0; k < m; ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));
        Matrix<T, 2, 1> dij = neighbors(k, Eigen::seq(2, 3)).matrix();
        T d = dij.norm();

        // If the two cells are contacting ... 
        if (d < 2 * R)
        {
            Array<T, 2, 1> ri = cells(i, __colseq_r); 
            Array<T, 2, 1> ni = cells(i, __colseq_n);
            Array<T, 2, 1> dri = cells(i, __colseq_dr); 
            Array<T, 2, 1> dni = cells(i, __colseq_dn); 
            Array<T, 2, 1> rj = cells(j, __colseq_r); 
            Array<T, 2, 1> nj = cells(j, __colseq_n); 
            Array<T, 2, 1> drj = cells(j, __colseq_dr); 
            Array<T, 2, 1> dnj = cells(j, __colseq_dn); 
            Matrix<T, 2, 1> dijn = dij / d; 
            T si = neighbors(k, 4); 
            T sj = neighbors(k, 5);

            // Determine the contact point 
            Array<T, 2, 1> contact = (ri + si * ni + rj + sj * nj) / 2; 

            // Determine the angular velocities of the two cells (note that 
            // these are really the z-coordinates of 3-D vectors that are 
            // parallel with the z-axis)
            T angvel_i = (ni(0) == 0 ? -dni(0) / ni(1) : dni(1) / ni(0)); 
            T angvel_j = (nj(0) == 0 ? -dnj(0) / nj(1) : dnj(1) / nj(0));  

            // Get the relative velocity of the two cells at the contact point ... 
            Array<T, 2, 1> ui = contact - ri; 
            Array<T, 2, 1> uj = contact - rj;
            Array<T, 2, 1> ci, cj;
            ci << -angvel_i * ui(1), angvel_i * ui(0); 
            cj << -angvel_j * uj(1), angvel_j * uj(0);
            Array<T, 2, 1> vij = (dri + ci) - (drj + cj);

            // ... and its projection onto the normal direction ... 
            Array<T, 2, 1> vijn = vij.matrix().dot(dijn) * dijn.array();

            // ... and the corresponding rejection 
            Array<T, 2, 1> vijt = vij - vijn; 

            // Get the square root of the overlap between the two cells,
            // multiplied by the radius 
            T sqrt_overlap = sqrt((2 * R - d) * R); 

            // Derivatives of the dissipation w.r.t the velocities of cell i 
            // and cell j 
            //
            // Note that this is the *negative* of the force on cell i due
            // to cell j (and vice versa)
            Array<T, 2, 4> dPdq_ij = Array<T, 2, 4>::Zero();  
            dPdq_ij(0, Eigen::seq(0, 1)) = eta * sqrt_overlap * vijt;
            dPdq_ij(1, Eigen::seq(0, 1)) = -dPdq_ij(0, Eigen::seq(0, 1)); 

            // Derivatives of the dissipation w.r.t the orientational velocities 
            // of cell i and cell j
            Array<T, 2, 1> wi, wj;
            wi << ui(1) / ni(1), ui(0) / ni(0); 
            wj << uj(1) / nj(1), uj(0) / nj(0); 
            dPdq_ij(0, Eigen::seq(2, 3)) = eta * sqrt_overlap * (wi * vijt);
            dPdq_ij(1, Eigen::seq(2, 3)) = -eta * sqrt_overlap * (wj * vijt);

            #ifdef DEBUG_CHECK_FRICTION_FORCES_NAN
                if (dPdq_ij.isNaN().any())
                {
                    pairForcesSummary<T>(
                        ri, ni, dri, dni, cells(i, __colidx_half_l),
                        rj, nj, drj, dnj, cells(j, __colidx_half_l), 
                        dij.array(), si, sj, dPdq_ij
                    );
                    std::cout << "Contact point = (" << contact(0) << ", "
                              << contact(1) << ")" << std::endl; 
                    std::cout << "Cell 1 angular velocity = " << angvel_i << std::endl; 
                    std::cout << "Cell 2 angular velocity = " << angvel_j << std::endl; 
                    std::cout << "Relative velocity = (" << vij(0) << ", "
                              << vij(1) << ")" << std::endl;
                    std::cout << "Relative normal velocity = (" << vijn(0) << ", "
                              << vijn(1) << ")" << std::endl; 
                    std::cout << "Relative tangential velocity = (" << vijt(0) << ", "
                              << vijt(1) << ")" << std::endl;
                    throw std::runtime_error("Found nan in cell-cell friction forces"); 
                }
            #endif

            dPdq.row(i) += dPdq_ij.row(0); 
            dPdq.row(j) += dPdq_ij.row(1); 
        }
    }

    return dPdq; 
}

/**
 * Given the current positions, orientations, lengths, viscosity coefficients,
 * and surface friction coefficients for the given population of cells, compute
 * their translational and orientational velocities.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed, and there is *no* cell-cell tangential friction. 
 *
 * @param cells Current population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param noise Noise to be added to each generalized force used to compute
 *              the velocities. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2).
 * @param boundary_idx Pre-computed vector of indices of peripheral cells. 
 * @param confine_params Parameters required to compute confinement forces.
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 4> getVelocities(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                   const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                   const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                   const int iter, const T R, const T Rcell,
                                   const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                                   const T surface_contact_density,
                                   const Ref<const Array<T, Dynamic, 4> >& noise,
                                   const AdhesionMode adhesion_mode,
                                   std::unordered_map<std::string, T>& adhesion_params,
                                   const ConfinementMode confine_mode, 
                                   std::vector<int>& boundary_idx,
                                   std::unordered_map<std::string, T>& confine_params)
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
    // Moreover, one of the orientational velocities can be obtained from 
    // the other, as in 
    //
    // dny = -nx * dnx / ny
    //
    int n = cells.rows(); 
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 2> prefactors = compositeViscosityForcePrefactors<T>(
        cells, R, surface_contact_density
    );
    Array<T, Dynamic, 1> K = prefactors.col(0);
    Array<T, Dynamic, 1> L = prefactors.col(1);

    // Compute the repulsive forces ... 
    Array<T, Dynamic, 4> dEdq_repulsion = cellCellRepulsiveForces<T>(
        cells, neighbors, iter, R, Rcell, cell_cell_prefactors
    );

    // ... the adhesive forces (if adhesion is present) ... 
    Array<T, Dynamic, 4> dEdq_adhesion = Array<T, Dynamic, 4>::Zero(n, 4); 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        dEdq_adhesion = cellCellAdhesiveForces<T>(
            cells, neighbors, to_adhere, iter, R, Rcell, adhesion_mode,
            adhesion_params
        );
    }

    // ... and the confinement forces (if present)
    Array<T, Dynamic, 4> dEdq_confine = Array<T, Dynamic, 4>::Zero(n, 4); 
    if (confine_mode == ConfinementMode::RADIAL)
    {
        const T rest_radius_factor = confine_params["rest_radius_factor"]; 
        const T spring_const = confine_params["spring_const"];
        const T max_area = getMaxArea<T>(cells, R); 
        const T rest_radius = rest_radius_factor * sqrt(max_area / boost::math::constants::pi<T>()); 
        Matrix<T, 2, 1> center = Matrix<T, 2, 1>::Zero();
        dEdq_confine = radialConfinementForces<T>(
            cells, boundary_idx, R, center, rest_radius, spring_const
        );
        #ifdef DEBUG_CHECK_CONFINEMENT_FORCES_NAN
            for (int i = 0; i < n; ++i)
            {
                if (dEdq_confine.row(i).isNaN().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in confinement forces for cell "
                              << i << std::endl;
                    cellForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colidx_half_l),
                        dEdq_confine.row(i).transpose()
                    );
                }
            }
        #endif
    }
    else if (confine_mode == ConfinementMode::CHANNEL) 
    {
        const T short_section_y = confine_params["short_section_y"]; 
        const T left_long_section_x = confine_params["left_long_section_x"]; 
        const T right_long_section_x = confine_params["right_long_section_x"]; 
        const T spring_const = confine_params["spring_const"]; 
        dEdq_confine = channelConfinementForces<T>(
            cells, boundary_idx, R, short_section_y, left_long_section_x, 
            right_long_section_x, spring_const
        );
        #ifdef DEBUG_CHECK_CONFINEMENT_FORCES_NAN
            for (int i = 0; i < n; ++i)
            {
                if (dEdq_confine.row(i).isNaN().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in confinement forces for cell "
                              << i << std::endl;
                    cellForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colidx_half_l),
                        dEdq_confine.row(i).transpose()
                    );
                }
            }
        #endif
    }

    // Combine the three types of forces (with the noise) 
    Array<T, Dynamic, 4> dEdq = dEdq_repulsion + dEdq_adhesion + dEdq_confine + noise;

    // Set mult = 2 * lambda
    Array<T, Dynamic, 1> mult = (
        cells.col(__colidx_nx) * dEdq.col(2) + cells.col(__colidx_ny) * dEdq.col(3)
    );

    // Solve the Lagrangian equations of motion for each cell in the population
    Array<T, Dynamic, 2> dEdn_constrained = (
        dEdq(Eigen::all, Eigen::seq(2, 3)) -
        cells(Eigen::all, __colseq_n).colwise() * mult
    );
    velocities.col(0) = -dEdq.col(0) / K;
    velocities.col(1) = -dEdq.col(1) / K; 
    velocities.col(2) = -dEdn_constrained.col(0) / L;
    velocities.col(3) = -dEdn_constrained.col(1) / L;

    return velocities;  
}

/**
 * Truncate the cell-surface friction coefficients so that the cell velocities
 * in the next timestep (which should be similar to the given array of cell
 * velocities for the current timestep) obey Coulomb's law of friction.
 *
 * The given array of cell data is updated in place. 
 *
 * @param cells Current population of cells.
 * @param R Cell radius, including the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param surface_contact_density Cell-surface contact area density.
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
 */
template <typename T>
void truncateSurfaceFrictionCoeffsCoulomb(Ref<Array<T, Dynamic, Dynamic> > cells,
                                          const T R, const T E0,
                                          const T surface_contact_density,
                                          const T surface_coulomb_coeff)
{
    int n = cells.rows(); 

    // Compute the cell-surface friction coefficient bound for each cell
    // (which is velocity-dependent)
    const T surface_delta = surface_contact_density * surface_contact_density / R;
    Array<T, Dynamic, 1> speeds = cells(Eigen::all, __colseq_dr).matrix().rowwise().norm().array();
    Array<T, Dynamic, 1> eta1_bounds = (
        2 * E0 * surface_delta * surface_coulomb_coeff * R / (surface_contact_density * speeds)
    );

    // Truncate each cell-surface friction coefficient
    for (int i = 0; i < n; ++i)
        cells(i, __colidx_eta1) = min(eta1_bounds(i), cells(i, __colidx_maxeta1)); 
}

/**
 * Normalize the orientation vectors of all cells in the given population.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Current population of cells. 
 */
template <typename T>
void normalizeOrientations(Ref<Array<T, Dynamic, Dynamic> > cells)
{
    Array<T, Dynamic, 1> norms = cells(Eigen::all, __colseq_n).matrix().rowwise().norm().array();
    assert((norms > 0).all() && "Zero norms encountered during orientation normalization");
    cells.col(__colidx_nx) /= norms; 
    cells.col(__colidx_ny) /= norms;
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
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero. 
 * @param dt Timestep.
 * @param iter Iteration number. Only used for debugging output.  
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of four pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2).
 * @param boundary_idx Pre-computed vector of indices of peripheral cells. 
 * @param confine_params Parameters required to compute confinement forces.
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 4> >
    stepRungeKuttaAdaptive(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                           const Ref<const Array<T, Dynamic, 1> >& b,
                           const Ref<const Array<T, Dynamic, 1> >& bs, 
                           const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                           const Ref<const Array<T, Dynamic, 6> >& neighbors,
                           const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                           const T dt, const int iter, const T R, const T Rcell,
                           const Ref<const Array<T, 4, 1> >& cell_cell_prefactors,
                           const T surface_contact_density, const T max_noise,
                           boost::random::mt19937& rng,
                           boost::random::uniform_01<>& uniform_dist,
                           const AdhesionMode adhesion_mode, 
                           std::unordered_map<std::string, T>& adhesion_params,
                           const ConfinementMode confine_mode, 
                           std::vector<int>& boundary_idx, 
                           std::unordered_map<std::string, T>& confine_params)
{
    #ifdef DEBUG_CHECK_NEIGHBOR_DISTANCES_ZERO
        for (int k = 0; k < neighbors.rows(); ++k)
        {
            if (neighbors(k, Eigen::seq(2, 3)).matrix().norm() < 1e-8)
            {
                int i = neighbors(k, 0); 
                int j = neighbors(k, 1); 
                std::cerr << "Iteration " << iter
                          << ": Found near-zero distance between cells "
                          << i << " and " << j << std::endl;
                pairConfigSummary<T>(
                    static_cast<int>(cells(i, __colidx_id)),
                    cells(i, __colseq_r).matrix(),
                    cells(i, __colseq_n).matrix(), cells(i, __colidx_half_l),
                    cells(i, __colseq_dr).matrix(),
                    static_cast<int>(cells(j, __colidx_id)),
                    cells(j, __colseq_r).matrix(),
                    cells(j, __colseq_n).matrix(), cells(j, __colidx_half_l),
                    cells(j, __colseq_dr).matrix()
                );
                throw std::runtime_error("Found near-zero distance");
            }
        }
    #endif

    // Compute velocities at given partial timesteps
    //
    // Sample noise components prior to velocity calculations, so that they 
    // are the same in each calculation 
    int n = cells.rows(); 
    Array<T, Dynamic, 4> noise = Array<T, Dynamic, 4>::Zero(n, 4);  
    if (max_noise > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                T r = uniform_dist(rng); 
                noise(i, j) = -max_noise + 2 * max_noise * r; 
            }
        }
    }
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 4> > velocities;
    Array<T, Dynamic, 4> v0 = getVelocities<T>(
        cells, neighbors, to_adhere, iter, R, Rcell, cell_cell_prefactors,
        surface_contact_density, noise, adhesion_mode, adhesion_params,
        confine_mode, boundary_idx, confine_params
    );
    velocities.push_back(v0);
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 4> multipliers = Array<T, Dynamic, 4>::Zero(n, 4);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, __colseq_coords) += multipliers * dt;
        normalizeOrientations<T>(cells_i);    // Renormalize orientations after each modification
        Array<T, Dynamic, 4> vi = getVelocities<T>(
            cells_i, neighbors, to_adhere, iter, R, Rcell, cell_cell_prefactors,
            surface_contact_density, noise, adhesion_mode, adhesion_params,
            confine_mode, boundary_idx, confine_params
        );
        velocities.push_back(vi);
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
    cells_new(Eigen::all, __colseq_coords) += delta1; 
    Array<T, Dynamic, 4> errors = delta1 - delta2;

    // Store computed velocities 
    cells_new(Eigen::all, __colseq_velocities) = velocities_final1; 
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells_new);

    return std::make_pair(cells_new, errors);
}

#endif
