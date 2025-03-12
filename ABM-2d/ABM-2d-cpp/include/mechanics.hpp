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
 *     3/12/2025
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
#include "adhesion.hpp"
#include "confinement.hpp"

using namespace Eigen;

using std::min; 
using boost::multiprecision::min;
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::pow; 
using boost::multiprecision::pow; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * An enum that enumerates the different adhesion force types. 
 */
enum class AdhesionMode
{
    NONE = 0,
    JKR = 1,
    KIHARA = 2,
    GBK = 3
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
 * The input forces here are assumed to be an array of generalized forces in 
 * the Lagrangian framework. 
 *
 * @param r Cell center.
 * @param n Cell orientation.
 * @param half_l Cell half-length.
 * @param dEdq Array of generalized forces. 
 */
template <typename T>
void cellLagrangianForcesSummary(const Ref<const Array<T, 2, 1> >& r,
                                 const Ref<const Array<T, 2, 1> >& n,
                                 const T half_l,
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
 * The input forces here are assumed to be an array of generalized forces in 
 * the Lagrangian framework. 
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
void pairLagrangianForcesSummary(const Ref<const Array<T, 2, 1> >& r1,
                                 const Ref<const Array<T, 2, 1> >& n1,
                                 const Ref<const Array<T, 2, 1> >& dr1, 
                                 const Ref<const Array<T, 2, 1> >& dn1,
                                 const T half_l1, 
                                 const Ref<const Array<T, 2, 1> >& r2,
                                 const Ref<const Array<T, 2, 1> >& n2,
                                 const Ref<const Array<T, 2, 1> >& dr2, 
                                 const Ref<const Array<T, 2, 1> >& dn2,
                                 const T half_l2,  
                                 const Ref<const Array<T, 2, 1> >& d12,
                                 const T s, const T t,
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
 * Output an error message pertaining to the given cell and force exerted 
 * upon it. 
 *
 * @param r Cell center.
 * @param n Cell orientation.
 * @param half_l Cell half-length.
 * @param force Force on input cell. 
 * @param torque Torque on input cell. 
 */
template <typename T>
void cellNewtonianForceSummary(const Ref<const Array<T, 2, 1> >& r,
                               const Ref<const Array<T, 2, 1> >& n,
                               const Ref<const Array<T, 2, 1> >& dr, 
                               const Ref<const Array<T, 2, 1> >& dn,
                               const T half_l,
                               const Ref<const Array<T, 2, 1> >& force,
                               const T torque)
{
    std::cerr << std::setprecision(10)
              << "Cell center = (" << r(0) << ", " << r(1) << ")" << std::endl
              << "Cell orientation = (" << n(0) << ", " << n(1) << ")" << std::endl
              << "Cell translational velocity = (" << dr(0) << ", " << dr(1) << ")" << std::endl
              << "Cell orientational velocity = (" << dn(0) << ", " << dn(1) << ")" << std::endl
              << "Cell half-length = " << half_l << std::endl
              << "Force = (" << force(0) << ", " << force(1) << ")" << std::endl
              << "Torque = " << torque << std::endl; 
}

/**
 * Output an error message pertaining to the given cell-cell configuration
 * and the force on cell 2 due to cell 1.
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
 * @param force Force vector from cell 1 to cell 2. 
 */
template <typename T>
void pairNewtonianForceSummary(const Ref<const Array<T, 2, 1> >& r1,
                               const Ref<const Array<T, 2, 1> >& n1,
                               const Ref<const Array<T, 2, 1> >& dr1, 
                               const Ref<const Array<T, 2, 1> >& dn1,
                               const T half_l1, 
                               const Ref<const Array<T, 2, 1> >& r2,
                               const Ref<const Array<T, 2, 1> >& n2,
                               const Ref<const Array<T, 2, 1> >& dr2, 
                               const Ref<const Array<T, 2, 1> >& dn2,
                               const T half_l2,  
                               const Ref<const Array<T, 2, 1> >& d12,
                               const T s, const T t,
                               const Ref<const Array<T, 2, 1> >& force)
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
              << "Force = (" << force(0) << ", " << force(1) << ")" << std::endl; 
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
                    static_cast<int>(cells(j, __colidx_id)),
                    cells(j, __colseq_r).matrix(),
                    cells(j, __colseq_n).matrix(),
                    cells(j, __colidx_half_l),
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
            static_cast<int>(cells(j, __colidx_id)), 
            cells(j, __colseq_r).matrix(),
            cells(j, __colseq_n).matrix(),
            cells(j, __colidx_half_l),
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

/* -------------------------------------------------------------------- // 
//                    LAGRANGIAN GENERALIZED FORCES                     // 
// -------------------------------------------------------------------- */ 

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
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param prefactors Array of three pre-computed prefactors, namely
 *                   `2.5 * E0 * sqrt(R)`,
 *                   `2.5 * E0 * sqrt(R) * pow(2 * R - 2 * Rcell, 1.5)`, and
 *                   `2.5 * Ecell * sqrt(Rcell)`.
 * @returns Derivatives of the cell-cell interaction energies with respect 
 *          to cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 4> cellCellRepulsiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                             const T dt, const int iter,
                                             const T R, const T Rcell,
                                             const Ref<const Array<T, 3, 1> >& prefactors)
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
    //     prefactors(0) = 2.5 * E0 * std::sqrt(R)
    //     prefactors(1) = 2.5 * E0 * std::sqrt(R) * std::pow(2 * R - 2 * Rcell, 1.5)
    //     prefactors(2) = 2.5 * Ecell * std::sqrt(Rcell)

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
        // Case 1: the overlap is positive but less than 2 * (R - Rcell) (i.e.,
        // it is limited to within the EPS coating)
        T prefactor = 0; 
        if (overlap > 0 && overlap < 2 * (R - Rcell))
        {
            // The force magnitude is 2.5 * E0 * sqrt(R) * pow(overlap, 1.5)
            prefactor = prefactors(0) * pow(overlap, 1.5); 
        }
        // Case 2: the overlap is instead greater than 2 * R - 2 * Rcell
        // (i.e., it encroaches into the bodies of the two cells)
        else if (overlap >= 2 * (R - Rcell))
        {
            // The force magnitude is 2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5)
            // + 2.5 * Ecell * sqrt(Rcell) * pow(2 * Rcell + overlap - 2 * R, 1.5) 
            T term = prefactors(2) * pow(overlap - 2 * (R - Rcell), 1.5);
            prefactor = prefactors(1) + term;
        }

        if (overlap > 0)
        {
            // Use formulas from You et al. (2018, 2019, 2021), which allows
            // for omitting the Lagrange multiplier 
            Array<T, 2, 1> vij = prefactor * dir_ij;
            Array<T, 2, 1> ni = cells(i, __colseq_n); 
            Array<T, 2, 1> nj = cells(j, __colseq_n);
            T wi = ni.matrix().dot(vij.matrix()); 
            T wj = nj.matrix().dot(vij.matrix());  
            Array<T, 2, 4> forces;
            forces << vij(0),                      vij(1),
                      si * (-wi * ni(0) + vij(0)), si * (-wi * ni(1) + vij(1)), 
                      -vij(0),                     -vij(1),
                      sj * (wj * nj(0) - vij(0)),  sj * (wj * nj(1) - vij(1));
            #ifdef DEBUG_CHECK_REPULSIVE_FORCES_NAN
                if (forces.isNaN().any() || forces.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in repulsive forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    pairLagrangianForcesSummary<T>(
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
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param mode Choice of potential used to model cell-cell adhesion. Can be
 *             NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param params Parameters required to compute cell-cell adhesion forces. 
 * @returns Derivatives of the cell-cell adhesion energies with respect to  
 *          cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 4> cellCellAdhesiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                            const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                            const T dt, const int iter,
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
            if (mode == AdhesionMode::JKR)
            {
                const T strength = params["strength"];
                const T dmin = params["mindist"];
                // Enforce orientation vector norm constraint
                forces = strength * forcesJKRLagrange<T, 2>(
                    ni, nj, dij, R, si, sj, dmin, true
                ); 
            }
            else if (mode == AdhesionMode::KIHARA) 
            {
                const T strength = params["strength"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                // Enforce orientation vector norm constraint 
                forces = strength * forcesKiharaLagrange<T, 2>(
                    ni, nj, dij, R, si, sj, expd, dmin, true
                );
            }
            else if (mode == AdhesionMode::GBK)
            {
                const T strength = params["strength"];
                const T exp1 = params["anisotropy_exp1"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                // Enforce orientation vector norm constraint 
                forces = strength * forcesGBKLagrange<T, 2>(
                    ri, ni, half_li, rj, nj, half_lj, R, Rcell, dij, si, sj,
                    expd, exp1, dmin, true
                ); 
            }
            #ifdef DEBUG_CHECK_ADHESIVE_FORCES_NAN
                if (forces.isNaN().any() || forces.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in adhesive forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    pairLagrangianForcesSummary<T>(
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
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param eta Array of cell-cell friction coefficients between cells of 
 *            different groups. 
 * @returns Derivatives of the dissipation due to cell-cell tangential friction
 *          for each pair of neighboring cells. 
 */
template <typename T>
Array<T, Dynamic, 4> cellCellFrictionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                            const T dt, const int iter, const T R,
                                            const Ref<const Array<T, Dynamic, Dynamic> >& eta)
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

            // Determine the cell-cell friction coefficient 
            int gi = static_cast<int>(cells(i, __colidx_group));
            int gj = static_cast<int>(cells(j, __colidx_group));
            T eta_ij = (gi < gj ? eta(gi - 1, gj - 1) : eta(gj - 1, gi - 1)); 

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
            dPdq_ij(0, Eigen::seq(0, 1)) = eta_ij * sqrt_overlap * vijt;
            dPdq_ij(1, Eigen::seq(0, 1)) = -dPdq_ij(0, Eigen::seq(0, 1)); 

            // Derivatives of the dissipation w.r.t the orientational velocities 
            // of cell i and cell j
            Array<T, 2, 1> wi, wj;
            wi << ui(1) / ni(1), ui(0) / ni(0); 
            wj << uj(1) / nj(1), uj(0) / nj(0); 
            dPdq_ij(0, Eigen::seq(2, 3)) = eta_ij * sqrt_overlap * (wi * vijt);
            dPdq_ij(1, Eigen::seq(2, 3)) = -eta_ij * sqrt_overlap * (wj * vijt);

            #ifdef DEBUG_CHECK_FRICTION_FORCES_NAN
                if (dPdq_ij.isNaN().any() || dPdq_ij.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in friction forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    pairLagrangianForcesSummary<T>(
                        ri, ni, dri, dni, cells(i, __colidx_half_l),
                        rj, nj, drj, dnj, cells(j, __colidx_half_l), 
                        dij.array(), si, sj, dPdq_ij
                    );
                    std::cerr << "Cell-cell friction coefficient = " << eta_ij << std::endl;
                    std::cerr << "Contact point = (" << contact(0) << ", "
                              << contact(1) << ")" << std::endl; 
                    std::cerr << "Cell 1 angular velocity = " << angvel_i << std::endl; 
                    std::cerr << "Cell 2 angular velocity = " << angvel_j << std::endl; 
                    std::cerr << "Relative velocity = (" << vij(0) << ", "
                              << vij(1) << ")" << std::endl;
                    std::cerr << "Relative normal velocity = (" << vijn(0) << ", "
                              << vijn(1) << ")" << std::endl; 
                    std::cerr << "Relative tangential velocity = (" << vijt(0) << ", "
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

/* -------------------------------------------------------------------- // 
//                     NEWTONIAN FORCES AND TORQUES                     // 
// -------------------------------------------------------------------- */

/**
 * Get the masses of the given cells. 
 *
 * @param cells Current population of cells. 
 * @param density Constant density of each cell, including the EPS. 
 * @param R Cell radius, including the EPS. 
 * @returns Array of masses. 
 */
template <typename T>
Array<T, Dynamic, 1> getMasses(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                               const T density, const T R)
{
    Array<T, Dynamic, 1> masses(cells.rows());
    const T R2 = R * R;
    const T R3 = R2 * R; 
    const T cap_volume = 4.0 * R3 / 3.0;
    for (int i = 0; i < cells.rows(); ++i)
        masses(i) = cap_volume + R2 * cells(i, __colidx_l);

    return density * boost::math::constants::pi<T>() * masses; 
}

/**
 * Get the moments of inertia of the given cells.
 *
 * Column 0 gives the moment of inertia along each cell's orientation vector. 
 * Column 1 gives the moment of inertia along any vector orthogonal to each 
 * cell's orientation vector.
 *
 * These formulas were taken from Warren et al. eLife 2019. 
 *
 * @param cells Current population of cells.
 * @param density Constant density of each cell, including the EPS. 
 * @param R Cell radius, including the EPS. 
 * @returns Array of moments of inertia. 
 */
template <typename T>
Array<T, Dynamic, 2> getMomentsOfInertia(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                         const T density, const T R)
{
    Array<T, Dynamic, 2> moments(cells.rows(), 2);
    const T D = 2 * R;
    const T D2 = D * D; 
    const T D3 = D2 * D; 
    const T D4 = D3 * D; 
    for (int i = 0; i < cells.rows(); ++i)
    {
        T l = cells(i, __colidx_l); 
        T l2 = l * l;
        T l3 = l2 * l; 
        moments(i, 0) = D4 * (l / 32.0 + D / 60.0);
        moments(i, 1) = D2 * (4 * D3 + 15 * l * D2 + 20 * l2 * D + 10 * l3) / 480.0;
    }

    return boost::math::constants::pi<T>() * density * moments; 
}

/**
 * Given an array of forces in two dimensions, the cell positions, and the 
 * point within each cell at which each force is exerted, compute the 
 * corresponding torques.
 *
 * @param cells Current population of cells. 
 * @param forces Array of cell-cell forces. 
 * @param idx Array of interacting cell indices (each row i, j entails a 
 *            force on cell j by cell i). 
 * @param points Array of points at which each force is exerted.
 * @returns Corresponding array of torques. 
 */
template <typename T>
Array<T, Dynamic, 1> getTorques(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                const Ref<const Array<T, Dynamic, 2> >& forces,
                                const Ref<const Array<int, Dynamic, 2> >& idx, 
                                const Ref<const Array<T, Dynamic, 2> >& points)
{
    const int nforces = forces.rows(); 
    Array<T, Dynamic, 1> torques(nforces); 
    for (int i = 0; i < nforces; ++i)
    {
        int j = idx(i, 1);    // Get index of cell on which the force is exerted 
        Array<T, 2, 1> a = points.row(i) - cells(j, __colseq_r); 
        Array<T, 2, 1> b = forces.row(i); 
        torques(i) = a(0) * b(1) - a(1) * b(0); 
    }

    return torques; 
}

/**
 * Compute the cell-cell repulsive forces for each pair of neighboring cells.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed.
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param prefactors Array of three pre-computed prefactors, namely
 *                   `2.5 * E0 * sqrt(R)`,
 *                   `2.5 * E0 * sqrt(R) * pow(2 * R - 2 * Rcell, 1.5)`, and
 *                   `2.5 * Ecell * sqrt(Rcell)`.
 * @returns Forces and torques due to cell-cell repulsion for each pair of 
 *          neighboring cells. 
 */
template <typename T>
Array<T, Dynamic, 3> cellCellRepulsiveForcesNewton(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                   const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                                   const T dt, const int iter,
                                                   const T R, const T Rcell,
                                                   const Ref<const Array<T, 3, 1> >& prefactors)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 3>::Zero(n, 3); 

    // Compute distance vector magnitude, direction, and corresponding
    // cell-cell overlap for every pair of neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array(); 
    Array<T, Dynamic, 2> directions = neighbors(Eigen::all, Eigen::seq(2, 3)).colwise() / magnitudes;
    Array<T, Dynamic, 1> overlaps = 2 * R - magnitudes;  

    // Note that:
    //     prefactors(0) = 2.5 * E0 * std::sqrt(R)
    //     prefactors(1) = 2.5 * E0 * std::sqrt(R) * std::pow(2 * R - 2 * Rcell, 1.5)
    //     prefactors(2) = 2.5 * Ecell * std::sqrt(Rcell)

    // For each pair of neighboring cells ...
    Array<T, Dynamic, 2> forces(0, 2); 
    Array<T, Dynamic, 2> points(0, 2); 
    Array<int, Dynamic, 2> idx(0, 2);
    int nforces = 0; 
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
        // Case 1: the overlap is positive but less than 2 * (R - Rcell) (i.e., it 
        // is limited to within the EPS coating)
        T prefactor = 0; 
        if (overlap > 0 && overlap < 2 * (R - Rcell))
        {
            // The force magnitude is 2.5 * E0 * sqrt(R) * pow(overlap, 1.5)
            prefactor = prefactors(0) * pow(overlap, 1.5); 
        }
        // Case 2: the overlap is instead greater than 2 * R - 2 * Rcell
        // (i.e., it encroaches into the bodies of the two cells)
        else if (overlap >= 2 * (R - Rcell))
        {
            // The force magnitude is 2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5)
            // + 2.5 * Ecell * sqrt(Rcell) * pow(2 * Rcell + overlap - 2 * R, 1.5) 
            T term = prefactors(2) * pow(overlap - 2 * (R - Rcell), 1.5);
            prefactor = prefactors(1) + term;
        }

        if (overlap > 0)
        {
            // Compute the forces exerted on cell j by cell i and vice versa 
            Array<T, 2, 1> vij = prefactor * dir_ij;
            nforces += 2; 
            forces.conservativeResize(nforces, 2); 
            points.conservativeResize(nforces, 2); 
            idx.conservativeResize(nforces, 2); 
            forces.row(nforces - 2) = vij; 
            forces.row(nforces - 1) = -vij;
            idx(nforces - 2, 0) = i; 
            idx(nforces - 2, 1) = j; 
            idx(nforces - 1, 0) = j; 
            idx(nforces - 1, 1) = i; 

            // Compute the contact point between cell i and cell j
            //
            // TODO Is this correct? Should this be the centerline point? 
            Array<T, 2, 1> ri = cells(i, __colseq_r) + si * cells(i, __colseq_n); 
            Array<T, 2, 1> dij = neighbors(k, Eigen::seq(2, 3));
            Array<T, 2, 1> contact = ri + 0.5 * dij; 
            points.row(nforces - 2) = contact;
            points.row(nforces - 1) = contact;

            #ifdef DEBUG_CHECK_REPULSIVE_FORCES_NAN
                if (vij.isNaN().any() || vij.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in repulsive forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl;
                    Array<T, 2, 1> force = forces.row(nforces - 2);   // i -> j 
                    pairNewtonianForceSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colseq_dr), cells(i, __colseq_dn), 
                        cells(i, __colidx_half_l),
                        cells(j, __colseq_r), cells(j, __colseq_n),
                        cells(j, __colseq_dr), cells(j, __colseq_dn), 
                        cells(j, __colidx_half_l),
                        neighbors(k, Eigen::seq(2, 3)), si, sj, force 
                    );
                    throw std::runtime_error("Found nan in repulsive forces"); 
                }
            #endif
        }
    }
    
    // Get corresponding torques 
    Array<T, Dynamic, 1> torques = getTorques<T>(cells, forces, idx, points);

    // Sum all forces and torques exerted on each cell 
    Array<T, Dynamic, 3> forces_total = Array<T, Dynamic, 3>::Zero(n, 3); 
    for (int i = 0; i < nforces; ++i)
    {
        forces_total(idx(i, 1), Eigen::seq(0, 1)) += forces.row(i); 
        forces_total(idx(i, 1), 2) += torques(i); 
    }

    return forces_total; 
}

/**
 * Compute the cell-cell adhesive forces for each pair of neighboring cells.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param mode Choice of potential used to model cell-cell adhesion. Can be
 *             NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param params Parameters required to compute cell-cell adhesion forces. 
 * @returns Forces and torques due to cell-cell adhesion for each pair of 
 *          neighboring cells. 
 */
template <typename T>
Array<T, Dynamic, 3> cellCellAdhesiveForcesNewton(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                  const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                                  const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                                  const T dt, const int iter,
                                                  const T R, const T Rcell, 
                                                  const AdhesionMode mode, 
                                                  std::unordered_map<std::string, T>& params)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 3>::Zero(n, 3); 

    // Compute distance vector magnitude, direction, and corresponding
    // cell-cell overlap for every pair of neighboring cells
    Array<T, Dynamic, 1> magnitudes = neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array(); 
    Array<T, Dynamic, 1> overlaps = 2 * R - magnitudes;  

    // For each pair of neighboring cells ...
    Array<T, Dynamic, 2> forces(0, 2);
    Array<T, Dynamic, 3> torques(0, 3);  
    Array<T, Dynamic, 2> points(0, 2); 
    Array<int, Dynamic, 2> idx(0, 2);
    int nforces = 0; 
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

            // Get the forces and torques on each cell due to the other 
            Array<T, 2, 1> force_ji;
            Array<T, 3, 1> torque_ji, torque_ij;
            // TODO Add JKR forces
            if (mode == AdhesionMode::KIHARA) 
            {
                const T strength = params["strength"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                force_ji = strength * forceKiharaNewton<T, 2>(dij, R, si, sj, expd, dmin);
                Matrix<T, 3, 1> ui, uj; 
                ui << ni(0), ni(1), 0; 
                uj << nj(0), nj(1), 0; 
                torque_ji = (si * ui).cross(force_ji.matrix()).array(); 
                torque_ij = (sj * uj).cross(-force_ji.matrix()).array();
            }
            else if (mode == AdhesionMode::GBK)
            {
                const T strength = params["strength"];
                const T exp1 = params["anisotropy_exp1"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                force_ji = strength * forceGBKNewton<T, 2>(
                    ni, half_li, nj, half_lj, R, Rcell, dij, expd, exp1, dmin
                );
                torque_ji = strength * torqueGBKNewton<T, 2>(
                    ni, half_li, nj, half_lj, R, Rcell, dij, si, sj, expd, exp1, dmin
                );
                torque_ij = strength * torqueGBKNewton<T, 2>(
                    nj, half_lj, ni, half_li, R, Rcell, -dij, sj, si, expd, exp1, dmin
                );                    
            }
            nforces += 2; 
            forces.conservativeResize(nforces, 2);
            torques.conservativeResize(nforces, 3); 
            points.conservativeResize(nforces, 2); 
            idx.conservativeResize(nforces, 2); 
            forces.row(nforces - 2) = -force_ji;
            forces.row(nforces - 1) = force_ji;
            torques.row(nforces - 2) = torque_ij; 
            torques.row(nforces - 1) = torque_ji; 
            idx(nforces - 2, 0) = i; 
            idx(nforces - 2, 1) = j; 
            idx(nforces - 1, 0) = j; 
            idx(nforces - 1, 1) = i;

            #ifdef DEBUG_CHECK_ADHESIVE_FORCES_NAN
                if (force_ji.isNaN().any() || force_ji.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in adhesive forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl;
                    Array<T, 2, 1> force = forces.row(nforces - 2);   // i -> j 
                    pairNewtonianForceSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colseq_dr), cells(i, __colseq_dn), 
                        cells(i, __colidx_half_l),
                        cells(j, __colseq_r), cells(j, __colseq_n),
                        cells(j, __colseq_dr), cells(j, __colseq_dn), 
                        cells(j, __colidx_half_l),
                        neighbors(k, Eigen::seq(2, 3)), si, sj, force 
                    );
                    throw std::runtime_error("Found nan in adhesive forces"); 
                }
            #endif
        }
    }
    
    // Sum all forces and torques exerted on each cell 
    Array<T, Dynamic, 3> forces_total = Array<T, Dynamic, 3>::Zero(n, 3); 
    for (int i = 0; i < nforces; ++i)
    {
        forces_total(idx(i, 1), Eigen::seq(0, 1)) += forces.row(i); 
        forces_total(idx(i, 1), 2) += torques(i, 2); 
    }

    return forces_total;
}

/**
 * Compute the forces due to cell-cell tangential friction for each pair of
 * neighboring cells. 
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed.
 *
 * @param cells Current population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param eta Array of cell-cell friction coefficients between cells of 
 *            different groups. 
 * @returns Forces and torques due to cell-cell tangential friction for each 
 *          pair of neighboring cells. 
 */
template <typename T>
Array<T, Dynamic, 3> cellCellFrictionForcesNewton(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                  const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                                  const T dt, const int iter, const T R,
                                                  const Ref<const Array<T, Dynamic, Dynamic> >& eta)
{
    int n = cells.rows();       // Number of cells
    int m = neighbors.rows();   // Number of neighboring pairs

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 3>::Zero(n, 3);

    // Initialize array of derivatives 
    Array<T, Dynamic, 2> forces(0, 2); 
    Array<T, Dynamic, 2> points(0, 2); 
    Array<int, Dynamic, 2> idx(0, 2);
    int nforces = 0; 
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

            // Determine the cell-cell friction coefficient 
            int gi = static_cast<int>(cells(i, __colidx_group));
            int gj = static_cast<int>(cells(j, __colidx_group));
            T eta_ij = (gi < gj ? eta(gi - 1, gj - 1) : eta(gj - 1, gi - 1)); 

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

            // Compute the force on cell i due to cell j
            //
            // Note that vijt = vij - vijn increases with the velocity of cell i, 
            // and the force on cell i should therefore go in the opposite 
            // direction as vijt 
            Array<T, 2, 1> force_ji = -eta_ij * sqrt_overlap * vijt;

            // Compute the forces exerted on cell i by cell j and vice versa 
            nforces += 2; 
            forces.conservativeResize(nforces, 2); 
            points.conservativeResize(nforces, 2); 
            idx.conservativeResize(nforces, 2); 
            forces.row(nforces - 2) = -force_ji;
            forces.row(nforces - 1) = force_ji;
            points.row(nforces - 2) = contact; 
            points.row(nforces - 1) = contact;
            idx(nforces - 2, 0) = i; 
            idx(nforces - 2, 1) = j; 
            idx(nforces - 1, 0) = j; 
            idx(nforces - 1, 1) = i;

            #ifdef DEBUG_CHECK_FRICTION_FORCES_NAN
                if (force_ji.isNaN().any() || force_ji.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in friction forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    Array<T, 2, 1> force = forces.row(nforces - 2);   // i -> j 
                    pairNewtonianForceSummary<T>(
                        ri, ni, dri, dni, cells(i, __colidx_half_l),
                        rj, nj, drj, dnj, cells(j, __colidx_half_l), 
                        dij.array(), si, sj, force
                    );
                    std::cerr << "Cell-cell friction coefficient = " << eta_ij << std::endl;
                    std::cerr << "Contact point = (" << contact(0) << ", "
                              << contact(1) << ")" << std::endl; 
                    std::cerr << "Cell 1 angular velocity = " << angvel_i << std::endl; 
                    std::cerr << "Cell 2 angular velocity = " << angvel_j << std::endl; 
                    std::cerr << "Relative velocity = (" << vij(0) << ", "
                              << vij(1) << ")" << std::endl;
                    std::cerr << "Relative normal velocity = (" << vijn(0) << ", "
                              << vijn(1) << ")" << std::endl; 
                    std::cerr << "Relative tangential velocity = (" << vijt(0) << ", "
                              << vijt(1) << ")" << std::endl;
                    throw std::runtime_error("Found nan in cell-cell friction forces"); 
                }
            #endif
        }
    }

    // Get corresponding torques 
    Array<T, Dynamic, 1> torques = getTorques<T>(cells, forces, idx, points);

    // Sum all forces and torques exerted on each cell 
    Array<T, Dynamic, 3> forces_total = Array<T, Dynamic, 3>::Zero(n, 3); 
    for (int i = 0; i < nforces; ++i)
    {
        forces_total(idx(i, 1), Eigen::seq(0, 1)) += forces.row(i); 
        forces_total(idx(i, 1), 2) += torques(i); 
    }

    return forces_total;
}

/**
 * Compute the forces due to ambient viscosity and cell-surface friction 
 * for each cell.
 *
 * @param cells Current population of cells.  
 * @param prefactors Viscosity force prefactors computed for each cell 
 *                   in the population.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @returns Forces and torques due to ambient viscosity and cell-surface
 *          friction. 
 */
template <typename T>
Array<T, Dynamic, 3> viscosityForcesNewton(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                           const Ref<const Array<T, Dynamic, 2> >& prefactors,
                                           const T dt, const int iter)
{
    int n = cells.rows();       // Number of cells

    // Compute the viscosity forces, which are simply linear in the velocities
    Array<T, Dynamic, 3> forces(n, 3);
    forces(Eigen::all, Eigen::seq(0, 1)) = -(cells(Eigen::all, __colseq_dr).colwise() * prefactors.col(0));

    // Compute the corresponding torques TODO Check this
    Array<T, Dynamic, 1> angvels(n); 
    for (int i = 0; i < n; ++i)
    {
        if (abs(cells(i, __colidx_nx)) > 1e-2)
            angvels(i) = cells(i, __colidx_dny) / cells(i, __colidx_nx); 
        else 
            angvels(i) = -cells(i, __colidx_dnx) / cells(i, __colidx_ny);
    }
    forces.col(2) = -prefactors.col(1) * angvels;

    #ifdef DEBUG_CHECK_VISCOSITY_FORCES_NAN
        if (forces.isNaN().any() || forces.isInf().any())
        {
            for (int i = 0; i < n; ++i)
            {
                if (forces.row(i).isNaN().any() || forces.row(i).isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in viscosity force for cell "
                              << i << std::endl; 
                    std::cerr << "Timestep: " << dt << std::endl;
                    Array<T, 2, 1> force = forces(i, Eigen::seq(0, 1));
                    T torque = forces(i, 2); 
                    cellNewtonianForceSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colseq_dr), cells(i, __colseq_dn), 
                        cells(i, __colidx_half_l), force, torque  
                    );
                    throw std::runtime_error("Found nan in viscosity forces");
                }
            }
        }
    #endif

    return forces; 
}

/* -------------------------------------------------------------------- // 
//                      ADDITIONAL UTILITY FUNCTIONS                    //
// -------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------- // 
//             VELOCITY UPDATES IN THE LAGRANGIAN FRAMEWORK             //
// -------------------------------------------------------------------- */

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
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output.
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of three pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param noise Noise to be added to each generalized force used to compute
 *              the velocities.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
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
                                   const T dt, const int iter, const T R, const T Rcell,
                                   const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
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
        cells, neighbors, dt, iter, R, Rcell, cell_cell_prefactors
    );

    // ... and the adhesive forces (if present) ... 
    Array<T, Dynamic, 4> dEdq_adhesion = Array<T, Dynamic, 4>::Zero(n, 4); 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        dEdq_adhesion = cellCellAdhesiveForces<T>(
            cells, neighbors, to_adhere, dt, iter, R, Rcell, adhesion_mode,
            adhesion_params
        );
    }

    // ... and the cell-cell friction forces (if present) ...
    /*
    Array<T, Dynamic, 4> dEdq_friction = Array<T, Dynamic, 4>::Zero(n, 4);
    if ((eta_cell_cell > 0).any())
    {
        dEdq_friction = cellCellFrictionForces<T>(
            cells, neighbors, dt, iter, R, eta_cell_cell
        ); 
    }
    */

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
                if (dEdq_confine.row(i).isNaN().any() || dEdq_confine.row(i).isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in confinement forces for cell "
                              << i << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    cellLagrangianForcesSummary<T>(
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
                if (dEdq_confine.row(i).isNaN().any() || dEdq_confine.row(i).isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in confinement forces for cell "
                              << i << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl; 
                    cellLagrangianForcesSummary<T>(
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

    /*
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
    */
    velocities.col(0) = -dEdq.col(0) / K;
    velocities.col(1) = -dEdq.col(1) / K; 
    velocities.col(2) = -dEdq.col(2) / L;
    velocities.col(3) = -dEdq.col(3) / L;

    return velocities;  
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
 * @param cell_cell_prefactors Array of three pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param surface_contact_density Cell-surface contact area density.
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
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
                           const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
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
                std::cerr << "Timestep: " << dt << std::endl; 
                pairConfigSummaryWithVelocities<T>(
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
        cells, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors,
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
            cells_i, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors,
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

/* -------------------------------------------------------------------- // 
//              VELOCITY UPDATES IN THE NEWTONIAN FRAMEWORK             //
// -------------------------------------------------------------------- */

/**
 * Run one step of the velocity Verlet method for the given timestep, using 
 * forces and torques in the Newtonian framework.  
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed, and the cell data are updated in place. 
 *
 * @param cells Current population of cells. This array is overwritten by
 *              this function. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param dt Timestep. 
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of three pre-computed prefactors for 
 *                             cell-cell interaction forces.
 * @param density Constant density of each cell, including the EPS. 
 * @param surface_contact_density Cell-surface contact area density.
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param eta_cell_cell Array of cell-cell friction coefficients between cells 
 *                      in different groups. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2).
 * @param boundary_idx Pre-computed vector of indices of peripheral cells. 
 * @param confine_params Parameters required to compute confinement forces.
 */
template <typename T>
void stepVerlet(Ref<Array<T, Dynamic, Dynamic> > cells,
                const Ref<const Array<T, Dynamic, 6> >& neighbors,
                const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                const T dt, const int iter, const T R, const T Rcell,
                const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
                const T density, const T surface_contact_density,
                const T max_noise, boost::random::mt19937& rng,
                boost::random::uniform_01<>& uniform_dist,
                const Ref<const Array<T, Dynamic, Dynamic> >& eta_cell_cell, 
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
                std::cerr << "Timestep: " << dt << std::endl; 
                pairConfigSummaryWithVelocities<T>(
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

    int n = cells.rows();
    
    // Compute the masses of the cells 
    Array<T, Dynamic, 1> masses = getMasses<T>(cells, density, R);

    // Compute the moments of inertia of the cells 
    Array<T, Dynamic, 2> moments = getMomentsOfInertia<T>(cells, density, R);

    // Sample noise components to be added to the forces and torques
    //
    // TODO Allow for different ranges for forces and for torques 
    Array<T, Dynamic, 3> noise = Array<T, Dynamic, 3>::Zero(n, 3);  
    if (max_noise > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                T r = uniform_dist(rng); 
                noise(i, j) = -max_noise + 2 * max_noise * r; 
            }
        }
    }

    // Compute the repulsive forces and torques ... 
    Array<T, Dynamic, 3> forces_repulsion = cellCellRepulsiveForcesNewton<T>(
        cells, neighbors, dt, iter, R, Rcell, cell_cell_prefactors
    );

    // ... and the adhesive forces and torques (if present) ... 
    Array<T, Dynamic, 3> forces_adhesion = Array<T, Dynamic, 3>::Zero(n, 3); 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        forces_adhesion = cellCellAdhesiveForcesNewton<T>(
            cells, neighbors, to_adhere, dt, iter, R, Rcell, adhesion_mode,
            adhesion_params
        );
    }

    // ... and the cell-cell friction forces and torques (if present) ...
    Array<T, Dynamic, 3> forces_friction = Array<T, Dynamic, 3>::Zero(n, 3);
    if ((eta_cell_cell > 0).any())
    {
        forces_friction = cellCellFrictionForcesNewton<T>(
            cells, neighbors, dt, iter, R, eta_cell_cell
        );
    }

    // ... and the ambient viscosity and cell-surface friction forces ... 
    Array<T, Dynamic, 2> prefactors = compositeViscosityForcePrefactors<T>(
        cells, R, surface_contact_density
    );
    Array<T, Dynamic, 3> forces_viscosity = viscosityForcesNewton<T>(
        cells, prefactors, dt, iter
    );

    // ... and the confinement forces (if present)
    // TODO Implement this!
    /*
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
                    std::cerr << "Timestep: " << dt << std::endl; 
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
                    std::cerr << "Timestep: " << dt << std::endl; 
                    cellForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colidx_half_l),
                        dEdq_confine.row(i).transpose()
                    );
                }
            }
        #endif
    }
    */

    // Combine all the forces (with the noise)
    //
    // TODO Include confinement forces
    Array<T, Dynamic, 3> forces = (
        forces_repulsion + forces_adhesion + forces_friction + forces_viscosity + noise
    );

    // Extract the current cell velocities and angular velocities 
    Array<T, Dynamic, 2> velocities = cells(Eigen::all, __colseq_dr);
    Array<T, Dynamic, 1> angvels(n); 
    for (int i = 0; i < n; ++i)
    {
        if (abs(cells(i, __colidx_nx)) > 1e-2)
            angvels(i) = cells(i, __colidx_dny) / cells(i, __colidx_nx); 
        else 
            angvels(i) = -cells(i, __colidx_dnx) / cells(i, __colidx_ny);
    }

    // Update by a half-timestep ... 
    Array<T, Dynamic, 2> accelerations = forces(Eigen::all, Eigen::seq(0, 1)).colwise() / masses;
    velocities += 0.5 * dt * accelerations;
    angvels += 0.5 * dt * forces.col(2) / moments.col(1);

    // Update cell positions and orientations
    cells(Eigen::all, __colseq_r) += dt * velocities;
    for (int i = 0; i < n; ++i)
    {
        Array<T, 2, 1> cross; 
        cross << -angvels(i) * cells(i, __colidx_ny),
                  angvels(i) * cells(i, __colidx_nx); 
        cells(i, __colseq_n) += dt * cross;
    }
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells);

    // Update cell velocities and angular velocities with the intermediate 
    // values
    cells(Eigen::all, __colseq_dr) = velocities;
    cells.col(__colidx_dnx) = -angvels * cells.col(__colidx_ny);
    cells.col(__colidx_dny) = angvels * cells.col(__colidx_nx);

    // Re-sample noise components to be added to the forces and torques
    //
    // TODO Allow for different ranges for forces and for torques 
    if (max_noise > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                T r = uniform_dist(rng); 
                noise(i, j) = -max_noise + 2 * max_noise * r; 
            }
        }
    }

    // Re-compute the repulsive forces and torques ... 
    forces_repulsion = cellCellRepulsiveForcesNewton<T>(
        cells, neighbors, dt, iter, R, Rcell, cell_cell_prefactors
    );

    // ... and the adhesive forces and torques (if present) ... 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        forces_adhesion = cellCellAdhesiveForcesNewton<T>(
            cells, neighbors, to_adhere, dt, iter, R, Rcell, adhesion_mode,
            adhesion_params
        );
    }

    // ... and the cell-cell friction forces and torques (if present) ...
    if ((eta_cell_cell > 0).any())
    {
        forces_friction = cellCellFrictionForcesNewton<T>(
            cells, neighbors, dt, iter, R, eta_cell_cell
        ); 
    }

    // ... and the ambient viscosity and cell-surface friction forces ...
    forces_viscosity = viscosityForcesNewton<T>(cells, prefactors, dt, iter); 

    // ... and the confinement forces (if present)
    // TODO Implement this!
    /*
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
                    std::cerr << "Timestep: " << dt << std::endl; 
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
                    std::cerr << "Timestep: " << dt << std::endl; 
                    cellForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colidx_half_l),
                        dEdq_confine.row(i).transpose()
                    );
                }
            }
        #endif
    }
    */

    // Combine all the forces (with the noise)
    //
    // TODO Include confinement forces
    forces = (
        forces_repulsion + forces_adhesion + forces_friction + forces_viscosity + noise
    );

    // Update cell velocities and angular velocities with the re-computed forces 
    accelerations = forces(Eigen::all, Eigen::seq(0, 1)).colwise() / masses;
    cells(Eigen::all, __colseq_dr) += 0.5 * dt * accelerations; 
    angvels += 0.5 * dt * (forces.col(2) / moments.col(1)); 
    cells.col(__colidx_dnx) = -angvels * cells.col(__colidx_ny);
    cells.col(__colidx_dny) = angvels * cells.col(__colidx_nx);
}

#endif
