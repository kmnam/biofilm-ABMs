/**
 * Implementations of cell-cell and cell-surface interaction forces.
 *
 * In what follows, a population of N cells is represented as a 2-D array
 * of N rows, whose columns are as specified in `indices.hpp`.
 *
 * Additional features may be included in the array but these are not
 * relevant for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/21/2025
 */

#ifndef BIOFILM_MECHANICS_3D_HPP
#define BIOFILM_MECHANICS_3D_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <iomanip>
#include <omp.h>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "indices.hpp"
#include "integrals.hpp"
#include "distances.hpp"
#include "adhesion.hpp"

using namespace Eigen;

// Expose math functions for both standard and boost MPFR types
using std::pow;
using boost::multiprecision::pow;
using std::abs;
using boost::multiprecision::abs;
using std::isnan;
using boost::multiprecision::isnan; 

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
void cellLagrangianForcesSummary(const Ref<const Array<T, 3, 1> >& r,
                                 const Ref<const Array<T, 3, 1> >& n,
                                 const T half_l,
                                 const Ref<const Array<T, 6, 1> >& dEdq)
{
    std::cerr << std::setprecision(10)
              << "Cell center = (" << r(0) << ", " << r(1) << ", " << r(2) << ")" << std::endl
              << "Cell orientation = (" << n(0) << ", " << n(1) << ", " << n(2) << ")" << std::endl
              << "Cell half-length = " << half_l << std::endl
              << "Generalized forces: " << std::endl
              << " - w.r.t r(0) = " << dEdq(0) << std::endl
              << " - w.r.t r(1) = " << dEdq(1) << std::endl
              << " - w.r.t r(2) = " << dEdq(2) << std::endl  
              << " - w.r.t n(0) = " << dEdq(3) << std::endl
              << " - w.r.t n(1) = " << dEdq(4) << std::endl
              << " - w.r.t n(2) = " << dEdq(5) << std::endl;
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
void pairLagrangianForcesSummary(const Ref<const Array<T, 3, 1> >& r1,
                                 const Ref<const Array<T, 3, 1> >& n1,
                                 const Ref<const Array<T, 3, 1> >& dr1, 
                                 const Ref<const Array<T, 3, 1> >& dn1,
                                 const T half_l1, 
                                 const Ref<const Array<T, 3, 1> >& r2,
                                 const Ref<const Array<T, 3, 1> >& n2,
                                 const Ref<const Array<T, 3, 1> >& dr2, 
                                 const Ref<const Array<T, 3, 1> >& dn2,
                                 const T half_l2,  
                                 const Ref<const Array<T, 3, 1> >& d12,
                                 const T s, const T t,
                                 const Ref<const Array<T, 2, 6> >& dEdq)
{
    std::cerr << std::setprecision(10)
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ", " << r1(2) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ", " << n1(2) << ")" << std::endl
              << "Cell 1 translational velocity = (" << dr1(0) << ", " << dr1(1) << ", " << dr1(2) << ")" << std::endl
              << "Cell 1 orientational velocity = (" << dn1(0) << ", " << dn1(1) << ", " << dn1(2) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ", " << r2(2) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ", " << n2(2) << ")" << std::endl
              << "Cell 2 translational velocity = (" << dr2(0) << ", " << dr2(1) << ", " << dr2(2) << ")" << std::endl
              << "Cell 2 orientational velocity = (" << dn2(0) << ", " << dn2(1) << ", " << dn2(2) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl
              << "Distance vector = (" << d12(0) << ", " << d12(1) << ", " << d12(2) << ")" << std::endl 
              << "Cell-body coordinate of contact point along cell 1 = " << s << std::endl
              << "Cell-body coordinate of contact point along cell 2 = " << t << std::endl 
              << "Generalized forces: " << std::endl
              << " - w.r.t r1(0) = " << dEdq(0, 0) << std::endl
              << " - w.r.t r1(1) = " << dEdq(0, 1) << std::endl
              << " - w.r.t r1(2) = " << dEdq(0, 2) << std::endl 
              << " - w.r.t n1(0) = " << dEdq(0, 3) << std::endl
              << " - w.r.t n1(1) = " << dEdq(0, 4) << std::endl
              << " - w.r.t n1(2) = " << dEdq(0, 5) << std::endl
              << " - w.r.t r2(0) = " << dEdq(1, 0) << std::endl
              << " - w.r.t r2(1) = " << dEdq(1, 1) << std::endl
              << " - w.r.t r2(2) = " << dEdq(1, 2) << std::endl 
              << " - w.r.t n2(0) = " << dEdq(1, 3) << std::endl
              << " - w.r.t n2(1) = " << dEdq(1, 4) << std::endl
              << " - w.r.t n2(2) = " << dEdq(1, 5) << std::endl;
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
void cellNewtonianForceSummary(const Ref<const Array<T, 3, 1> >& r,
                               const Ref<const Array<T, 3, 1> >& n,
                               const Ref<const Array<T, 3, 1> >& dr, 
                               const Ref<const Array<T, 3, 1> >& dn,
                               const T half_l,
                               const Ref<const Array<T, 3, 1> >& force,
                               const Ref<const Array<T, 3, 1> >& torque)
{
    std::cerr << std::setprecision(10)
              << "Cell center = (" << r(0) << ", " << r(1) << ", " << r(2) << ")" << std::endl
              << "Cell orientation = (" << n(0) << ", " << n(1) << ", " << n(2) << ")" << std::endl
              << "Cell translational velocity = (" << dr(0) << ", " << dr(1) << ", " << dr(2) << ")" << std::endl
              << "Cell orientational velocity = (" << dn(0) << ", " << dn(1) << ", " << dn(2) << ")" << std::endl
              << "Cell half-length = " << half_l << std::endl
              << "Force = (" << force(0) << ", " << force(1) << ", " << force(2) << ")" << std::endl
              << "Torque = (" << torque(0) << ", " << torque(1) << ", " << torque(2) << ")" << std::endl;
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
void pairNewtonianForceSummary(const Ref<const Array<T, 3, 1> >& r1,
                               const Ref<const Array<T, 3, 1> >& n1,
                               const Ref<const Array<T, 3, 1> >& dr1, 
                               const Ref<const Array<T, 3, 1> >& dn1,
                               const T half_l1, 
                               const Ref<const Array<T, 3, 1> >& r2,
                               const Ref<const Array<T, 3, 1> >& n2,
                               const Ref<const Array<T, 3, 1> >& dr2, 
                               const Ref<const Array<T, 3, 1> >& dn2,
                               const T half_l2,  
                               const Ref<const Array<T, 3, 1> >& d12,
                               const T s, const T t,
                               const Ref<const Array<T, 3, 1> >& force)
{
    std::cerr << std::setprecision(10)
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ", " << r1(2) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ", " << n1(2) << ")" << std::endl
              << "Cell 1 translational velocity = (" << dr1(0) << ", " << dr1(1) << ", " << dr1(2) << ")" << std::endl
              << "Cell 1 orientational velocity = (" << dn1(0) << ", " << dn1(1) << ", " << dn1(2) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ", " << r2(2) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ", " << n2(2) << ")" << std::endl
              << "Cell 2 translational velocity = (" << dr2(0) << ", " << dr2(1) << ", " << dr2(2) << ")" << std::endl
              << "Cell 2 orientational velocity = (" << dn2(0) << ", " << dn2(1) << ", " << dn2(2) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl
              << "Distance vector = (" << d12(0) << ", " << d12(1) << ", " << d12(2) << ")" << std::endl 
              << "Cell-body coordinate of contact point along cell 1 = " << s << std::endl
              << "Cell-body coordinate of contact point along cell 2 = " << t << std::endl 
              << "Force = (" << force(0) << ", " << force(1) << ", " << force(2) << ")" << std::endl; 
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

    // Instantiate kernel to be passed into distBetweenCells()
    K kernel;

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
        Matrix<T, 3, 1> dist_ij = std::get<0>(result); 
        T si = std::get<1>(result);
        T sj = std::get<2>(result);
        neighbors(k, Eigen::seq(2, 4)) = dist_ij.array(); 
        neighbors(k, 5) = si; 
        neighbors(k, 6) = sj;
    }
}

/* -------------------------------------------------------------------- // 
//                    LAGRANGIAN GENERALIZED FORCES                     // 
// -------------------------------------------------------------------- */ 

/**
 * Compute the derivatives of the cell-surface repulsion energy for each cell
 * with respect to the cell's z-position and z-orientation.
 *
 * @param cells Existing population of cells.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param ss Cell-body coordinates at which each cell-surface overlap is zero. 
 * @param R Cell radius.
 * @param E0 Elastic modulus of EPS.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero.
 * @param multithread If true, use multithreading. 
 * @returns Derivatives of the cell-surface repulsion energies with respect to
 *          cell z-positions and z-orientations.   
 */
template <typename T>
Array<T, Dynamic, 2> cellSurfaceRepulsionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                const T dt, const int iter,
                                                const Ref<const Array<T, Dynamic, 1> >& ss,
                                                const T R, const T E0,
                                                const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                                const bool multithread)
{
    Array<T, Dynamic, 2> dEdq = Array<T, Dynamic, 2>::Zero(cells.rows(), 2); 

    // For each cell ...
    const T prefactor0 = 2 * E0;
    const T prefactor1 = (8. / 3.) * E0 * pow(R, 0.5); 
    const T prefactor2 = 2 * E0 * pow(R, 0.5);
    #pragma omp parallel for if(multithread)
    for (int i = 0; i < cells.rows(); ++i)
    {
        // If the z-coordinate of the cell's orientation is near zero ... 
        if (assume_2d(i))
        {
            // In this case, dEdq(i, 1) is always zero and dEdq(i, 0) is
            // nonzero only if phi > 0
            T phi = R - cells(i, __colidx_rz);
            if (phi > 0)
                dEdq(i, 0) = -prefactor0 * phi * cells(i, __colidx_l);
        }
        // Otherwise ...
        else
        {
            // Compute the derivative of the cell-surface repulsion energy 
            // with respect to z-position
            T nz2 = cells(i, __colidx_nz) * cells(i, __colidx_nz);
            T int1 = integral1<T>(    // Integral of \delta_i(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 1.0, ss(i)
            );
            T int2 = integral1<T>(    // Integral of \sqrt{\delta_i(s)}
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 0.5, ss(i)
            );
            dEdq(i, 0) = -prefactor0 * ((1.0 - nz2) * int1 + sqrt(R) * nz2 * int2);

            // Compute the derivative of the cell-surface repulsion energy 
            // with respect to z-orientation
            T int3 = integral1<T>(    // Integral of \delta_i^2(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 2.0, ss(i)
            );
            T int4 = integral2<T>(    // Integral of s * \delta_i(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 1.0, ss(i)
            );
            T int5 = integral1<T>(    // Integral of \delta_i^{3/2}(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 1.5, ss(i)
            );
            T int6 = integral2<T>(    // Integral of s * \sqrt{\delta_i(s)}
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 0.5, ss(i)
            );
            dEdq(i, 1) -= prefactor0 * cells(i, __colidx_nz) * int3;
            dEdq(i, 1) -= prefactor0 * (1 - nz2) * int4;
            dEdq(i, 1) += prefactor1 * cells(i, __colidx_nz) * int5;
            dEdq(i, 1) -= prefactor2 * nz2 * int6;
        }
    }

    #ifdef DEBUG_CHECK_CELL_SURFACE_REPULSION_FORCES_NAN
        for (int i = 0; i < cells.rows(); ++i)
        {
            if (dEdq.row(i).isNaN().any() || dEdq.row(i).isInf().any())
            {
                std::cerr << "Iteration " << iter
                          << ": Found nan in repulsive forces between cell " 
                          << i << " and surface" << std::endl;
                std::cerr << "Timestep: " << dt << std::endl;
                std::cerr << "2D assumption: " << assume_2d(i) << std::endl; 
                Array<T, 6, 1> dEdq_extended; 
                dEdq_extended << 0, 0, dEdq(i, 0), 0, 0, dEdq(i, 1);  
                cellLagrangianForcesSummary<T>(
                    cells(i, __colseq_r), cells(i, __colseq_n),
                    cells(i, __colidx_half_l), dEdq_extended
                ); 
                throw std::runtime_error("Found nan in cell-surface repulsive forces"); 
            }
        }
    #endif

    return dEdq;
}

/**
 * Compute the derivatives of the cell-surface adhesion energy for each cell 
 * with respect to the cell's z-position and z-orientation. 
 *
 * @param cells Existing population of cells.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param ss Cell-body coordinates at which each cell-surface overlap is zero. 
 * @param R Cell radius.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero.
 * @param multithread If true, use multithreading.  
 * @returns Derivatives of the cell-surface adhesion energies with respect to
 *          cell z-positions and z-orientations.   
 */
template <typename T>
Array<T, Dynamic, 2> cellSurfaceAdhesionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const T dt, const int iter,
                                               const Ref<const Array<T, Dynamic, 1> >& ss,
                                               const T R,
                                               const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                               const bool multithread)
{
    Array<T, Dynamic, 2> dEdq = Array<T, Dynamic, 2>::Zero(cells.rows(), 2);

    // For each cell ...
    const T prefactor0 = pow(R, 0.5) / 2;
    const T prefactor1 = 2 * boost::math::constants::pi<T>() * R;
    const T prefactor2 = 2 * pow(R, 0.5);
    #pragma omp parallel for if(multithread)
    for (int i = 0; i < cells.rows(); ++i)
    {
        // If the z-coordinate of the cell's orientation is zero ... 
        if (assume_2d(i))
        {
            // In this case, dEdq(i, 1) is always zero and dEdq(i, 0) is
            // nonzero only if phi > 0
            T phi = R - cells(i, __colidx_rz);
            if (phi > 0)
                dEdq(i, 0) = cells(i, __colidx_sigma0) * prefactor0 * cells(i, __colidx_l) / pow(phi, 0.5);
        }
        // Otherwise ... 
        else
        {
            // Compute the derivative of the cell-surface adhesion energy 
            // with respect to z-position
            T nz2 = cells(i, __colidx_nz) * cells(i, __colidx_nz);
            T int1 = integral1<T>(    // Integral of \delta_i^{-1/2}(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), -0.5, ss(i)
            );
            T term2 = 0.0;
            if (abs(ss(i)) < cells(i, __colidx_half_l))
                term2 = (prefactor1 / 2) * cells(i, __colidx_nz);
            dEdq(i, 0) = prefactor0 * (1 - nz2) * int1 + term2;
            dEdq(i, 0) *= cells(i, __colidx_sigma0);

            // Compute the derivative of the cell-surface adhesion energy
            // with respect to z-orientation
            T int2 = integral1<T>(    // Integral of \delta_i^{1/2}(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), 0.5, ss(i)
            );
            T int3 = integral2<T>(    // Integral of s * \delta_i^{-1/2}(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), -0.5, ss(i)
            );
            T int4 = integral4<T>(    // Integral of \Theta(\delta_i(s))
                cells(i, __colidx_nz), cells(i, __colidx_half_l), ss(i)
            );
            T term4 = 0.0;
            if (abs(ss(i)) < cells(i, __colidx_half_l))
                term4 = (prefactor1 / 2) * (R - cells(i, __colidx_rz));
            dEdq(i, 1) += prefactor2 * cells(i, __colidx_nz) * int2; 
            dEdq(i, 1) += prefactor0 * (1 - nz2) * int3;
            dEdq(i, 1) -= prefactor1 * cells(i, __colidx_nz) * int4;
            dEdq(i, 1) += term4;
            dEdq(i, 1) *= cells(i, __colidx_sigma0);
        }
    }

    #ifdef DEBUG_CHECK_CELL_SURFACE_ADHESION_FORCES_NAN
        for (int i = 0; i < cells.rows(); ++i)
        {
            if (dEdq.row(i).isNaN().any() || dEdq.row(i).isInf().any())
            {
                std::cerr << "Iteration " << iter
                          << ": Found nan in adhesive forces between cell " 
                          << i << " and surface" << std::endl;
                std::cerr << "Timestep: " << dt << std::endl;
                std::cerr << "2D assumption: " << assume_2d(i) << std::endl; 
                Array<T, 6, 1> dEdq_extended; 
                dEdq_extended << 0, 0, dEdq(i, 0), 0, 0, dEdq(i, 1);  
                cellLagrangianForcesSummary<T>(
                    cells(i, __colseq_r), cells(i, __colseq_n),
                    cells(i, __colidx_half_l), dEdq_extended
                ); 
                throw std::runtime_error("Found nan in cell-surface adhesive forces"); 
            }
        }
    #endif

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
 * This function assumes that the cell does overlap with the surface, and 
 * that its z-orientation is nonzero. 
 *
 * @param rz z-position of given cell.
 * @param nz z-orientation of given cell.
 * @param l Length of given cell.
 * @param half_l Half-length of given cell.
 * @param ss Cell-body coordinate at which cell-surface overlap is zero.
 * @param eta0 Ambient viscosity of given cell.
 * @param eta1 Surface friction coefficient of given cell.
 * @param R Cell radius.
 * @param idx Cell index. Only used for debugging output. 
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @returns The 6x6 matrix defined above for the given cell. 
 */
template <typename T>
Array<T, 6, 6> compositeViscosityForceMatrix(const T rz, const T nz,
                                             const T l, const T half_l,
                                             const T ss, const T eta0,
                                             const T eta1, const T R,
                                             const int idx, const T dt,
                                             const int iter)
{
    Array<T, 6, 6> M = Array<T, 6, 6>::Zero(6, 6);

    #ifdef DEBUG_CHECK_CELL_ZORIENTATIONS_NONZERO
        if (isnan(ss) || nz <= 0)
            throw std::runtime_error(
                "Composite viscosity matrix function assumes positive cell z-orientation"
            ); 
    #endif
    #ifdef DEBUG_CHECK_CELL_SURFACE_OVERLAP_NONZERO
        if (abs(ss) >= half_l)
            throw std::runtime_error(
                "Composite viscosity matrix function assumes positive cell-surface overlap"
            ); 
    #endif
    
    T term1 = eta0 * l;
    T term2 = eta0 * l * l * l / 12;
    std::tuple<T, T, T> integrals = areaIntegrals<T>(rz, nz, R, half_l, ss); 
    T term3 = (eta1 / R) * std::get<0>(integrals);
    T term4 = (eta1 / R) * std::get<1>(integrals);
    T term5 = (eta1 / R) * std::get<2>(integrals);
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

    #ifdef DEBUG_CHECK_VISCOSITY_FORCES_NAN
        if (M.isNaN().any() || M.isInf().any())
        {
            std::cerr << "Iteration " << iter
                      << ": Found nan in viscosity forces for cell " << idx << std::endl; 
            std::cerr << "Timestep: " << dt << std::endl;
            std::cerr << std::setprecision(10)
                      << "Cell center z-position = " << rz << std::endl
                      << "Cell z-orientation = " << nz << std::endl 
                      << "Cell half-length = " << half_l << std::endl
                      << "Generalized ambient viscosity force prefactors: " << std::endl
                      << " - w.r.t dr(0), dr(1), dr(2) = " << term1 << std::endl
                      << " - w.r.t dn(0), dn(1), dr(2) = " << term2 << std::endl
                      << "Generalized cell-surface friction force integrals: " << std::endl
                      << " - integral 1 = " << term3 << std::endl
                      << " - integral 2 = " << term4 << std::endl
                      << " - integral 3 = " << term5 << std::endl; 
            throw std::runtime_error("Found nan in viscosity forces");
        }
    #endif

    return M;
}

/**
 * Compute the derivatives of the cell-cell repulsion energies for each 
 * cell with respect to the cell's position and orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Existing population of cells.
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
 * @returns Derivatives of the cell-cell repulsion energies with respect to
 *          cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellCellRepulsiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                             const T dt, const int iter,
                                             const T R, const T Rcell,
                                             const Ref<const Array<T, 3, 1> >& prefactors)
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
    //     prefactors(0) = 2.5 * E0 * std::sqrt(R)
    //     prefactors(1) = 2.5 * E0 * std::sqrt(R) * std::pow(2 * R - 2 * Rcell, 1.5)
    //     prefactors(2) = 2.5 * Ecell * std::sqrt(Rcell)

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
            // Compute the derivatives of the cell-cell repulsion energy
            // between cells i and j
            Array<T, 3, 1> vij = prefactor * dir_ij;
            Array<T, 2, 6> forces;
            forces(0, Eigen::seq(0, 2)) = vij; 
            forces(0, Eigen::seq(3, 5)) = si * vij;
            forces(1, Eigen::seq(0, 2)) = -vij; 
            forces(1, Eigen::seq(3, 5)) = -sj * vij;  
            #ifdef DEBUG_CHECK_CELL_CELL_REPULSIVE_FORCES_NAN
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
                        neighbors(k, Eigen::seq(2, 4)), si, sj, 
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
Array<T, Dynamic, 6> cellCellAdhesiveForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                            const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                            const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                            const T dt, const int iter,
                                            const T R, const T Rcell, 
                                            const T E0, const AdhesionMode mode, 
                                            std::unordered_map<std::string, T>& params)
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
            Matrix<T, 3, 1> ri = cells(i, __colseq_r).matrix();
            Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
            Matrix<T, 3, 1> rj = cells(j, __colseq_r).matrix();
            Matrix<T, 3, 1> nj = cells(j, __colseq_n).matrix();
            T half_li = cells(i, __colidx_half_l);
            T half_lj = cells(j, __colidx_half_l);
            Matrix<T, 3, 1> dij = neighbors(k, Eigen::seq(2, 4)).matrix();
            T si = neighbors(k, 5); 
            T sj = neighbors(k, 6); 

            // Get the corresponding forces
            Array<T, 2, 6> forces;
            if (mode == AdhesionMode::JKR)
            {
                const T gamma = params["gamma"]; 
                // Enforce orientation vector norm constraint
                forces = forcesIsotropicJKRLagrange<T, 3>(
                    ni, nj, dij, R, E0, gamma, si, sj, true
                ); 
            }
            else if (mode == AdhesionMode::KIHARA) 
            {
                const T strength = params["strength"];
                const T expd = params["distance_exp"]; 
                const T dmin = params["mindist"];
                // Enforce orientation vector norm constraint 
                forces = strength * forcesKiharaLagrange<T, 3>(
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
                forces = strength * forcesGBKLagrange<T, 3>(
                    ri, ni, half_li, rj, nj, half_lj, R, Rcell, dij, si, sj,
                    expd, exp1, dmin, true
                ); 
            }
            #ifdef DEBUG_CHECK_CELL_CELL_ADHESIVE_FORCES_NAN
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
 * Given the current positions, orientations, and lengths for the given 
 * population of cells, compute the total conservative forces on the cells.  
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
 * @param E0 Elastic modulus of EPS.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param no_surface If true, omit the surface from the simulation. 
 * @param multithread If true, use multithreading. 
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 6> getConservativeForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                           const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                           const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                           const T dt, const int iter,
                                           const T R, const T Rcell,
                                           const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
                                           const T E0,
                                           const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                           const AdhesionMode adhesion_mode,
                                           std::unordered_map<std::string, T>& adhesion_params,
                                           const bool no_surface,
                                           const bool multithread)
{
    int n = cells.rows(); 

    // Get cell-body coordinates at which cell-surface overlap is zero for 
    // each cell
    Array<T, Dynamic, 1> ss(n); 
    for (int i = 0; i < n; ++i)
    {
        if (assume_2d(i))
            ss(i) = std::numeric_limits<T>::quiet_NaN();
        else
            ss(i) = sstar(cells(i, __colidx_rz), cells(i, __colidx_nz), R); 
    }
    
    // Compute the cell-cell repulsive forces 
    Array<T, Dynamic, 6> dEdq_repulsion = cellCellRepulsiveForces<T>(
        cells, neighbors, dt, iter, R, Rcell, cell_cell_prefactors 
    );

    // Compute the cell-cell adhesive forces (if present) 
    Array<T, Dynamic, 6> dEdq_adhesion = Array<T, Dynamic, 6>::Zero(n, 6); 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        dEdq_adhesion = cellCellAdhesiveForces<T>(
            cells, neighbors, to_adhere, dt, iter, R, Rcell, adhesion_mode,
            adhesion_params
        ); 
    }

    // Compute the cell-surface interaction forces (if present)
    Array<T, Dynamic, 2> dEdq_surface_repulsion = Array<T, Dynamic, 2>::Zero(n, 2);
    Array<T, Dynamic, 2> dEdq_surface_adhesion = Array<T, Dynamic, 2>::Zero(n, 2); 
    if (!no_surface)
    { 
        dEdq_surface_repulsion = cellSurfaceRepulsionForces<T>(
            cells, dt, iter, ss, R, E0, assume_2d, multithread
        );
        dEdq_surface_adhesion = cellSurfaceAdhesionForces<T>(
            cells, dt, iter, ss, R, assume_2d, multithread
        );
    }

    // Combine the forces accordingly 
    Array<T, Dynamic, 6> forces = -dEdq_repulsion - dEdq_adhesion; 
    forces.col(2) -= dEdq_surface_repulsion.col(0); 
    forces.col(5) -= dEdq_surface_repulsion.col(1);
    forces.col(2) -= dEdq_surface_adhesion.col(0); 
    forces.col(5) -= dEdq_surface_adhesion.col(1);

    return forces;  
}

/* -------------------------------------------------------------------- // 
//                     NEWTONIAN FORCES AND TORQUES                     // 
// -------------------------------------------------------------------- */
// TODO Implement this 

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
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
 */
template <typename T>
void truncateSurfaceFrictionCoeffsCoulomb(Ref<Array<T, Dynamic, Dynamic> > cells,
                                          const T R, const T E0,
                                          const T surface_coulomb_coeff)
{
    // TODO Implement this 
}

/**
 * Normalize the orientation vectors of all cells in the given population,
 * and redirect all orientation vectors so that they always have positive
 * z-coordinates.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 */
template <typename T>
void normalizeOrientations(Ref<Array<T, Dynamic, Dynamic> > cells, const T dt,
                           const int iter)
{
    Array<T, Dynamic, 1> norms = cells(Eigen::all, __colseq_n).matrix().rowwise().norm().array();
    #ifdef DEBUG_CHECK_ORIENTATION_NORMS_NONZERO
        for (int i = 0; i < cells.rows(); ++i)
        {
            if (norms(i) < 1e-8)
            {
                std::cerr << "Iteration " << iter
                          << ": Found near-zero orientation for cell " << i << std::endl; 
                std::cerr << "Timestep: " << dt << std::endl;
                std::cerr << "Cell center = (" << cells(i, __colidx_rx) << ", "
                                               << cells(i, __colidx_ry) << ", "
                                               << cells(i, __colidx_rz) << ")" << std::endl
                          << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                    << cells(i, __colidx_ny) << ", "
                                                    << cells(i, __colidx_nz) << ")" << std::endl
                          << "Cell half-length = " << cells(i, __colidx_half_l) << std::endl;
            }
        }
    #endif
    cells.col(__colidx_nx) /= norms; 
    cells.col(__colidx_ny) /= norms;
    cells.col(__colidx_nz) /= norms;

    // Ensure that all z-orientations are positive 
    for (int i = 0; i < cells.rows(); ++i)
    {
        if (cells(i, __colidx_nz) < 0)
        {
            cells(i, __colidx_nx) *= -1;
            cells(i, __colidx_ny) *= -1;
            cells(i, __colidx_nz) *= -1;
        }
        #ifdef DEBUG_WARN_ORIENTATION_REVERSED
            std::cout << "[WARN] Iteration " << iter
                      << ": orientation reversed for cell " << i << std::endl; 
        #endif
    }
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
 * been pre-computed.
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
 * @param E0 Elastic modulus of EPS.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero. 
 * @param noise Noise to be added to each generalized force used to compute
 *              the velocities.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param no_surface If true, omit the surface from the simulation. 
 * @param multithread If true, use multithreading. 
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 6> getVelocities(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                   const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                   const Ref<const Array<int, Dynamic, 1> >& to_adhere,
                                   const T dt, const int iter, const T R, const T Rcell,
                                   const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
                                   const T E0,
                                   const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                   const Ref<const Array<T, Dynamic, 6> >& noise,
                                   const AdhesionMode adhesion_mode,
                                   std::unordered_map<std::string, T>& adhesion_params,
                                   const bool no_surface,
                                   const bool multithread)
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
    Array<T, Dynamic, 1> ss(n); 
    for (int i = 0; i < n; ++i)
    {
        if (assume_2d(i))
            ss(i) = std::numeric_limits<T>::quiet_NaN();
        else
            ss(i) = sstar(cells(i, __colidx_rz), cells(i, __colidx_nz), R); 
    }

    // Compute all conservative forces and add noise 
    Array<T, Dynamic, 6> forces = getConservativeForces<T>(
        cells, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors, 
        E0, assume_2d, adhesion_mode, adhesion_params, no_surface, multithread
    );
    forces += noise; 
    
    // For each cell ...
    #pragma omp parallel for if(multithread)
    for (int i = 0; i < n; ++i)
    {
        // Is the cell contacting the surface?
        bool contacting_surface = false;
        if (!assume_2d(i))    // If the cell is not horizontal, then ... 
        {
            // Identify the cell height at either endpoint and check that
            // there is some cell-surface overlap 
            T z1 = cells(i, __colidx_rz) - cells(i, __colidx_half_l) * cells(i, __colidx_nz); 
            T z2 = cells(i, __colidx_rz) + cells(i, __colidx_half_l) * cells(i, __colidx_nz);
            contacting_surface = (z1 < R || z2 < R); 
        }
        else    // If the cell is horizontal, then ... 
        {
            // Simply check the height of the cell center 
            contacting_surface = (cells(i, __colidx_rz) < R); 
        }

        // If there is no surface or the cell is not contacting the surface,
        // solve the simplified 3D system of equations with no cell-surface
        // friction 
        if (no_surface || !contacting_surface)
        {
            T K = cells(i, __colidx_eta0) * cells(i, __colidx_l);
            T L = K * cells(i, __colidx_l) * cells(i, __colidx_l) / 12;
            T mult = -(
                cells(i, __colidx_nx) * forces(i, 3) +
                cells(i, __colidx_ny) * forces(i, 4) +
                cells(i, __colidx_nz) * forces(i, 5)
            ); 
            velocities(i, Eigen::seq(0, 2)) = forces(i, Eigen::seq(0, 2)) / K;  
            velocities(i, Eigen::seq(3, 5)) = (forces(i, Eigen::seq(3, 5)) + mult * cells(i, __colseq_n)) / L;
            #ifdef DEBUG_CHECK_VELOCITIES_NAN
                if (velocities.row(i).isNaN().any() || velocities.row(i).isInf().any())
                {
                    std::cerr << std::setprecision(10); 
                    std::cerr << "Iteration " << iter
                              << ": Found nan in velocities of cell " << i << std::endl; 
                    std::cerr << "Timestep: " << dt << std::endl;
                    std::cerr << "Contacting surface: 0" << std::endl; 
                    std::cerr << "Cell center = (" << cells(i, __colidx_rx) << ", "
                                                   << cells(i, __colidx_ry) << ", "
                                                   << cells(i, __colidx_rz) << ")" << std::endl
                              << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                        << cells(i, __colidx_ny) << ", "
                                                        << cells(i, __colidx_nz) << ")" << std::endl
                              << "Cell half-length = " << cells(i, __colidx_half_l) << std::endl
                              << "Cell translational velocity = (" << velocities(i, 0) << ", "
                                                                   << velocities(i, 1) << ", "
                                                                   << velocities(i, 2) << ")" << std::endl
                              << "Cell orientational velocity = (" << velocities(i, 3) << ", "
                                                                   << velocities(i, 4) << ", "
                                                                   << velocities(i, 5) << ")" << std::endl
                              << "Constraint variable = " << mult / 2.0 << std::endl
                              << "Composite viscosity prefactors = " << K << ", " << L << std::endl
                              << "Conservative force vector = (" << forces(i, 0) << ", "
                                                                 << forces(i, 1) << ", "
                                                                 << forces(i, 2) << ", "
                                                                 << forces(i, 3) << ", "
                                                                 << forces(i, 4) << ", "
                                                                 << forces(i, 5) << ")" << std::endl; 
                    throw std::runtime_error("Found nan in velocities"); 
                }
            #endif
        }
        // Otherwise, if the cell is roughly horizontal, solve the 2D system
        // of equations with cell-surface friction 
        else if (assume_2d(i))
        {
            T K, L;                              // Viscosity force prefactors 
            T a = sqrt(R * (R - cells(i, __colidx_rz))); 
            K = cells(i, __colidx_l) * (
                cells(i, __colidx_eta0) + cells(i, __colidx_eta1) * a / R
            );  
            L = K * cells(i, __colidx_l) * cells(i, __colidx_l) / 12;
            T mult = -(cells(i, __colidx_nx) * forces(i, 3) + cells(i, __colidx_ny) * forces(i, 4)); 
            velocities(i, 0) = forces(i, 0) / K; 
            velocities(i, 1) = forces(i, 1) / K;
            velocities(i, 3) = (forces(i, 3) + mult * cells(i, __colidx_nx)) / L;
            velocities(i, 4) = (forces(i, 4) + mult * cells(i, __colidx_ny)) / L;

            // Set velocities in z-direction, which may be nonzero due to noise 
            velocities(i, 2) = forces(i, 2) / (cells(i, __colidx_eta0) * cells(i, __colidx_l));
            velocities(i, 5) = 12 * forces(i, 5) / (
                cells(i, __colidx_eta0) * cells(i, __colidx_l) * cells(i, __colidx_l) *
                cells(i, __colidx_l)
            );
            #ifdef DEBUG_CHECK_VELOCITIES_NAN
                if (velocities.row(i).isNaN().any() || velocities.row(i).isInf().any())
                {
                    std::cerr << std::setprecision(10); 
                    std::cerr << "Iteration " << iter
                              << ": Found nan in velocities of cell " << i << std::endl; 
                    std::cerr << "Timestep: " << dt << std::endl;
                    std::cerr << "Contacting surface: 1" << std::endl; 
                    std::cerr << "2D assumption: 1" << std::endl; 
                    std::cerr << "Cell center = (" << cells(i, __colidx_rx) << ", "
                                                   << cells(i, __colidx_ry) << ", "
                                                   << cells(i, __colidx_rz) << ")" << std::endl
                              << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                        << cells(i, __colidx_ny) << ", "
                                                        << cells(i, __colidx_nz) << ")" << std::endl
                              << "Cell half-length = " << cells(i, __colidx_half_l) << std::endl
                              << "Cell translational velocity = (" << velocities(i, 0) << ", "
                                                                   << velocities(i, 1) << ", "
                                                                   << velocities(i, 2) << ")" << std::endl
                              << "Cell orientational velocity = (" << velocities(i, 3) << ", "
                                                                   << velocities(i, 4) << ", "
                                                                   << velocities(i, 5) << ")" << std::endl
                              << "Constraint variable = " << mult / 2.0 << std::endl
                              << "Composite viscosity prefactors = " << K << ", " << L << std::endl
                              << "Conservative force vector = (" << forces(i, 0) << ", "
                                                                 << forces(i, 1) << ", "
                                                                 << forces(i, 2) << ", "
                                                                 << forces(i, 3) << ", "
                                                                 << forces(i, 4) << ", "
                                                                 << forces(i, 5) << ")" << std::endl; 
                    throw std::runtime_error("Found nan in velocities"); 
                }
            #endif
        }
        else     // Otherwise, solve the full 3D system of equations 
        {
            /*
            Array<T, 7, 7> A = Array<T, 7, 7>::Zero();
            Array<T, 7, 1> b = Array<T, 7, 1>::Zero();

            // Compute the derivatives of the dissipation with respect to the 
            // cell's translational and orientational velocities
            A(Eigen::seq(0, 5), Eigen::seq(0, 5)) = compositeViscosityForceMatrix<T>(
                cells(i, __colidx_rz), cells(i, __colidx_nz), cells(i, __colidx_l),
                cells(i, __colidx_half_l), ss(i), cells(i, __colidx_eta0),
                cells(i, __colidx_eta1), R, i, dt, iter
            );
            A(3, 6) = -2 * cells(i, __colidx_nx); 
            A(4, 6) = -2 * cells(i, __colidx_ny);
            A(5, 6) = -2 * cells(i, __colidx_nz);
            A(6, 3) = cells(i, __colidx_nx); 
            A(6, 4) = cells(i, __colidx_ny);
            A(6, 5) = cells(i, __colidx_nz);

            // Solve the corresponding linear system
            //
            // TODO Which decomposition to use?
            b.head(6) = forces.row(i);
            auto LU = A.matrix().partialPivLu();
            Array<T, 7, 1> x = LU.solve(b.matrix()).array();
            //auto QR = A.matrix().colPivHouseholderQr(); 
            //Array<T, 7, 1> x = QR.solve(b.matrix()).array();
            */

            // Solve the linear system explicitly, using back-substitution 
            const T rz = cells(i, __colidx_rz); 
            const T nz = cells(i, __colidx_nz); 
            const T l = cells(i, __colidx_l); 
            const T half_l = cells(i, __colidx_half_l);
            const T eta0 = cells(i, __colidx_eta0); 
            const T eta1 = cells(i, __colidx_eta1); 
            std::tuple<T, T, T> integrals = areaIntegrals<T>(rz, nz, R, half_l, ss(i)); 
            const T c1 = eta0 * l + (eta1 / R) * std::get<0>(integrals);
            const T c2 = eta0 * l; 
            const T c3 = c2 * l * l / 12 + (eta1 / R) * std::get<2>(integrals);
            const T c4 = c2 * l * l / 12; 
            const T c5 = (eta1 / R) * std::get<1>(integrals);
            const T term1 = c3 - (c5 * c5 / c1);
            const T w4 = forces(i, 3) - forces(i, 0) * (c5 / c1); 
            const T w5 = forces(i, 4) - forces(i, 1) * (c5 / c1); 

            // Calculate the Lagrange multiplier ...  
            const T nx = cells(i, __colidx_nx); 
            const T ny = cells(i, __colidx_ny); 
            const T nx2 = nx / term1; 
            const T ny2 = ny / term1; 
            const T nz2 = nz / c4; 
            const T alpha = 2 * (nx * nx2 + ny * ny2 + nz * nz2);
            const T beta = -(w4 * nx2 + w5 * ny2 + forces(i, 5) * nz2);
            const T lambda = beta / alpha;

            // ... then calculate the velocities via back-substitution 
            velocities(i, 5) = (forces(i, 5) + 2 * nz * lambda) / c4;
            velocities(i, 4) = (w5 + 2 * ny * lambda) / term1; 
            velocities(i, 3) = (w4 + 2 * nx * lambda) / term1; 
            velocities(i, 2) = forces(i, 2) / c2; 
            velocities(i, 1) = (forces(i, 1) - c5 * velocities(i, 4)) / c1; 
            velocities(i, 0) = (forces(i, 0) - c5 * velocities(i, 3)) / c1;

            // TODO Update debug here
            #ifdef DEBUG_CHECK_VELOCITIES_NAN
                //if (x.isNaN().any() || x.isInf().any())
                if (velocities.row(i).isNaN().any() || velocities.row(i).isInf().any())
                {
                    std::cerr << std::setprecision(10); 
                    std::cerr << "Iteration " << iter
                              << ": Found nan in velocities of cell " << i << std::endl; 
                    std::cerr << "Timestep: " << dt << std::endl;
                    std::cerr << "Contacting surface: 1" << std::endl; 
                    std::cerr << "2D assumption: 0" << std::endl; 
                    std::cerr << "Cell center = (" << cells(i, __colidx_rx) << ", "
                                                   << cells(i, __colidx_ry) << ", "
                                                   << cells(i, __colidx_rz) << ")" << std::endl
                              << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                        << cells(i, __colidx_ny) << ", "
                                                        << cells(i, __colidx_nz) << ")" << std::endl
                              << "Cell half-length = " << cells(i, __colidx_half_l) << std::endl
                              //<< "Cell translational velocity = (" << x(0) << ", "
                              //                                     << x(1) << ", "
                              //                                     << x(2) << ")" << std::endl
                              //<< "Cell orientational velocity = (" << x(3) << ", "
                              //                                     << x(4) << ", "
                              //                                     << x(5) << ")" << std::endl
                              //<< "Constraint variable = " << x(6) << std::endl
                              << "Cell translational velocity = (" << velocities(i, 0) << ", "
                                                                   << velocities(i, 1) << ", "
                                                                   << velocities(i, 2) << ")" << std::endl
                              << "Cell orientational velocity = (" << velocities(i, 3) << ", "
                                                                   << velocities(i, 4) << ", "
                                                                   << velocities(i, 5) << ")" << std::endl
                              << "Constraint variable = " << lambda << std::endl
                              << "Composite viscosity force matrix = " << std::endl;
                    /* 
                    for (int j = 0; j < 7; ++j)
                    {
                        std::cerr << "  [";
                        for (int k = 0; k < 6; ++k)
                            std::cerr << A(j, k) << ", ";
                        std::cerr << A(j, 6) << "]" << std::endl; 
                    }
                    std::cerr << "Conservative force vector = (" << b(0) << ", "
                                                                 << b(1) << ", "
                                                                 << b(2) << ", "
                                                                 << b(3) << ", "
                                                                 << b(4) << ", "
                                                                 << b(5) << ")" << std::endl;
                    */
                    throw std::runtime_error("Found nan in velocities"); 
                }
            #endif
            //velocities.row(i) = x.head(6);
        }
    }

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
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population.
 * @param to_adhere Boolean array specifying whether, for each pair of 
 *                  neighboring cells, the adhesive force is nonzero.
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param cell_cell_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
 * @param E0 Elastic modulus of EPS. 
 * @param nz_threshold Threshold for judging whether each cell's z-orientation
 *                     is zero. 
 * @param max_rxy_noise Maximum noise to be added to each generalized force in
 *                      the x- and y-directions.
 * @param max_rz_noise Maximum noise to be added to each generalized force in
 *                     the z-direction.
 * @param max_nxy_noise Maximum noise to be added to each generalized torque in
 *                      the x- and y-directions.
 * @param max_nz_noise Maximum noise to be added to each generalized torque in
 *                     the z-direction.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR (1), KIHARA (2), or GBK (3).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param no_surface If true, omit the surface from the simulation. 
 * @param n_start_multithread Minimum number of cells at which to start using
 *                            multithreading. 
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 6> >
    stepRungeKuttaAdaptive(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                           const Ref<const Array<T, Dynamic, 1> >& b,
                           const Ref<const Array<T, Dynamic, 1> >& bs, 
                           const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                           const Ref<const Array<T, Dynamic, 7> >& neighbors,
                           const Ref<const Array<int, Dynamic, 1> >& to_adhere, 
                           const T dt, const int iter, const T R, const T Rcell,
                           const Ref<const Array<T, 3, 1> >& cell_cell_prefactors,
                           const T E0, const T nz_threshold, const T max_rxy_noise,
                           const T max_rz_noise, const T max_nxy_noise,
                           const T max_nz_noise, boost::random::mt19937& rng,
                           boost::random::uniform_01<>& uniform_dist,
                           const AdhesionMode adhesion_mode,
                           std::unordered_map<std::string, T>& adhesion_params,
                           const bool no_surface,
                           const int n_start_multithread = 50)
{
    #ifdef DEBUG_CHECK_NEIGHBOR_DISTANCES_ZERO
        for (int k = 0; k < neighbors.rows(); ++k)
        {
            if (neighbors(k, Eigen::seq(2, 4)).matrix().norm() < 1e-8)
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

    // Define an array of indicators for whether each cell is roughly horizontal
    //
    // This array is utilized for each step in the Runge-Kutta update
    int n = cells.rows(); 
    Array<int, Dynamic, 1> assume_2d(n);
    for (int i = 0; i < n; ++i)
        assume_2d(i) = (cells(i, __colidx_nz) < nz_threshold); 

    // Compute velocities at given partial timesteps 
    //
    // Sample noise components prior to velocity calculations, so that they 
    // are the same in each calculation 
    Array<T, Dynamic, 6> noise = Array<T, Dynamic, 6>::Zero(n, 6);  
    if (max_rxy_noise > 0 || max_rz_noise > 0 || max_nxy_noise > 0 || max_nz_noise > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                T r = uniform_dist(rng); 
                if (j == 0 || j == 1)
                    noise(i, j) = -max_rxy_noise + 2 * max_rxy_noise * r;
                else if (j == 3 || j == 4)
                    noise(i, j) = -max_nxy_noise + 2 * max_nxy_noise * r;
                else if (j == 2) 
                    noise(i, j) = -max_rz_noise + 2 * max_rz_noise * r;
                else    // j == 5
                    noise(i, j) = -max_nz_noise + 2 * max_nz_noise * r; 
            }
        }
    }
    #ifdef DEBUG_CHECK_NOISE_MAGNITUDES
        for (int i = 0; i < n; ++i)
        { 
            if (abs(noise(i, 0)) > max_rxy_noise || abs(noise(i, 1)) > max_rxy_noise ||
                abs(noise(i, 2)) > max_rz_noise  || abs(noise(i, 3)) > max_nxy_noise ||
                abs(noise(i, 4)) > max_nxy_noise || abs(noise(i, 5)) > max_nz_noise)
            {
                std::cerr << std::setprecision(10);
                std::cerr << "Iteration " << iter
                          << ": Generated out-of-bounds noise vector for cell " << i << std::endl;
                std::cerr << "Timestep: " << dt << std::endl;
                std::cerr << "Noise = (" << noise(i, 0) << ", " << noise(i, 1) << ", "
                                         << noise(i, 2) << ", " << noise(i, 3) << ", "
                                         << noise(i, 4) << ", " << noise(i, 5) << ")"
                                         << std::endl;  
                throw std::runtime_error("Generated out-of-bounds noise vectors"); 
            }
        }
    #endif 

    int s = b.size();
    bool multithread = (n >= n_start_multithread); 
    std::vector<Array<T, Dynamic, 6> > velocities;
    Array<T, Dynamic, 6> v0 = getVelocities<T>(
        cells, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors,
        E0, assume_2d, noise, adhesion_mode, adhesion_params, no_surface,
        multithread
    );
    velocities.push_back(v0);
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 6> multipliers = Array<T, Dynamic, 6>::Zero(n, 6);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, __colseq_coords) += multipliers * dt;
        normalizeOrientations<T>(cells_i, dt, iter);    
        Array<T, Dynamic, 6> vi = getVelocities<T>(
            cells_i, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors,
            E0, assume_2d, noise, adhesion_mode, adhesion_params, no_surface,
            multithread
        );
        velocities.push_back(vi);
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
    cells_new(Eigen::all, __colseq_coords) += delta1;
    Array<T, Dynamic, 6> errors = delta1 - delta2;

    #ifdef DEBUG_CHECK_VELOCITIES_NAN
        for (int i = 0; i < n; ++i)
        {
            if (velocities_final1.row(i).isNaN().any() ||
                velocities_final1.row(i).isInf().any() ||
                errors.row(i).isNaN().any() || errors.row(i).isInf().any())
            {
                std::cerr << "Iteration " << iter
                          << ": Found nan in velocities of cell " << i << std::endl; 
                std::cerr << "Timestep: " << dt << std::endl;
                std::cerr << "2D assumption: " << assume_2d(i) << std::endl; 
                std::cerr << "Cell center = (" << cells(i, __colidx_rx) << ", "
                                               << cells(i, __colidx_ry) << ", "
                                               << cells(i, __colidx_rz) << ")" << std::endl
                          << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                    << cells(i, __colidx_ny) << ", "
                                                    << cells(i, __colidx_nz) << ")" << std::endl
                          << "Cell half-length = " << cells(i, __colidx_half_l) << std::endl
                          << "Cell translational velocity = (" << velocities_final1(i, 0) << ", "
                                                               << velocities_final1(i, 1) << ", "
                                                               << velocities_final1(i, 2) << ")" << std::endl
                          << "Cell orientational velocity = (" << velocities_final1(i, 3) << ", "
                                                               << velocities_final1(i, 4) << ", "
                                                               << velocities_final1(i, 5) << ")" << std::endl
                          << "Errors = (" << errors(i, 0) << ", " << errors(i, 1) << ", "
                                          << errors(i, 2) << ", " << errors(i, 3) << ", "
                                          << errors(i, 4) << ", " << errors(i, 5) << ")" << std::endl; 
                throw std::runtime_error("Found nan in velocities"); 
            }
        }
    #endif

    // Store computed velocities 
    cells_new(Eigen::all, __colseq_velocities) = velocities_final1; 
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells_new, dt, iter); 

    return std::make_pair(cells_new, errors);
}

/* -------------------------------------------------------------------- // 
//              VELOCITY UPDATES IN THE NEWTONIAN FRAMEWORK             //
// -------------------------------------------------------------------- */
// TODO To be implemented 

#endif
