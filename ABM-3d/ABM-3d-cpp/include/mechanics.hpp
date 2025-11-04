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
 *     10/8/2025
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
using std::min; 
using boost::multiprecision::min; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * An enum that enumerates the different cell-cell adhesion force types. 
 */
enum class AdhesionMode
{
    NONE = 0,
    JKR_ISOTROPIC = 1,
    JKR_ANISOTROPIC = 2
};

/**
 * An enum that enumerates the different cell-cell friction force types. 
 */
enum class FrictionMode
{
    NONE = 0, 
    KINETIC = 1
};

/**
 * A struct that contains all necessary pre-computed values for calculating
 * JKR forces.  
 */
template <typename T>
struct JKRData
{
    T max_gamma;           // Maximum value of gamma (surface adhesion energy density) 
    T gamma_switch_rate;   // Timescale for switching between zero and maximal adhsion
    bool gamma_fixed;      // Whether or not gamma is fixed
    bool initialized;      // Whether or not the struct is populated 

    // Meshes of input variables
    Matrix<T, Dynamic, 1> overlaps; 
    Matrix<T, Dynamic, 1> gamma; 
    Matrix<T, Dynamic, 1> theta; 
    Matrix<T, Dynamic, 1> half_l; 
    Matrix<T, Dynamic, 1> centerline_coords;
    Matrix<T, Dynamic, 1> Rx; 
    Matrix<T, Dynamic, 1> phi;

    // Destructor 
    virtual ~JKRData() = default; 

    // Placeholder force evaluation methods in 3-D
    virtual std::pair<Array<T, 2, 6>, T> getJKRForces(const Ref<const Matrix<T, 3, 1> >& r1, 
                                                      const Ref<const Matrix<T, 3, 1> >& n1, 
                                                      const T half_l1, 
                                                      const Ref<const Matrix<T, 3, 1> >& r2, 
                                                      const Ref<const Matrix<T, 3, 1> >& n2, 
                                                      const T half_l2, 
                                                      const Ref<const Matrix<T, 3, 1> >& d12, 
                                                      const T R,
                                                      const T E0,
                                                      const T gamma,  
                                                      const T s,
                                                      const T t,
                                                      const bool include_constraint = true, 
                                                      const T max_overlap = -1, 
                                                      const bool interpolate = true) = 0; 
}; 

template <typename T>
struct IsotropicJKRDataFixedGamma : public JKRData<T>
{
    std::unordered_map<int, T> contact_radii;     // overlap -> radius
    
    // Constructor with input/output meshes  
    IsotropicJKRDataFixedGamma(const T max_gamma,
                               const Ref<const Matrix<T, Dynamic, 1> >& overlaps, 
                               std::unordered_map<int, T>& contact_radii) : JKRData<T>() 
    {
        this->max_gamma = max_gamma; 
        this->gamma_switch_rate = std::numeric_limits<T>::infinity(); 
        this->gamma_fixed = true; 
        this->initialized = true;
        this->overlaps = overlaps; 
        this->contact_radii = contact_radii; 
    }

    // Force evaluation method in 3-D 
    std::pair<Array<T, 2, 6>, T> getJKRForces(const Ref<const Matrix<T, 3, 1> >& r1, 
                                              const Ref<const Matrix<T, 3, 1> >& n1, 
                                              const T half_l1, 
                                              const Ref<const Matrix<T, 3, 1> >& r2, 
                                              const Ref<const Matrix<T, 3, 1> >& n2, 
                                              const T half_l2, 
                                              const Ref<const Matrix<T, 3, 1> >& d12, 
                                              const T R,
                                              const T E0,
                                              const T gamma,  
                                              const T s,
                                              const T t,
                                              const bool include_constraint = true, 
                                              const T max_overlap = -1, 
                                              const bool interpolate = true)
    {
        return forcesIsotropicJKRLagrange<T, 3>(
            n1, n2, d12, R, E0, gamma, s, t, this->overlaps, this->contact_radii,
            include_constraint, max_overlap, interpolate
        ); 
    }  
};

template <typename T>
struct IsotropicJKRDataVariableGamma : public JKRData<T>
{
    TupleToScalarTable<T, 2> contact_radii;     // (overlap, gamma) -> radius
    
    // Constructor with input/output meshes  
    IsotropicJKRDataVariableGamma(const T max_gamma, const T gamma_switch_rate, 
                                  const Ref<const Matrix<T, Dynamic, 1> >& overlaps,
                                  const Ref<const Matrix<T, Dynamic, 1> >& gamma,  
                                  TupleToScalarTable<T, 2>& contact_radii) : JKRData<T>() 
    {
        this->max_gamma = max_gamma; 
        this->gamma_switch_rate = gamma_switch_rate; 
        this->gamma_fixed = false; 
        this->initialized = true;
        this->overlaps = overlaps; 
        this->gamma = gamma; 
        this->contact_radii = contact_radii; 
    }
    
    // Force evaluation method in 3-D 
    std::pair<Array<T, 2, 6>, T> getJKRForces(const Ref<const Matrix<T, 3, 1> >& r1, 
                                              const Ref<const Matrix<T, 3, 1> >& n1, 
                                              const T half_l1, 
                                              const Ref<const Matrix<T, 3, 1> >& r2, 
                                              const Ref<const Matrix<T, 3, 1> >& n2, 
                                              const T half_l2, 
                                              const Ref<const Matrix<T, 3, 1> >& d12, 
                                              const T R,
                                              const T E0,
                                              const T gamma,  
                                              const T s,
                                              const T t,
                                              const bool include_constraint = true, 
                                              const T max_overlap = -1, 
                                              const bool interpolate = true)
    {
        return forcesIsotropicJKRLagrange<T, 3>(
            n1, n2, d12, R, E0, gamma, s, t, this->overlaps, this->gamma, 
            this->contact_radii, include_constraint, max_overlap, interpolate
        ); 
    }  
};

template <typename T>
struct AnisotropicJKRDataFixedGamma : public JKRData<T>
{
    // (angle between contact point and orientation vector, cell half-length,
    //  normalized centerline coordinate) -> (Rx, Ry) at contact point 
    TupleToTupleTable<T, 3, 2> curvature_radii;

    // (maximum principal radius of curvature for cell 1, 
    //  maximum principal radius of curvature for cell 2, 
    //  angle between cell orientations, 
    //  overlap) -> (force, radius)
    TupleToTupleTable<T, 4, 2> forces; 

    // Constructor with input/output meshes  
    AnisotropicJKRDataFixedGamma(const T max_gamma,
                                 const Ref<const Matrix<T, Dynamic, 1> >& overlaps,
                                 const Ref<const Matrix<T, Dynamic, 1> >& theta, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& half_l, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& centerline_coords, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& Rx, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& phi, 
                                 TupleToTupleTable<T, 3, 2>& curvature_radii, 
                                 TupleToTupleTable<T, 4, 2>& forces) : JKRData<T>() 
    {
        this->max_gamma = max_gamma; 
        this->gamma_switch_rate = std::numeric_limits<T>::infinity(); 
        this->gamma_fixed = true; 
        this->initialized = true;
        this->overlaps = overlaps; 
        this->theta = theta; 
        this->half_l = half_l; 
        this->centerline_coords = centerline_coords; 
        this->Rx = Rx; 
        this->phi = phi; 
        this->curvature_radii = curvature_radii; 
        this->forces = forces; 
    }

    // Force evaluation method in 3-D 
    std::pair<Array<T, 2, 6>, T> getJKRForces(const Ref<const Matrix<T, 3, 1> >& r1, 
                                              const Ref<const Matrix<T, 3, 1> >& n1, 
                                              const T half_l1, 
                                              const Ref<const Matrix<T, 3, 1> >& r2, 
                                              const Ref<const Matrix<T, 3, 1> >& n2, 
                                              const T half_l2, 
                                              const Ref<const Matrix<T, 3, 1> >& d12, 
                                              const T R,
                                              const T E0,
                                              const T gamma,  
                                              const T s,
                                              const T t,
                                              const bool include_constraint = true, 
                                              const T max_overlap = -1, 
                                              const bool interpolate = true)
    {
        return forcesAnisotropicJKRLagrange<T, 3>(
            n1, half_l1, n2, half_l2, d12, R, E0, gamma, s, t, this->theta, 
            this->half_l, this->centerline_coords, this->curvature_radii, 
            this->Rx, this->phi, this->overlaps, this->forces, include_constraint,
            max_overlap
        ); 
    }  
}; 

template <typename T>
struct AnisotropicJKRDataVariableGamma : public JKRData<T>
{
    // (angle between contact point and orientation vector, cell half-length,
    //  normalized centerline coordinate) -> (Rx, Ry) at contact point 
    TupleToTupleTable<T, 3, 2> curvature_radii;

    // (maximum principal radius of curvature for cell 1, 
    //  maximum principal radius of curvature for cell 2, 
    //  angle between cell orientations, 
    //  overlap, gamma) -> (force, radius)
    TupleToTupleTable<T, 5, 2> forces; 

    // Constructor with input/output meshes  
    AnisotropicJKRDataVariableGamma(const T max_gamma, const T gamma_switch_rate,
                                    const Ref<const Matrix<T, Dynamic, 1> >& overlaps,
                                    const Ref<const Matrix<T, Dynamic, 1> >& theta, 
                                    const Ref<const Matrix<T, Dynamic, 1> >& half_l, 
                                    const Ref<const Matrix<T, Dynamic, 1> >& centerline_coords, 
                                    const Ref<const Matrix<T, Dynamic, 1> >& Rx, 
                                    const Ref<const Matrix<T, Dynamic, 1> >& phi,
                                    const Ref<const Matrix<T, Dynamic, 1> >& gamma, 
                                    TupleToTupleTable<T, 3, 2>& curvature_radii, 
                                    TupleToTupleTable<T, 5, 2>& forces) : JKRData<T>() 
    {
        this->max_gamma = max_gamma; 
        this->gamma_switch_rate = gamma_switch_rate; 
        this->gamma_fixed = false; 
        this->initialized = true;
        this->overlaps = overlaps;
        this->gamma = gamma;  
        this->theta = theta; 
        this->half_l = half_l; 
        this->centerline_coords = centerline_coords; 
        this->Rx = Rx; 
        this->phi = phi; 
        this->curvature_radii = curvature_radii; 
        this->forces = forces; 
    }
    
    // Force evaluation method in 3-D 
    std::pair<Array<T, 2, 6>, T> getJKRForces(const Ref<const Matrix<T, 3, 1> >& r1, 
                                              const Ref<const Matrix<T, 3, 1> >& n1, 
                                              const T half_l1, 
                                              const Ref<const Matrix<T, 3, 1> >& r2, 
                                              const Ref<const Matrix<T, 3, 1> >& n2, 
                                              const T half_l2, 
                                              const Ref<const Matrix<T, 3, 1> >& d12, 
                                              const T R,
                                              const T E0,
                                              const T gamma,  
                                              const T s,
                                              const T t,
                                              const bool include_constraint = true, 
                                              const T max_overlap = -1, 
                                              const bool interpolate = true)
    {
        return forcesAnisotropicJKRLagrange<T, 3>(
            n1, half_l1, n2, half_l2, d12, R, E0, gamma, s, t, this->theta, 
            this->half_l, this->centerline_coords, this->curvature_radii, 
            this->Rx, this->phi, this->overlaps, this->gamma, this->forces,
            include_constraint, max_overlap
        ); 
    }  
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
 * @param dr Cell translational velocity. 
 * @param dn Cell orientational velocity. 
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
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces.  
 * @returns Derivatives of the cell-surface repulsion energies with respect to
 *          cell z-positions and z-orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellSurfaceRepulsionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                const T dt, const int iter,
                                                const Ref<const Array<T, Dynamic, 1> >& ss,
                                                const T R, const T E0,
                                                const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                                const bool multithread,
                                                const bool include_constraint = false) 
{
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(cells.rows(), 6);

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
            // In this case, the torque is always zero and the force is
            // nonzero only if phi > 0
            T phi = R - cells(i, __colidx_rz);
            if (phi > 0)
                dEdq(i, 2) = -prefactor0 * phi * cells(i, __colidx_l);
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
            dEdq(i, 2) = -prefactor0 * ((1.0 - nz2) * int1 + sqrt(R) * nz2 * int2);

            // Compute the derivative of the cell-surface repulsion energy 
            // with respect to z-orientation
            if (!include_constraint)
            {
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
                dEdq(i, 5) -= prefactor0 * cells(i, __colidx_nz) * int3;
                dEdq(i, 5) -= prefactor0 * (1 - nz2) * int4;
                dEdq(i, 5) += prefactor1 * cells(i, __colidx_nz) * int5;
                dEdq(i, 5) -= prefactor2 * nz2 * int6;
            }
            else 
            {
                T int3 = integral2<T>(    // Integral of s * \delta_i(s)
                    cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                    cells(i, __colidx_half_l), 1.0, ss(i)
                );
                T int4 = integral2<T>(    // Integral of s * \sqrt{\delta_i(s)}
                    cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                    cells(i, __colidx_half_l), 0.5, ss(i)
                );
                Matrix<T, 3, 1> cross; 
                cross << cells(i, __colidx_ny), -cells(i, __colidx_nx), 0; 
                dEdq(i, 3) = -prefactor0 * cross(0) * ((1.0 - nz2) * int3 + sqrt(R) * nz2 * int4); 
                dEdq(i, 4) = -prefactor0 * cross(1) * ((1.0 - nz2) * int3 + sqrt(R) * nz2 * int4); 
            }
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
                std::cerr << "Cell-surface contact coordinate: " << ss(i) << std::endl;  
                cellLagrangianForcesSummary<T>(
                    cells(i, __colseq_r), cells(i, __colseq_n),
                    cells(i, __colidx_half_l), dEdq.row(i)
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
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces.  
 * @returns Derivatives of the cell-surface adhesion energies with respect to
 *          cell z-positions and z-orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellSurfaceAdhesionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const T dt, const int iter,
                                               const Ref<const Array<T, Dynamic, 1> >& ss,
                                               const T R,
                                               const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                               const bool multithread,
                                               const bool include_constraint = false)
{
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(cells.rows(), 6);

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
            // In this case, the torque is always zero and the force is
            // nonzero only if phi > 0
            T phi = R - cells(i, __colidx_rz);
            if (phi > 0)
                dEdq(i, 2) = cells(i, __colidx_sigma0) * prefactor0 * cells(i, __colidx_l) / pow(phi, 0.5);
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
            dEdq(i, 2) = prefactor0 * (1 - nz2) * int1 + term2;
            dEdq(i, 2) *= cells(i, __colidx_sigma0);

            // Compute the derivative of the cell-surface adhesion energy
            // with respect to z-orientation
            if (!include_constraint)
            {
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
                dEdq(i, 5) += prefactor2 * cells(i, __colidx_nz) * int2; 
                dEdq(i, 5) += prefactor0 * (1 - nz2) * int3;
                dEdq(i, 5) -= prefactor1 * cells(i, __colidx_nz) * int4;
                dEdq(i, 5) += term4;
                dEdq(i, 5) *= cells(i, __colidx_sigma0);
            }
            else 
            {
                T int2 = integral2<T>(    // Integral of s * \delta_i^{-1/2}(s)
                    cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                    cells(i, __colidx_half_l), -0.5, ss(i)
                );
                Matrix<T, 3, 1> cross; 
                cross << cells(i, __colidx_ny), -cells(i, __colidx_nx), 0; 
                T w1 = 0;
                T w2 = 0; 
                if (abs(ss(i)) < cells(i, __colidx_half_l))
                {
                    w1 = (prefactor1 / 2) * ss(i) * cells(i, __colidx_nz) * cross(0); 
                    w2 = (prefactor1 / 2) * ss(i) * cells(i, __colidx_nz) * cross(1); 
                }
                dEdq(i, 3) = prefactor0 * (1 - nz2) * int2 * cross(0) - w1; 
                dEdq(i, 3) *= cells(i, __colidx_sigma0); 
                dEdq(i, 4) = prefactor0 * (1 - nz2) * int2 * cross(1) - w2;
                dEdq(i, 4) *= cells(i, __colidx_sigma0);  
            }
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
                cellLagrangianForcesSummary<T>(
                    cells(i, __colseq_r), cells(i, __colseq_n),
                    cells(i, __colidx_half_l), dEdq.row(i)
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
 * Calculate the (negatives of the) tangential friction forces between
 * contacting cells.
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param R Cell radius, including the EPS.
 * @param E0 Elastic modulus of EPS.
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient.
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive. 
 * @param no_surface If true, omit the surface from the simulation.
 * @param dz_threshold If the z-orientation of the distance vector is smaller
 *                     than this value, compute the friction force in 2-D. 
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces.
 * @returns Array of forces and torques due to cell-cell friction. 
 */
template <typename T>
Array<T, Dynamic, 6> cellCellFrictionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                            const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                            const T R, const T E0,
                                            const int colidx_eta_cell_cell,
                                            const T cell_cell_coulomb_coeff,
                                            const bool no_surface, 
                                            const T dz_threshold, 
                                            const bool include_constraint = false)
{
    int n = cells.rows();       // Number of cells
    int m = neighbors.rows();   // Number of cell-cell neighbors
    T Req = R / 2;              // Equivalent radius 

    // If there are no neighboring pairs of cells, return zero
    if (m == 0)
        return Array<T, Dynamic, 6>::Zero(n, 6);

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, z-position, x-orientation,
    // y-orientation, z-orientation
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(n, 6);

    // Compute cell-cell distance and overlap for each pair of neighboring cells 
    Array<T, Dynamic, 1> distances = neighbors(Eigen::all, Eigen::seq(2, 4)).matrix().rowwise().norm().array();
    Array<T, Dynamic, 1> overlaps = 2 * R - distances; 

    // Compute cell-cell friction forces ...
    //
    // First get the angular velocity of each cell 
    Matrix<T, Dynamic, 3> angvels = Matrix<T, Dynamic, 3>::Zero(n, 3);
    for (int i = 0; i < n; ++i)
    {
        Matrix<T, 3, 1> u = cells(i, __colseq_n).matrix();
        Matrix<T, 3, 1> v = cells(i, __colseq_dn).matrix();  
        angvels.row(i) = u.cross(v).transpose(); 
    } 
   
    // Then, for each pair of neighboring cells ...  
    for (int k = 0; k < m; ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));
        Matrix<T, 3, 1> dij = neighbors(k, Eigen::seq(2, 4)).matrix();
        T si = neighbors(k, 5);                       // Cell-body coordinate along cell i
        T sj = neighbors(k, 6);                       // Cell-body coordinate along cell j
        Array<T, 3, 1> dijn = dij / distances(k);     // Normalized distance vector 
        T overlap = overlaps(k);                      // Cell-cell overlap

        // If the cells are overlapping ...
        if (overlap > 0)
        {
            // Get the contact point
            Matrix<T, 3, 1> ri = cells(i, __colseq_r).matrix(); 
            Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
            T half_li = cells(i, __colidx_half_l); 
            Matrix<T, 3, 1> rj = cells(j, __colseq_r).matrix(); 
            Matrix<T, 3, 1> nj = cells(j, __colseq_n).matrix();
            T half_lj = cells(j, __colidx_half_l); 
            Matrix<T, 3, 1> qij = ri + si * ni + dij / 2;

            // Determine the cell-cell friction coefficient 
            T eta = min(cells(i, colidx_eta_cell_cell), cells(j, colidx_eta_cell_cell));

            // Determine whether the frictional force should be constrained
            // to 2-D
            bool assume_2d = false; 
            if (!no_surface)
            {
                // If the distance vector has small z-component ...  
                if (abs(dijn(2)) < dz_threshold)
                {
                    Matrix<T, 3, 1> pi = ri + si * ni; 
                    Matrix<T, 3, 1> pj = rj + sj * nj; 
                    if (pi(2) < 1.01 * R && pj(2) < 1.01 * R)
                    {
                        assume_2d = true;
                    }
                }
            }

            // Get the relative velocity of the cells at the contact point
            Matrix<T, 3, 1> vi = cells(i, __colseq_dr).matrix() + si * cells(i, __colseq_dn).matrix();
            Matrix<T, 3, 1> vj = cells(j, __colseq_dr).matrix() + sj * cells(j, __colseq_dn).matrix();
            Matrix<T, 3, 1> rvij = vi - vj;

            // Get the rejection of the relative velocity along the tangential
            // direction
            if (assume_2d)
            {
                dij(2) = 0;
                T dist = dij.norm(); 
                dijn = (dij / dist).array();
                overlap = 2 * R - dist;
                rvij(2) = 0;  
            }
            Matrix<T, 3, 1> vijt = rvij - rvij.dot(dijn.matrix()) * dijn.matrix();

            // Get the cell-cell friction force on each cell 
            //
            // The force on cell i should act in the opposite direction as 
            // vijt; the force on cell j should act in the same direction as
            // vijt
            //
            // Use the Hertzian contact radius for the friction force, 
            // following Parteli et al. (2014)
            Matrix<T, 3, 1> force = -eta * sqrt(Req * overlap) * vijt;

            // Truncate the friction force according to Coulomb's law of 
            // friction, if desired
            if (cell_cell_coulomb_coeff > 0)
            {
                T force_norm = force.norm(); 
                T max_norm = cell_cell_coulomb_coeff * (4. / 3.) * E0 * sqrt(Req) * pow(overlap, 1.5); 
                if (force_norm > max_norm)
                    force *= (max_norm / force_norm); 
            }

            // Get the corresponding torques, which include the orientation
            // norm constraint
            Matrix<T, 3, 1> torque_i, torque_j; 
            if (!include_constraint)
            {
                torque_i = si * force;
                torque_j = -sj * force;
            }
            else 
            { 
                torque_i = (si * ni).cross(force); 
                torque_j = (sj * nj).cross(-force);
            } 

            // Update forces
            //
            // Note that this function should return the negatives of
            // the forces  
            dEdq(i, Eigen::seq(0, 2)) -= force.array().transpose();
            dEdq(i, Eigen::seq(3, 5)) -= torque_i.array().transpose();  
            dEdq(j, Eigen::seq(0, 2)) += force.array().transpose();
            dEdq(j, Eigen::seq(3, 5)) -= torque_j.array().transpose();
        }
    }

    return dEdq; 
}

/**
 * Compute the derivatives of the cell-cell interaction energy for each pair
 * of neighboring cells, with respect to each cell's position and orientation
 * coordinates.
 *
 * This function assumes that cell-cell interactions are purely repulsive.  
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
 * @param repulsion_prefactors Array of three pre-computed prefactors for 
 *                             cell-cell repulsion; see below for values. 
 * @param nz_threshold Threshold for judging whether each cell's z-orientation
 *                     is zero. 
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient.
 * @param no_surface If true, omit the surface from the simulation.
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces. 
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive. 
 * @returns Derivatives of the cell-cell adhesion energies with respect to  
 *          cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellCellInteractionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                               const T dt, const int iter,
                                               const T R, const T Rcell, const T E0, 
                                               const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                                               const T nz_threshold, 
                                               const FrictionMode friction_mode, 
                                               const int colidx_eta_cell_cell,
                                               const bool no_surface,
                                               const bool include_constraint = false, 
                                               const T cell_cell_coulomb_coeff = 1.0)
{
    const int n = cells.rows();       // Number of cells
    const int m = neighbors.rows();   // Number of cell-cell neighbors
    const T Req = R / 2;              // Equivalent radius

    // If there are no neighboring pairs of cells, return zero
    if (m == 0)
        return Array<T, Dynamic, 6>::Zero(n, 6);

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, z-position, x-orientation,
    // y-orientation, z-orientation
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(n, 6);

    // Maintain array of contact radii for all pairs of neighboring cells 
    Array<T, Dynamic, 1> radii = Array<T, Dynamic, 1>::Zero(m);  

    // Compute cell-cell distance and overlap for each pair of neighboring cells 
    Array<T, Dynamic, 1> distances = neighbors(Eigen::all, Eigen::seq(2, 4)).matrix().rowwise().norm().array();
    Array<T, Dynamic, 1> overlaps = 2 * R - distances;

    // For each pair of neighboring cells ...
    for (int k = 0; k < m; ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));

        // Check that the two cells overlap 
        if (overlaps(k) > 0)
        {
            // Extract the cell position and orientation vectors 
            Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
            Matrix<T, 3, 1> nj = cells(j, __colseq_n).matrix();
            T half_li = cells(i, __colidx_half_l);
            T half_lj = cells(j, __colidx_half_l);
            Matrix<T, 3, 1> dij = neighbors(k, Eigen::seq(2, 4)).matrix();
            T si = neighbors(k, 5);                      // Cell-body coordinate along cell i
            T sj = neighbors(k, 6);                      // Cell-body coordinate along cell j
            Array<T, 3, 1> dijn = dij / distances(k);    // Normalized distance vector
            Array<T, 2, 6> forces = Array<T, 2, 6>::Zero();
            T radius = 0; 
                
            // Cap overlap distance at 2 * (R - Rcell)
            T max_overlap = 2 * (R - Rcell);

            // Calculate the Hertzian repulsive force ...  
            //
            // Define the repulsive force magnitude and contact radius  
            T magnitude = 0; 

            // Case 1: the overlap is positive but less than 2 * (R - Rcell)
            // (i.e., it is limited to within the EPS coating)
            if (overlaps(k) <= max_overlap)
            {
                // The force magnitude is (4 / 3) * E0 * sqrt(R / 2) * pow(overlap, 1.5)
                magnitude = repulsion_prefactors(0) * pow(overlaps(k), 1.5);
                radius = sqrt(Req * overlaps(k)); 
            }
            // Case 2: the overlap is instead greater than 2 * R - 2 * Rcell
            // (i.e., it encroaches into the bodies of the two cells)
            else
            {
                // The (soft-shell) force magnitude is
                // (4 / 3) * E0 * sqrt(R / 2) * pow(2 * (R - Rcell), 1.5)
                magnitude = repulsion_prefactors(2);
                radius = sqrt(Req * max_overlap); 

                // The additional (hard-core) Hertzian force magnitude is
                // (4 / 3) * Ecell * sqrt(Rcell / 2) * pow(2 * Rcell + overlap - 2 * R, 1.5)
                magnitude += repulsion_prefactors(1) * pow(overlaps(k) - max_overlap, 1.5);
            }
        
            // Compute the derivatives of the cell-cell repulsion energy
            // between cells i and j
            Array<T, 3, 1> vij = magnitude * dijn; 
            forces(0, Eigen::seq(0, 2)) = vij; 
            forces(1, Eigen::seq(0, 2)) = -vij;
            if (!include_constraint)
            {
                forces(0, Eigen::seq(3, 5)) = si * vij;
                forces(1, Eigen::seq(3, 5)) = -sj * vij;
            }
            else    // Correct torques to account for orientation norm constraint
            {
                //T wi = ni.dot(-vij.matrix()); 
                //T wj = nj.dot(-vij.matrix()); 
                //forces(0, Eigen::seq(3, 5)) = si * (wi * ni.array() + vij); 
                //forces(1, Eigen::seq(3, 5)) = sj * (-wj * nj.array() - vij); 
                forces(0, Eigen::seq(3, 5)) = (si * ni).cross(vij.matrix()).array(); 
                forces(1, Eigen::seq(3, 5)) = (sj * nj).cross(-vij.matrix()).array();  
            }

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
                        dij.array(), si, sj, forces
                    );
                    throw std::runtime_error("Found nan in repulsive forces"); 
                }
            #endif

            // Update forces and store contact radius
            dEdq.row(i) += forces.row(0); 
            dEdq.row(j) += forces.row(1);
            radii(k) = radius;  
        }
    }

    // Compute cell-cell friction forces, if desired 
    if (friction_mode != FrictionMode::NONE)
    {
        dEdq += cellCellFrictionForces<T>(
            cells, neighbors, R, E0, colidx_eta_cell_cell, cell_cell_coulomb_coeff,
            no_surface, 1e-3, include_constraint
        ); 
    }

    return dEdq; 
}

/**
 * Compute the derivatives of the cell-cell interaction energy for each pair
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
 * @param repulsion_prefactors Array of three pre-computed prefactors for 
 *                             cell-cell repulsion; see below for values. 
 * @param nz_threshold Threshold for judging whether each cell's z-orientation
 *                     is zero. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR_ISOTROPIC (1), or JKR_ANISOTROPIC
 *                      (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param jkr_data Pre-computed values for calculating JKR forces. 
 * @param colidx_gamma Column index for cell-cell adhesion surface energy 
 *                     density.
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient.
 * @param no_surface If true, omit the surface from the simulation.
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces. 
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive. 
 * @returns Derivatives of the cell-cell adhesion energies with respect to  
 *          cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 6> cellCellInteractionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                               const T dt, const int iter,
                                               const T R, const T Rcell, const T E0, 
                                               const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                                               const T nz_threshold, 
                                               const AdhesionMode adhesion_mode,
                                               std::unordered_map<std::string, T>& adhesion_params,
                                               std::unique_ptr<JKRData<T> >& jkr_data,
                                               const int colidx_gamma,
                                               const FrictionMode friction_mode, 
                                               const int colidx_eta_cell_cell,
                                               const bool no_surface,
                                               const bool include_constraint = false, 
                                               const T cell_cell_coulomb_coeff = 1.0)
{
    const int n = cells.rows();       // Number of cells
    const int m = neighbors.rows();   // Number of cell-cell neighbors
    const T Req = R / 2;              // Equivalent radius

    // If there are no neighboring pairs of cells, return zero
    if (m == 0)
        return Array<T, Dynamic, 6>::Zero(n, 6);

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, z-position, x-orientation,
    // y-orientation, z-orientation
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(n, 6);

    // Maintain array of contact radii for all pairs of neighboring cells 
    Array<T, Dynamic, 1> radii = Array<T, Dynamic, 1>::Zero(m);  

    // Compute cell-cell distance and overlap for each pair of neighboring cells 
    Array<T, Dynamic, 1> distances = neighbors(Eigen::all, Eigen::seq(2, 4)).matrix().rowwise().norm().array();
    Array<T, Dynamic, 1> overlaps = 2 * R - distances;

    #ifdef CHECK_ADHESION_MODE
        if (adhesion_mode == AdhesionMode::NONE)
        {
            throw std::runtime_error(
                "Cell-cell adhesion forces cannot be calculated without JKR "
                "adhesion mode"
            ); 
        }
    #endif

    // For each pair of neighboring cells ...
    for (int k = 0; k < m; ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1));

        // Check that the two cells overlap 
        if (overlaps(k) > 0)
        {
            // Extract the cell position and orientation vectors
            Matrix<T, 3, 1> ri = cells(i, __colseq_r).matrix();
            Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
            Matrix<T, 3, 1> rj = cells(j, __colseq_r).matrix();
            Matrix<T, 3, 1> nj = cells(j, __colseq_n).matrix();
            T half_li = cells(i, __colidx_half_l);
            T half_lj = cells(j, __colidx_half_l);
            Matrix<T, 3, 1> dij = neighbors(k, Eigen::seq(2, 4)).matrix();
            T si = neighbors(k, 5);                      // Cell-body coordinate along cell i
            T sj = neighbors(k, 6);                      // Cell-body coordinate along cell j
            Array<T, 3, 1> dijn = dij / distances(k);    // Normalized distance vector
            Array<T, 2, 6> forces = Array<T, 2, 6>::Zero();
            T radius = 0; 
                
            // Cap overlap distance at 2 * (R - Rcell)
            T max_overlap = 2 * (R - Rcell);

            // Get the surface adhesion energy density 
            T gamma = min(cells(i, colidx_gamma), cells(j, colidx_gamma));

            // If the surface adhesion energy density is nonzero ... 
            if (gamma > 0)
            {
                // Use pre-computed values if desired
                bool use_precomputed_values = jkr_data->initialized;  
                if (use_precomputed_values)
                {
                    auto result = jkr_data->getJKRForces(
                        ri, ni, half_li, rj, nj, half_lj, dij, R, E0, gamma, 
                        si, sj, include_constraint, max_overlap   
                    ); 
                    forces = result.first;
                    radius = result.second;
                }
                else    // Otherwise, compute forces from scratch  
                {
                    if (adhesion_mode == AdhesionMode::JKR_ISOTROPIC)
                    {
                        auto result = forcesIsotropicJKRLagrange<T, 3, 30>(
                            ni, nj, dij, R, E0, gamma, si, sj, include_constraint,
                            max_overlap
                        );
                        forces = result.first;
                        radius = result.second;
                    }
                    else
                    {
                        bool calibrate_endpoint_radii = static_cast<bool>(
                            adhesion_params["calibrate_endpoint_radii"]
                        );
                        T min_aspect_ratio = adhesion_params["min_aspect_ratio"];
                        T max_aspect_ratio = adhesion_params["max_aspect_ratio"];  
                        T project_tol = adhesion_params["ellipsoid_project_tol"]; 
                        int project_max_iter = static_cast<int>(
                            adhesion_params["ellipsoid_project_max_iter"]
                        );
                        T brent_tol = adhesion_params["brent_tol"]; 
                        int brent_max_iter = static_cast<int>(
                            adhesion_params["brent_max_iter"]
                        );
                        T init_bracket_dx = adhesion_params["init_bracket_dx"]; 
                        int n_tries_bracket = static_cast<int>(
                            adhesion_params["n_tries_bracket"]
                        ); 
                        T imag_tol = adhesion_params["jkr_imag_tol"]; 
                        T aberth_tol = adhesion_params["jkr_aberth_tol"];  
                        auto result = forcesAnisotropicJKRLagrange<T, 3, 30>(
                            ri, ni, half_li, rj, nj, half_lj, dij, R, E0,
                            gamma, si, sj, include_constraint, max_overlap,
                            calibrate_endpoint_radii, min_aspect_ratio,
                            max_aspect_ratio, project_tol, project_max_iter,
                            brent_tol, brent_max_iter, init_bracket_dx, 
                            n_tries_bracket, imag_tol, aberth_tol
                        );
                        forces = result.first;
                        radius = result.second; 
                    } 
                }
            } 
            // Otherwise, calculate the Hertzian repulsive force 
            else 
            {
                // Define the repulsive force magnitude and contact radius  
                T magnitude = 0; 

                // Case 1: the overlap is positive but less than 2 * (R - Rcell)
                // (i.e., it is limited to within the EPS coating)
                if (overlaps(k) <= 2 * (R - Rcell))
                {
                    // The force magnitude is (4 / 3) * E0 * sqrt(R / 2) * pow(overlap, 1.5)
                    magnitude = repulsion_prefactors(0) * pow(overlaps(k), 1.5);
                    radius = sqrt(Req * overlaps(k)); 
                }
                // Case 2: the overlap is instead greater than 2 * R - 2 * Rcell
                // (i.e., it encroaches into the bodies of the two cells)
                else
                {
                    // The (soft-shell) force magnitude is
                    // (4 / 3) * E0 * sqrt(R / 2) * pow(2 * (R - Rcell), 1.5)
                    magnitude = repulsion_prefactors(2);
                    radius = sqrt(Req * 2 * (R - Rcell)); 
                }
            
                // Compute the derivatives of the cell-cell repulsion energy
                // between cells i and j
                Array<T, 3, 1> vij = magnitude * dijn; 
                forces(0, Eigen::seq(0, 2)) = vij; 
                forces(1, Eigen::seq(0, 2)) = -vij;
                if (!include_constraint)
                {
                    forces(0, Eigen::seq(3, 5)) = si * vij;
                    forces(1, Eigen::seq(3, 5)) = -sj * vij;
                }
                else    // Correct torques to account for orientation norm constraint
                {
                    //T wi = ni.dot(-vij.matrix()); 
                    //T wj = nj.dot(-vij.matrix()); 
                    //forces(0, Eigen::seq(3, 5)) = si * (wi * ni.array() + vij); 
                    //forces(1, Eigen::seq(3, 5)) = sj * (-wj * nj.array() - vij); 
                    forces(0, Eigen::seq(3, 5)) = (si * ni).cross(vij.matrix()).array(); 
                    forces(1, Eigen::seq(3, 5)) = (sj * nj).cross(-vij.matrix()).array();  
                }
            }
            #ifdef DEBUG_CHECK_CELL_CELL_ADHESIVE_FORCES_NAN
                if (forces.isNaN().any() || forces.isInf().any())
                {
                    std::cerr << "Iteration " << iter
                              << ": Found nan in adhesive forces between cells " 
                              << i << " and " << j << std::endl;
                    std::cerr << "Timestep: " << dt << std::endl;
                    pairLagrangianForcesSummary<T>(
                        cells(i, __colseq_r), cells(i, __colseq_n),
                        cells(i, __colseq_dr), cells(i, __colseq_dn), 
                        cells(i, __colidx_half_l),
                        cells(j, __colseq_r), cells(j, __colseq_n),
                        cells(j, __colseq_dr), cells(j, __colseq_dn), 
                        cells(j, __colidx_half_l),
                        dij.array(), si, sj, forces
                    );
                    throw std::runtime_error("Found nan in adhesive forces"); 
                }
            #endif

            // Update forces and store contact radius
            dEdq.row(i) += forces.row(0); 
            dEdq.row(j) += forces.row(1);
            radii(k) = radius;  

            // If the overlap exceeds the maximum, add an additional 
            // hard-core repulsive force 
            if (overlaps(k) > 2 * (R - Rcell))
            {
                // The additional (hard-core) Hertzian force magnitude is
                // (4 / 3) * Ecell * sqrt(Rcell / 2) * pow(2 * Rcell + overlap - 2 * R, 1.5)
                T magnitude = repulsion_prefactors(1) * pow(overlaps(k) - 2 * (R - Rcell), 1.5);
                
                // Compute the derivatives of the cell-cell repulsion energy
                // between cells i and j
                Array<T, 3, 1> vij = magnitude * dijn; 
                Array<T, 2, 6> forces_hard;
                forces_hard(0, Eigen::seq(0, 2)) = vij; 
                forces_hard(1, Eigen::seq(0, 2)) = -vij;
                if (!include_constraint)
                {
                    forces_hard(0, Eigen::seq(3, 5)) = si * vij;
                    forces_hard(1, Eigen::seq(3, 5)) = -sj * vij;
                }
                else    // Correct torques to account for orientation norm constraint
                {
                    //T wi = ni.dot(-vij.matrix()); 
                    //T wj = nj.dot(-vij.matrix()); 
                    //forces_hard(0, Eigen::seq(3, 5)) = si * (wi * ni.array() + vij); 
                    //forces_hard(1, Eigen::seq(3, 5)) = sj * (-wj * nj.array() - vij);
                    forces_hard(0, Eigen::seq(3, 5)) = (si * ni).cross(vij.matrix()).array(); 
                    forces_hard(1, Eigen::seq(3, 5)) = (sj * nj).cross(-vij.matrix()).array();  
                }
                #ifdef DEBUG_CHECK_CELL_CELL_REPULSIVE_FORCES_NAN
                    if (forces_hard.isNaN().any() || forces_hard.isInf().any())
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
                            dij.array(), si, sj, forces_hard
                        );
                        throw std::runtime_error("Found nan in repulsive forces"); 
                    }
                #endif

                // Update forces 
                dEdq.row(i) += forces_hard.row(0); 
                dEdq.row(j) += forces_hard.row(1);
            }
        }
    }

    // Compute cell-cell friction forces, if desired 
    if (friction_mode != FrictionMode::NONE)
    {
        dEdq += cellCellFrictionForces<T>(
            cells, neighbors, R, E0, colidx_eta_cell_cell, cell_cell_coulomb_coeff,
            no_surface, 1e-3, include_constraint
        ); 
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
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param repulsion_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero.
 * @param nz_threshold Threshold for judging whether each cell's z-orientation
 *                     is zero. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR_ISOTROPIC (1), or JKR_ANISOTROPIC
 *                      (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param jkr_data Pre-computed values for calculating JKR forces. 
 * @param colidx_gamma Column index for cell-cell adhesion surface energy 
 *                     density.
 * @param friction_mode Choice of model for cell-cell friction. Can be NONE (0)
 *                      or KINETIC (1). 
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient. 
 * @param no_surface If true, omit the surface from the simulation. 
 * @param multithread If true, use multithreading.
 * @param include_constraint If true, add the contribution from the Lagrange 
 *                           multiplier due to the orientation vector norm 
 *                           constraint to the forces. 
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive. 
 * @returns Array of forces due to cell-cell and cell-surface interactions. 
 */
template <typename T>
Array<T, Dynamic, 6> getConservativeForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                           const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                           const T dt, const int iter,
                                           const T R, const T Rcell, const T E0,
                                           const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                                           const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                           const T nz_threshold, 
                                           const AdhesionMode adhesion_mode,
                                           std::unordered_map<std::string, T>& adhesion_params,
                                           std::unique_ptr<JKRData<T> >& jkr_data,
                                           const int colidx_gamma,
                                           const FrictionMode friction_mode, 
                                           const int colidx_eta_cell_cell,  
                                           const bool no_surface,
                                           const bool multithread,
                                           const bool include_constraint = false,
                                           const T cell_cell_coulomb_coeff = 1.0)
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

    // Compute the cell-cell interaction forces
    Array<T, Dynamic, 6> dEdq_cell_interaction;
    if (adhesion_mode == AdhesionMode::NONE)
    {
        dEdq_cell_interaction = cellCellInteractionForces<T>(
            cells, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors,
            nz_threshold, friction_mode, colidx_eta_cell_cell, no_surface,
            include_constraint, cell_cell_coulomb_coeff
        );
    }
    else
    {
        dEdq_cell_interaction = cellCellInteractionForces<T>(
            cells, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors,
            nz_threshold, adhesion_mode, adhesion_params, jkr_data, colidx_gamma,
            friction_mode, colidx_eta_cell_cell, no_surface, include_constraint,
            cell_cell_coulomb_coeff
        );
    }
    
    // Compute the cell-surface interaction forces (if present)
    Array<T, Dynamic, 6> dEdq_surface_repulsion = Array<T, Dynamic, 6>::Zero(n, 6);
    Array<T, Dynamic, 6> dEdq_surface_adhesion = Array<T, Dynamic, 6>::Zero(n, 6); 
    if (!no_surface)
    { 
        dEdq_surface_repulsion = cellSurfaceRepulsionForces<T>(
            cells, dt, iter, ss, R, E0, assume_2d, multithread, include_constraint
        );
        dEdq_surface_adhesion = cellSurfaceAdhesionForces<T>(
            cells, dt, iter, ss, R, assume_2d, multithread, include_constraint
        );
    }

    // Combine the forces accordingly 
    return -dEdq_cell_interaction - dEdq_surface_repulsion - dEdq_surface_adhesion; 
}

/* -------------------------------------------------------------------- // 
//         ADDITIONAL NEWTONIAN FORCES (FOR VERLET INTEGRATION)         // 
// -------------------------------------------------------------------- */ 

/**
 * Compute the forces and torques due to ambient viscosity on each cell.
 *
 * @param cells Existing population of cells.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output.
 * @param R Cell radius.
 * @returns Array of forces and torques due to ambient viscosity. 
 */
template <typename T>
Array<T, Dynamic, 6> ambientViscosityForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                            const T dt, const int iter, const T R)
{
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(cells.rows(), 6);

    // For each cell ...
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Get the angular velocity of the cell
        Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
        Matrix<T, 3, 1> dni = cells(i, __colseq_dn).matrix();  
        Matrix<T, 3, 1> angvel = ni.cross(dni);

        // Compute the force and torque
        T eta = cells(i, __colidx_eta0);
        T li = pow(cells(i, __colidx_l), 3); 
        dEdq(i, Eigen::seq(0, 2)) = eta * li * cells(i, __colseq_dr);
        dEdq(i, Eigen::seq(3, 5)) = eta * li * li * li * angvel.array() / 12; 
    }

    return dEdq; 
}

/**
 * Compute the forces and torques due to cell-surface friction on each cell. 
 *
 * @param cells Existing population of cells.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output.
 * @param ss Cell-body coordinates at which each cell-surface overlap is zero. 
 * @param R Cell radius.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero.
 * @param multithread If true, use multithreading. 
 * @returns Array of forces and torques due to cell-surface friction. 
 */
template <typename T>
Array<T, Dynamic, 6> cellSurfaceFrictionForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                               const T dt, const int iter,
                                               const Ref<const Array<T, Dynamic, 1> >& ss,
                                               const T R,
                                               const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                               const bool multithread)
{
    Array<T, Dynamic, 6> dEdq = Array<T, Dynamic, 6>::Zero(cells.rows(), 6);

    // For each cell ...
    #pragma omp parallel for if(multithread)
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Get the angular velocity of the cell
        Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
        Matrix<T, 3, 1> dni = cells(i, __colseq_dn).matrix();  
        Matrix<T, 3, 1> angvel = ni.cross(dni);

        // Get the projection of the cell's translational velocity onto the
        // xy-plane 
        Matrix<T, 3, 1> q; 
        q << cells(i, __colidx_drx), cells(i, __colidx_dry), 0;  

        // Get the projection of the cross product of the angular velocity 
        // with the orientation vector onto the xy-plane 
        Matrix<T, 3, 1> m;  
        m << angvel(1) * ni(2) - angvel(2) * ni(1), 
             angvel(2) * ni(0) - angvel(0) * ni(2), 
             0;

        // If the z-coordinate of the cell's orientation is zero ...
        T eta = cells(i, __colidx_eta1);  
        if (assume_2d(i))
        {
            // In this case, the force is nonzero only if phi > 0
            T phi = R - cells(i, __colidx_rz);
            if (phi > 0)
            {
                // Calculate the force ... 
                T area = sqrt(R) * sqrt(phi) * cells(i, __colidx_l); 
                dEdq(i, 0) = eta * q(0) * area / R; 
                dEdq(i, 1) = eta * q(1) * area / R;

                // ... and the torque, which depends on the cross product 
                // of the orientation vector with m 
                T cross_z = ni(0) * m(1) - ni(1) * m(0);
                T l2 = cells(i, __colidx_l) * cells(i, __colidx_l);  
                dEdq(i, 5) = eta * cross_z * area * l2 / (12 * R); 
            }
        }
        // Otherwise ...
        else
        {
            // Calculate the force ... 
            T int1 = areaIntegral1<T>(    // Cell-surface contact area 
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), ss(i)
            );
            T int2 = areaIntegral2<T>(    // Integral of s * a_i(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), ss(i)
            );
            T int3 = areaIntegral3<T>(    // Integral of s^2 * a_i(s)
                cells(i, __colidx_rz), cells(i, __colidx_nz), R,
                cells(i, __colidx_half_l), ss(i)
            );
            dEdq(i, 0) = eta * (q(0) * int1 + m(0) * int2) / R; 
            dEdq(i, 1) = eta * (q(1) * int1 + m(1) * int2) / R; 

            // ... and the torque, which depends on the cross products of 
            // the orientation vector with q and with m
            Matrix<T, 3, 1> cross1 = ni.cross(q); 
            Matrix<T, 3, 1> cross2 = ni.cross(m);  
            dEdq(i, 3) = eta * (cross1(0) * int2 + cross2(0) * int3) / R; 
            dEdq(i, 4) = eta * (cross1(1) * int2 + cross2(1) * int3) / R;
            dEdq(i, 5) = eta * (cross1(2) * int2 + cross2(2) * int3) / R;  
        }
    }

    return dEdq; 
}

/* -------------------------------------------------------------------- // 
//                      ADDITIONAL UTILITY FUNCTIONS                    //
// -------------------------------------------------------------------- */
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
 * Cell-cell friction is not incorporated in this function. 
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param dt Timestep. Only used for debugging output.
 * @param iter Iteration number. Only used for debugging output. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS.
 * @param repulsion_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
 * @param assume_2d If the i-th entry is true, assume that the i-th cell's
 *                  z-orientation is zero.
 * @param nz_threshold Threshold for judging whether each cell's z-orientation
 *                     is zero. 
 * @param noise Noise to be added to each generalized force used to compute
 *              the velocities.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), JKR_ISOTROPIC (1), or JKR_ANISOTROPIC
 *                      (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param jkr_data Pre-computed values for calculating JKR forces. 
 * @param colidx_gamma Column index for cell-cell adhesion surface energy 
 *                     density.
 * @param friction_mode Choice of model for cell-cell friction. Can be NONE (0)
 *                      or KINETIC (1). 
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient.
 * @param no_surface If true, omit the surface from the simulation. 
 * @param multithread If true, use multithreading.
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive.
 * @param cell_surface_coulomb_coeff Limiting cell-surface friction coefficient
 *                                   due to Coulomb's law. Ignored if not
 *                                   positive. 
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 6> getVelocities(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                   const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                   const T dt, const int iter, const T R,
                                   const T Rcell, const T E0, 
                                   const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                                   const Ref<const Array<int, Dynamic, 1> >& assume_2d,
                                   const T nz_threshold, 
                                   const Ref<const Array<T, Dynamic, 6> >& noise,
                                   const AdhesionMode adhesion_mode,
                                   std::unordered_map<std::string, T>& adhesion_params,
                                   std::unique_ptr<JKRData<T> >& jkr_data,
                                   const int colidx_gamma,
                                   const FrictionMode friction_mode,
                                   const int colidx_eta_cell_cell,  
                                   const bool no_surface, const bool multithread,
                                   const T cell_cell_coulomb_coeff = 1.0,
                                   const T cell_surface_coulomb_coeff = 1.0)
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
        cells, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors, 
        assume_2d, nz_threshold, adhesion_mode, adhesion_params, jkr_data,
        colidx_gamma, friction_mode, colidx_eta_cell_cell, no_surface,
        multithread, false, cell_cell_coulomb_coeff
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
            T K, L; 

            // If Coulomb's law for surface friction is not to be enforced ...  
            if (cell_surface_coulomb_coeff == 0)
            {
                T a = sqrt(R * (R - cells(i, __colidx_rz)));
                T eta_combined = cells(i, __colidx_eta0) + cells(i, __colidx_eta1) * a / R;  
                K = eta_combined * cells(i, __colidx_l);  
                L = K * cells(i, __colidx_l) * cells(i, __colidx_l) / 12;
            }
            // Otherwise ... 
            else 
            {
                // Calculate the ambient viscosity force prefactors  
                T K_vis = cells(i, __colidx_eta0) * cells(i, __colidx_l);
                T L_vis = K_vis * cells(i, __colidx_l) * cells(i, __colidx_l) / 12;

                // Truncate the cell-surface friction coefficient according to 
                // Coulomb's law, based on its current velocity and overlap 
                T a = sqrt(R * (R - cells(i, __colidx_rz)));
                T K_fric = cells(i, __colidx_eta1) * cells(i, __colidx_l) * a / R;
                T drxy = cells(i, Eigen::seq(__colidx_drx, __colidx_dry)).matrix().norm(); 
                T friction_norm = K_fric * drxy;
                T max_norm = cell_surface_coulomb_coeff * (
                    2 * E0 * (R - cells(i, __colidx_rz)) * cells(i, __colidx_l)
                ); 
                if (friction_norm > max_norm)
                {
                    T eta1_truncated = max_norm * R / (drxy * cells(i, __colidx_l) * a);
                    K_fric = eta1_truncated * cells(i, __colidx_l) * a / R; 
                } 
                T L_fric = K_fric * cells(i, __colidx_l) * cells(i, __colidx_l) / 12;

                // Calculate the composite viscosity force prefactors 
                K = K_vis + K_fric; 
                L = L_vis + L_fric;
            }

            // Calculate the Lagrange multiplier and the velocities 
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
            const T rz = cells(i, __colidx_rz); 
            const T nz = cells(i, __colidx_nz); 
            const T l = cells(i, __colidx_l); 
            const T half_l = cells(i, __colidx_half_l);
            const T eta0 = cells(i, __colidx_eta0); 
            T eta1 = cells(i, __colidx_eta1);
            std::tuple<T, T, T> integrals = areaIntegrals<T>(rz, nz, R, half_l, ss(i)); 

            // Truncate the cell-surface friction coefficient according to 
            // Coulomb's law
            if (cell_surface_coulomb_coeff > 0)
            {
                // Calculate the cell-surface friction force according to 
                // the current cell velocity
                T drxy = cells(i, Eigen::seq(__colidx_drx, __colidx_dry)).matrix().norm();
                T dnxy = cells(i, Eigen::seq(__colidx_dnx, __colidx_dny)).matrix().norm();  
                T friction_norm = std::get<0>(integrals) * drxy; 
                friction_norm += std::get<1>(integrals) * dnxy;
                friction_norm *= (eta1 / R);

                // Calculate the cell-surface repulsive force and truncate 
                // the cell-surface friction coefficient accordingly  
                T max_norm = (1 - nz * nz) * integral1<T>(rz, nz, R, half_l, 1, ss(i));
                max_norm += sqrt(R) * nz * nz * integral1<T>(rz, nz, R, half_l, 0.5, ss(i));
                max_norm *= (2 * E0 * cell_surface_coulomb_coeff);  
                if (friction_norm > max_norm)
                    eta1 = max_norm * R / (std::get<0>(integrals) * drxy + std::get<1>(integrals) * dnxy); 
            }

            // Solve the linear system explicitly, using back-substitution
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

            #ifdef DEBUG_CHECK_VELOCITIES_NAN
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
                                                   << rz << ")" << std::endl
                              << "Cell orientation = (" << cells(i, __colidx_nx) << ", "
                                                        << cells(i, __colidx_ny) << ", "
                                                        << nz << ")" << std::endl
                              << "Cell half-length = " << half_l << std::endl 
                              << "Cell translational velocity = (" << velocities(i, 0) << ", "
                                                                   << velocities(i, 1) << ", "
                                                                   << velocities(i, 2) << ")" << std::endl
                              << "Cell orientational velocity = (" << velocities(i, 3) << ", "
                                                                   << velocities(i, 4) << ", "
                                                                   << velocities(i, 5) << ")" << std::endl
                              << "Constraint variable = " << lambda << std::endl
                              << "Composite viscosity force matrix = " 
                              << compositeViscosityForceMatrix<T>(
                                     rz, nz, l, half_l, ss(i), eta0, eta1, R,
                                     i, dt, iter
                                 )
                              << std::endl;
                    throw std::runtime_error("Found nan in velocities"); 
                }
            #endif
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
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param repulsion_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
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
 *                      Can be NONE (0), JKR_ISOTROPIC (1), or JKR_ANISOTROPIC
 *                      (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param jkr_data Pre-computed values for calculating JKR forces.  
 * @param colidx_gamma Column index for cell-cell adhesion surface energy 
 *                     density.
 * @param friction_mode Choice of model for cell-cell friction. Can be NONE (0)
 *                      or KINETIC (1). 
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient.
 * @param no_surface If true, omit the surface from the simulation.
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive.
 * @param cell_surface_coulomb_coeff Limiting cell-surface friction coefficient
 *                                   due to Coulomb's law. Ignored if not
 *                                   positive. 
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
                           const T dt, const int iter, const T R,
                           const T Rcell, const T E0, 
                           const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                           const T nz_threshold, const T max_rxy_noise,
                           const T max_rz_noise, const T max_nxy_noise,
                           const T max_nz_noise, boost::random::mt19937& rng,
                           boost::random::uniform_01<>& uniform_dist,
                           const AdhesionMode adhesion_mode,
                           std::unordered_map<std::string, T>& adhesion_params,
                           std::unique_ptr<JKRData<T> >& jkr_data,
                           const int colidx_gamma, const FrictionMode friction_mode, 
                           const int colidx_eta_cell_cell, const bool no_surface,
                           const T cell_cell_coulomb_coeff = 1.0,
                           const T cell_surface_coulomb_coeff = 1.0,
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
    #ifdef DEBUG_WARN_CELL_BODIES_CONTACTING
        for (int k = 0; k < neighbors.rows(); ++k)
        {
            if (neighbors(k, Eigen::seq(2, 4)).matrix().norm() < 0.99 * (2 * Rcell))
            {
                int i = neighbors(k, 0); 
                int j = neighbors(k, 1); 
                std::cout << "[WARN] Iteration " << iter
                          << ": Found cell body contact between cells " 
                          << i << " and " << j << std::endl;
                std::cout << "Timestep: " << dt << std::endl; 
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
                //throw std::runtime_error("Found near-zero distance");
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
        cells, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors, 
        assume_2d, nz_threshold, noise, adhesion_mode, adhesion_params,
        jkr_data, colidx_gamma, friction_mode, colidx_eta_cell_cell, no_surface,
        multithread, cell_cell_coulomb_coeff, cell_surface_coulomb_coeff
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
            cells_i, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors,
            assume_2d, nz_threshold, noise, adhesion_mode, adhesion_params,
            jkr_data, colidx_gamma, friction_mode, colidx_eta_cell_cell,
            no_surface, multithread, cell_cell_coulomb_coeff,
            cell_surface_coulomb_coeff
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

/**
 * Run one step of the velocity Verlet method for the given timestep. 
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
 * @param rho Density of each cell. 
 * @param repulsion_prefactors Array of three pre-computed prefactors for
 *                             cell-cell repulsion forces.
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
 *                      Can be NONE (0), JKR_ISOTROPIC (1), or JKR_ANISOTROPIC
 *                      (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param jkr_data Pre-computed values for calculating JKR forces. 
 * @param colidx_gamma Column index for cell-cell adhesion surface energy 
 *                     density.
 * @param friction_mode Choice of model for cell-cell friction. Can be NONE (0)
 *                      or KINETIC (1). 
 * @param colidx_eta_cell_cell Column index for cell-cell friction coefficient. 
 * @param no_surface If true, omit the surface from the simulation.
 * @param cell_cell_coulomb_coeff Limiting cell-cell friction coefficient
 *                                due to Coulomb's law. Ignored if not positive.
 * @param cell_surface_coulomb_coeff Limiting cell-surface friction coefficient
 *                                   due to Coulomb's law. Ignored if not
 *                                   positive. 
 * @param n_start_multithread Minimum number of cells at which to start using
 *                            multithreading. 
 * @returns Updated population of cells. 
 */
template <typename T>
Array<T, Dynamic, Dynamic> stepVelocityVerlet(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                              const Ref<const Array<T, Dynamic, 7> >& neighbors,
                                              const T dt, const int iter, const T R,
                                              const T Rcell, const T E0, const T rho,
                                              const Ref<const Array<T, 3, 1> >& repulsion_prefactors,
                                              const T nz_threshold, 
                                              const T max_rxy_noise, 
                                              const T max_rz_noise, 
                                              const T max_nxy_noise, 
                                              const T max_nz_noise, 
                                              boost::random::mt19937& rng, 
                                              boost::random::uniform_01<>& uniform_dist, 
                                              const AdhesionMode adhesion_mode,
                                              std::unordered_map<std::string, T>& adhesion_params,
                                              std::unique_ptr<JKRData<T> >& jkr_data,
                                              const int colidx_gamma,
                                              const FrictionMode friction_mode, 
                                              const int colidx_eta_cell_cell, 
                                              const bool no_surface,
                                              const T cell_cell_coulomb_coeff = 1.0,
                                              const T cell_surface_coulomb_coeff = 1.0,
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
    #ifdef DEBUG_WARN_CELL_BODIES_CONTACTING
        for (int k = 0; k < neighbors.rows(); ++k)
        {
            if (neighbors(k, Eigen::seq(2, 4)).matrix().norm() < 0.99 * (2 * Rcell))
            {
                int i = neighbors(k, 0); 
                int j = neighbors(k, 1); 
                std::cout << "[WARN] Iteration " << iter
                          << ": Found cell body contact between cells " 
                          << i << " and " << j << std::endl;
                std::cout << "Timestep: " << dt << std::endl; 
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
                //throw std::runtime_error("Found near-zero distance");
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

    // Get the mass of each cell 
    Array<T, Dynamic, 1> masses(n); 
    for (int i = 0; i < n; ++i)
    {
        T length = cells(i, __colidx_l); 
        T volume = boost::math::constants::pi<T>() * ((4. / 3.) * pow(R, 3) + R * R * length);
        masses(i) = rho * volume; 
    }

    // Get the moments of inertia along the long and short axes of each cell 
    Array<T, Dynamic, 2> moments = Array<T, Dynamic, 2>::Zero(n, 2); 
    for (int i = 0; i < n; ++i)
    {
        T diam = 2 * R; 
        T d2 = diam * diam; 
        T d3 = d2 * diam; 
        T d4 = d3 * diam; 
        T length = cells(i, __colidx_l);
        T l2 = length * length; 
        T l3 = l2 * length; 
        moments(i, 0) = d4 * (length / 32 + diam / 60);
        moments(i, 1) = d2 * (4 * d3 + 15 * length * d2 + 20 * l2 * diam + 10 * l3) / 480;  
    }
    moments *= (rho * boost::math::constants::pi<T>());

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

    // Get the angular velocity of each cell 
    Array<T, Dynamic, 3> angvels = Array<T, Dynamic, 3>::Zero(n, 3);
    for (int i = 0; i < n; ++i)
    {
        Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
        Matrix<T, 3, 1> dni = cells(i, __colseq_dn).matrix();  
        angvels.row(i) = ni.cross(dni).array().transpose(); 
    }

    // Update cell positions, orientations, and velocities ... 
    Array<T, Dynamic, Dynamic> cells_new(cells); 
    bool multithread = (n >= n_start_multithread); 

    // Compute all conservative forces with the orientation vector norm 
    // constraint (with or without cell-cell friction), and add noise 
    Array<T, Dynamic, 6> forces = getConservativeForces<T>(
        cells, neighbors, dt, iter, R, Rcell, E0, repulsion_prefactors, 
        assume_2d, nz_threshold, adhesion_mode, adhesion_params, jkr_data,
        colidx_gamma, friction_mode, colidx_eta_cell_cell, no_surface,
        multithread, true, cell_cell_coulomb_coeff
    );
    forces += noise; 

    // Compute the ambient viscosity and cell-surface friction forces
    forces -= ambientViscosityForces<T>(cells, dt, iter, R);
    if (!no_surface)
    {
        Array<T, Dynamic, 6> cell_surface_friction = cellSurfaceFrictionForces<T>(
            cells, dt, iter, ss, R, assume_2d, multithread
        );

        // Truncate the cell-surface friction forces according to Coulomb's
        // law of friction, if desired 
        if (cell_surface_coulomb_coeff > 0)
        {
            Array<T, Dynamic, 6> cell_surface_repulsive = cellSurfaceRepulsionForces<T>(
                cells, dt, iter, ss, R, E0, assume_2d, multithread, true
            ); 
            for (int i = 0; i < n; ++i)
            {
                T friction_norm = cell_surface_friction.row(i).matrix().norm(); 
                T max_norm = cell_surface_coulomb_coeff * cell_surface_repulsive.row(i).matrix().norm(); 
                if (friction_norm > max_norm)
                    cell_surface_friction.row(i) *= (max_norm / friction_norm); 
            } 
        }
        forces -= cell_surface_friction;
    }

    // Decompose the torques into parallel and perpendicular contributions
    Array<T, Dynamic, 3> torques = forces(Eigen::all, Eigen::seq(3, 5)); 
    Array<T, Dynamic, 3> torques_par(n, 3), torques_perp(n, 3);  
    for (int i = 0; i < n; ++i)
    {
        Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix(); 
        torques_par.row(i) = torques.row(i).matrix().dot(ni) * ni;
        torques_perp.row(i) = torques.row(i) - torques_par.row(i); 
    }

    // Calculate the half-updated velocity and angular velocity of each cell
    Array<T, Dynamic, 3> delta1 = forces(Eigen::all, Eigen::seq(0, 2)).colwise() / masses;
    Array<T, Dynamic, 3> delta2 = torques_par.colwise() / moments.col(0); 
    delta2 += torques_perp.colwise() / moments.col(1);
    Array<T, Dynamic, 3> velocities_half = cells(Eigen::all, __colseq_dr) + 0.5 * dt * delta1; 
    Array<T, Dynamic, 3> angvels_half = angvels + 0.5 * dt * delta2;

    // Update cell positions and orientations for the full timestep, and 
    // translational and orientational velocities for the half-timestep 
    for (int i = 0; i < n; ++i)
    {
        // Update positions and orientations 
        Matrix<T, 3, 1> ri = cells(i, __colseq_r).matrix(); 
        Matrix<T, 3, 1> ni = cells(i, __colseq_n).matrix();
        T half_li = cells(i, __colidx_half_l); 
        Matrix<T, 3, 1> pi = ri - half_li * ni; 
        Matrix<T, 3, 1> qi = ri + half_li * ni; 
        Matrix<T, 3, 1> vi = velocities_half.row(i).matrix();
        Matrix<T, 3, 1> wi = angvels_half.row(i).matrix();
        Matrix<T, 3, 1> cross = wi.cross(pi - qi) / 2;  
        Matrix<T, 3, 1> p_new = pi + dt * (vi + cross);
        Matrix<T, 3, 1> q_new = qi + dt * (vi - cross);
        Matrix<T, 3, 1> r_new = (p_new + q_new) / 2; 
        Matrix<T, 3, 1> n_new = (q_new - p_new) / cells(i, __colidx_l); 
        cells_new(i, __colseq_r) = r_new.array(); 
        cells_new(i, __colseq_n) = n_new.array();

        // Update translational and orientational velocities 
        cells_new(i, __colseq_dr) = vi.array(); 
        cells_new(i, __colseq_dn) = wi.cross(ni).array(); 
    }

    // Renormalize orientations 
    normalizeOrientations<T>(cells_new, dt, iter);

    // Update cell-cell neighbor distances
    Array<T, Dynamic, 7> neighbors_new(neighbors);  
    updateNeighborDistances<T>(cells_new, neighbors_new);

    // Update cell-surface overlap coordinates
    for (int i = 0; i < n; ++i)
    {
        assume_2d(i) = (cells_new(i, __colidx_nz) < nz_threshold);
        if (assume_2d(i))
            ss(i) = std::numeric_limits<T>::quiet_NaN();
        else
            ss(i) = sstar(cells_new(i, __colidx_rz), cells_new(i, __colidx_nz), R); 
    }

    // Compute new forces with the updated positions, orientations, and 
    // velocities
    forces = getConservativeForces<T>(
        cells_new, neighbors_new, dt, iter, R, Rcell, E0, repulsion_prefactors, 
        assume_2d, nz_threshold, adhesion_mode, adhesion_params, jkr_data,
        colidx_gamma, friction_mode, colidx_eta_cell_cell, no_surface, multithread,
        true, cell_cell_coulomb_coeff
    );
    forces += noise; 
    forces -= ambientViscosityForces<T>(cells_new, dt, iter, R);
    if (!no_surface)
    {
        Array<T, Dynamic, 6> cell_surface_friction = cellSurfaceFrictionForces<T>(
            cells_new, dt, iter, ss, R, assume_2d, multithread
        );

        // Truncate the cell-surface friction forces according to Coulomb's
        // law of friction, if desired 
        if (cell_surface_coulomb_coeff > 0)
        {
            Array<T, Dynamic, 6> cell_surface_repulsive = cellSurfaceRepulsionForces<T>(
                cells_new, dt, iter, ss, R, E0, assume_2d, multithread, true
            ); 
            for (int i = 0; i < n; ++i)
            {
                T friction_norm = cell_surface_friction.row(i).matrix().norm(); 
                T max_norm = cell_surface_coulomb_coeff * cell_surface_repulsive.row(i).matrix().norm(); 
                if (friction_norm > max_norm)
                    cell_surface_friction.row(i) *= (max_norm / friction_norm); 
            } 
        }
        forces -= cell_surface_friction;
    }

    // Decompose the torques into parallel and perpendicular contributions
    torques = forces(Eigen::all, Eigen::seq(3, 5)); 
    for (int i = 0; i < n; ++i)
    {
        Matrix<T, 3, 1> ni = cells_new(i, __colseq_n).matrix(); 
        torques_par.row(i) = torques.row(i).matrix().dot(ni) * ni;
        torques_perp.row(i) = torques.row(i) - torques_par.row(i); 
    }

    // Calculate the updated velocity and angular velocity of each cell, and 
    // update the full array 
    delta1 = forces(Eigen::all, Eigen::seq(0, 2)).colwise() / masses;
    delta2 = torques_par.colwise() / moments.col(0); 
    delta2 += torques_perp.colwise() / moments.col(1); 
    cells_new(Eigen::all, __colseq_dr) += 0.5 * dt * delta1; 
    Array<T, Dynamic, 3> angvels_full = angvels_half + 0.5 * dt * delta2;
    for (int i = 0; i < n; ++i)
    {
        Matrix<T, 3, 1> ni = cells_new(i, __colseq_n).matrix(); 
        cells_new(i, __colseq_dn) = angvels_full.row(i).matrix().cross(ni).array(); 
    }

    // Renormalize orientations 
    normalizeOrientations<T>(cells_new, dt, iter);
   
    return cells_new; 
}

#endif
