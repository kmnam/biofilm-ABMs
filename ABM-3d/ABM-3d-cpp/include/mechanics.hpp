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
 *     1/22/2024
 */

#ifndef BIOFILM_MECHANICS_3D_HPP
#define BIOFILM_MECHANICS_3D_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "integrals.hpp"

using namespace Eigen;

/** 
 * Return the cell-body coordinate along a cell centerline that is nearest to
 * a given point.
 *
 * @param r Cell center.
 * @param n Cell orientation.
 * @param half_l Half of cell length.  
 * @param q Input point.
 * @returns Distance from cell to input point.
 */
template <typename T>
T nearestCellBodyCoordToPoint(const Ref<const Matrix<T, 3, 1> >& r, 
                              const Ref<const Matrix<T, 3, 1> >& n,
                              const T half_l,
                              const Ref<const Matrix<T, 3, 1> >& q)
{
    T s = (q - r).dot(n);
    if (std::abs(s) <= half_l)
        return s;
    else if (s > half_l)
        return half_l;
    else    // s < -half_l
        return -half_l;
}

/**
 * Return the shortest distance between the centerlines of two cells, along
 * with the cell-body coordinates at which the shortest distance is achieved.
 *
 * The distance vector returned by this function runs from cell 1 to cell 2.
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2.
 * @returns Shortest distance between the two cells, along with the cell-body
 *          coordinates at which the shortest distance is achieved. The
 *          distance is returned as a vector running from cell 1 to cell 2.
 */
template <typename T>
std::tuple<Matrix<T, 3, 1>, T, T> distBetweenCells(const Ref<const Matrix<T, 3, 1> >& r1,
                                                   const Ref<const Matrix<T, 3, 1> >& n1, 
                                                   const T half_l1,
                                                   const Ref<const Matrix<T, 3, 1> >& r2,
                                                   const Ref<const Matrix<T, 3, 1> >& n2,
                                                   const T half_l2)
{
    // Vector running from r1 to r2
    Matrix<T, 3, 1> r12 = r2 - r1;

    // We are looking for the values of s in [-l1/2, l1/2] and 
    // t in [-l2/2, l2/2] such that the norm of r12 + t*n2 - s*n1
    // is minimized
    T r12_dot_n1 = r12.dot(n1);
    T r12_dot_n2 = r12.dot(n2);
    T n1_dot_n2 = n1.dot(n2);
    T s_numer = r12_dot_n1 - n1_dot_n2 * r12_dot_n2;
    T t_numer = n1_dot_n2 * r12_dot_n1 - r12_dot_n2;
    T denom = 1 - n1_dot_n2 * n1_dot_n2;
    Matrix<T, 3, 1> dist = Matrix<T, 3, 1>::Zero(2);
    T s = 0;
    T t = 0;  

    // If the two centerlines are not parallel ...
    if (std::abs(denom) > 1e-6)
    {
        s = s_numer / denom;
        t = t_numer / denom; 
        // Check that the unconstrained minimum values of s and t lie within
        // the desired ranges
        if (std::abs(s) > half_l1 || std::abs(t) > half_l2)
        {
            // If not, find the side of the square [-l1/2, l1/2] by
            // [-l2/2, l2/2] in which the unconstrained minimum values
            // is nearest
            //
            // Region 1 (above top side):
            //     between t = -s - X and t = s - X
            // Region 2 (right of right side):
            //     between t = s - X and t = -s + X
            // Region 3 (below bottom side): 
            //     between t = -s + X and t = s + X
            // Region 4 (left of left side):
            //     between t = s + X and t = -s - X
            // where X = (l1 - l2) / 2
            T X = half_l1 - half_l2;
            T Y = s + X;
            T Z = s - X;
            if (t >= -Y && t >= Z)          // In region 1
            {
                // In this case, set t = l2 / 2 and find s
                Matrix<T, 3, 1> q = r2 + half_l2 * n2; 
                s = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q);
                t = half_l2;
            }
            else if (t < Z && t >= -Z)      // In region 2
            {
                // In this case, set s = l1 / 2 and find t
                Matrix<T, 3, 1> q = r1 + half_l1 * n1;
                t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q); 
                s = half_l1;
            }
            else if (t < -Z && t < Y)      // In region 3
            {
                // In this case, set t = -l2 / 2 and find s
                Matrix<T, 3, 1> q = r2 - half_l2 * n2;
                s = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q);
                t = -half_l2;
            }
            else    // t >= s + X and t < -s - X, in region 4
            {
                // In this case, set s = -l1 / 2 and find t
                Matrix<T, 3, 1> q = r1 - half_l1 * n1;
                t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q); 
                s = -half_l1; 
            }
        }
        // Compute distance vector between the two cells
        dist = r12 + t * n2 - s * n1; 
    }
    // Otherwise ... 
    else
    {
        // First, compute the shortest distance vectors from the cap
        // centers of cell 1 to cell 2
        Matrix<T, 3, 1> p1 = r1 - half_l1 * n1;                 // Endpoint of cell 1 for s = -l1 / 2
        Matrix<T, 3, 1> q1 = r1 + half_l1 * n1;                 // Endpoint of cell 1 for s = l1 / 2
        T t_p1 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, p1);   // Distance from cell 2 to p1
        T t_q1 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q1);   // Distance from cell 2 to q1
        Matrix<T, 3, 1> dist_from_p1 = (r2 + t_p1 * n2) - p1;   // Vector running towards cell 2 from p1
        Matrix<T, 3, 1> dist_from_q1 = (r2 + t_q1 * n2) - q1;   // Vector running towards cell 2 from q1
        T dp1_dot_n1 = std::abs(dist_from_p1.dot(n1));    // Dot product of vector from p1 to cell 2 with n1
        T dq1_dot_n1 = std::abs(dist_from_q1.dot(n1));    // Dot product of vector from q1 to cell 2 with n1
        T dp1_costheta = dp1_dot_n1 / dist_from_p1.norm();      // Cosine of corresponding angle
        T dq1_costheta = dq1_dot_n1 / dist_from_q1.norm();      // Cosine of corresponding angle

        // If both distance vectors are orthogonal to the orientation
        // of cell 1, then choose the distance vector from r1 to r2
        if (dp1_costheta < 1e-3 && dq1_costheta < 1e-3)
        {
            dist = r12;
            s = 0;
            t = 0;
        }
        // If both cell-body coordinates are equal, then they should 
        // both be equal to -l2 / 2 or both be equal to l2 / 2, in which
        // case we can choose whichever endpoint of cell 1 is closest
        else if (t_p1 == t_q1)
        {
            if (dist_from_p1.squaredNorm() < dist_from_q1.squaredNorm())
            {
                dist = dist_from_p1;
                s = -half_l1;
                t = t_p1;
            }
            else
            {
                dist = dist_from_q1;
                s = half_l1;
                t = t_q1;
            }
        }
        // Otherwise, the two cell-body coordinates should be different
        //
        // In this case, additionally calculate the shortest distance
        // vectors from the cap centers of cell 2 to cell 1 
        else
        {
            Matrix<T, 3, 1> p2 = r2 - half_l2 * n2;                 // Endpoint of cell 2 for t = -l2 / 2
            Matrix<T, 3, 1> q2 = r2 + half_l2 * n2;                 // Endpoint of cell 2 for t = l2 / 2
            T s_p2 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, p2);   // Distance from cell 1 to p2
            T s_q2 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q2);   // Distance from cell 1 to q2
            Matrix<T, 3, 1> dist_from_p2 = (r1 + s_p2 * n1) - p2;   // Vector from p2 running towards cell 1
                                                                    // (no need for vector from q2)
            if (dp1_costheta < 1e-3)
                dist = dist_from_p1;
            else if (dq1_costheta < 1e-3)
                dist = dist_from_q1;
            else     // If neither vectors are orthogonal to n1, then both 
                     // shortest distance vectors from cell 2 to cell 1 
                     // should be orthogonal to n2
                dist = -dist_from_p2;
            s = (s_p2 + s_q2) / 2;
            t = (t_p1 + t_q1) / 2;
        }
    }
    
    return std::make_tuple(dist, s, t); 
}

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
    const T prefactor1 = (8 / 3) * E0 * std::pow(R, 0.5); 
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
            dEdq(i, 1) -= prefactor0 * cells(i, 5) * int3;
            dEdq(i, 1) -= prefactor0 * (1 - nz2) * int4; 
            dEdq(i, 1) += prefactor1 * cells(i, 5) * int5;
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
            dEdq(i, 0) = prefactor0 * (1 - nz2) * int1;

            // Compute the derivative of the cell-surface adhesion energy
            // with respect to z-orientation
            T int2 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            T int3 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), -0.5, ss(i));
            T int4 = integral4(cells(i, 2), cells(i, 5), R, cells(i, 7), ss(i));
            dEdq(i, 1) += prefactor2 * cells(i, 5) * int2;
            dEdq(i, 1) += prefactor0 * (1 - nz2) * int3;
            dEdq(i, 1) -= prefactor1 * cells(i, 5) * int4;
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
                auto result = distBetweenCells<T>(
                    cells(i, Eigen::seq(0, 2)).matrix(),
                    cells(i, Eigen::seq(3, 5)).matrix(), cells(i, 7),
                    cells(j, Eigen::seq(0, 2)).matrix(),
                    cells(j, Eigen::seq(3, 5)).matrix(), cells(j, 7)
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
            cells(i, Eigen::seq(0, 2)).matrix(),
            cells(i, Eigen::seq(3, 5)).matrix(), cells(i, 7),
            cells(j, Eigen::seq(0, 2)).matrix(),
            cells(j, Eigen::seq(3, 5)).matrix(), cells(j, 7)
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
        else if (overlap >= R - Rcell)
        {
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
 * @param noise Pre-determined noise values to add to each generalized force.
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
                                                const Ref<const Array<T, 6, 1> >& noise)
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
    /*
    if (cells.rows() >= 3)    // TODO
    {
        std::cout << dEdq_cell << std::endl;
        for (int i = 0; i < n; ++i)
        {
            std::cout << cells(i, 2) << " " << cells(i, 5) << " " << R << " "
                      << cells(i, 7) << " " << ss(i) << std::endl;
            T int1 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.0, ss(i));
            T int2 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            T int3 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 2.0, ss(i));
            T int4 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.0, ss(i));
            T int5 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 1.5, ss(i));
            T int6 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            std::cout << "rep " << int1 << " " << int2 << " " << int3 << " "
                      << int4 << " " << int5 << " " << int6 << std::endl;
        }
        std::cout << dEdq_surface_repulsion << std::endl;
        for (int i = 0; i < n; ++i)
        {
            std::cout << cells(i, 2) << " " << cells(i, 5) << " " << R << " "
                      << cells(i, 7) << " " << ss(i) << std::endl;
            T int1 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), -0.5, ss(i));
            T int2 = integral1(cells(i, 2), cells(i, 5), R, cells(i, 7), 0.5, ss(i));
            T int3 = integral2(cells(i, 2), cells(i, 5), R, cells(i, 7), -0.5, ss(i));
            T int4 = integral4(cells(i, 2), cells(i, 5), R, cells(i, 7), ss(i));
            std::cout << "adh " << int1 << " " << int2 << " " << int3 << " " << int4 << std::endl;
        }
        std::cout << dEdq_surface_adhesion << std::endl;
    }
    */

    // For each cell ... 
    for (int i = 0; i < n; ++i)
    {
        Array<T, 7, 7> A = Array<T, 7, 7>::Zero(7, 7);
        Array<T, 7, 1> b = Array<T, 7, 1>::Zero(7);

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
        b(Eigen::seq(0, 5)) += noise;

        // Solve the corresponding linear system
        Array<T, 7, 1> x = A.matrix().colPivHouseholderQr().solve(b.matrix()).array();
        velocities.row(i) = x.head(6);
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
                                        std::function<T(boost::random::mt19937&)>& noise_dist)
{
    // Determine noise to add to each generalized force at each timestep
    Array<T, 6, 1> noise;
    if (cells.rows() == 1
        noise = Array<T, 6, 1>::Zero();
    else
        noise = noise_dist(rng) * Array<T, 6, 1>::Ones();

    // Compute velocities at given partial timesteps 
    int n = cells.rows(); 
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 6> > velocities; 
    velocities.push_back(
        getVelocitiesFromNeighbors<T>(
            cells, neighbors, R, Rcell, cell_cell_prefactors, E0,
            adhesion_energy_density, nz_threshold, noise
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
                adhesion_energy_density, nz_threshold, noise
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
