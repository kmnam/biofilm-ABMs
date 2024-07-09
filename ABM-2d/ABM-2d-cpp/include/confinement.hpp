/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/9/2024
 */

#ifndef BIOFILM_RADIAL_CONFINEMENT_HPP
#define BIOFILM_RADIAL_CONFINEMENT_HPP

#include <vector>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "boundaries.hpp"
#include "distances.hpp"

using namespace Eigen;

using std::abs;
using boost::multiprecision::abs;
using std::acos; 
using boost::multiprecision::acos; 

/**
 * Get the peripheral subset of cells in the biofilm from a simply connected
 * alpha-shape built from the cell centers.
 */
template <typename T>
std::vector<int> getBoundary(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    std::vector<double> x, y;
    for (int i = 0; i < cells.rows(); ++i)
    {
        x.push_back(static_cast<double>(cells(i, 0))); 
        y.push_back(static_cast<double>(cells(i, 1))); 
    }
    return Boundary2D(x, y).getSimplyConnectedBoundary().vertices;
}

/**
 * Get the radial confinement force.
 */
template <typename T>
Array<T, Dynamic, 4> radialConfinementForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             const Ref<const Array<T, Dynamic, 1> >& center,
                                             const T rest_radius, 
                                             const T spring_const)
{
    // Get the indices of the peripheral cells 
    std::vector<int> boundary = getBoundary(cells);

    // Maintain an array of generalized forces on each cell
    const int n = cells.rows(); 
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4); 

    // For each peripheral cell ... 
    for (const int j : boundary)
    {
        Matrix<T, 2, 1> rj = cells(j, Eigen::seq(0, 1)).transpose().matrix(); 
        Matrix<T, 2, 1> nj = cells(j, Eigen::seq(2, 3)).transpose().matrix();
        T half_lj = cells(j, 5);
        Matrix<T, 2, 1> pj = rj - half_lj * nj;
        Matrix<T, 2, 1> qj = rj + half_lj * nj;

        // Get the triangle formed by the center-to-endpoint vectors and 
        // the centerline
        T sj = 0;
        Matrix<T, 2, 1> d1 = pj - center;
        Matrix<T, 2, 1> d2 = qj - center;
        T norm_d1 = d1.norm(); 
        T norm_d2 = d2.norm();
        T theta1 = acos(d1.dot(nj) / norm_d1);
        T theta2 = acos(d2.dot(-nj) / norm_d2);
        T diff = abs(theta1 - theta2);

        // If the triangle is not isosceles, then set the contact point
        // between the cell and the membrane to be one of the endpoints 
        if (diff > 0.01 || norm_d1 > norm_d2)
            sj = -half_lj; 
        else if (diff > 0.01)
            sj = half_lj;

        // Get the radial displacement of the contact point from the center
        Matrix<T, 2, 1> dist = rj + sj * nj - center;
        T magnitude = dist.norm();
        Matrix<T, 2, 1> direction = dist / magnitude;
        T delta = magnitude - rest_radius;
        
        // If the radial displacement is greater than zero ...
        if (delta > 0)
        {
            // Determine the generalized forces acting on the peripheral cell
            T prefactor = spring_const * delta;
            dEdq(j, Eigen::seq(0, 1)) = prefactor * direction.transpose().array(); 
            dEdq(j, Eigen::seq(2, 3)) = prefactor * sj * direction.transpose().array();
        }
    }

    return dEdq;
}

#endif
