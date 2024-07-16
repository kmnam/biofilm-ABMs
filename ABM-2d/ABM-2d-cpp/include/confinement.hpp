/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/16/2024
 */

#ifndef BIOFILM_RADIAL_CONFINEMENT_HPP
#define BIOFILM_RADIAL_CONFINEMENT_HPP

#include <vector>
#include <utility>
#include <unordered_set>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "boundaries.hpp"
#include "distances.hpp"

using namespace Eigen;

using std::abs;
using boost::multiprecision::abs;
using std::acos; 
using boost::multiprecision::acos;
using std::sqrt;
using boost::multiprecision::sqrt;

/**
 * Get the maximum area covered by the given cells in the xy-plane. 
 *
 * This function obtains the area covered by the given cells without taking 
 * into account their positions and orientations (i.e., assuming that none
 * of the cells are overlapping).
 * 
 * @param cells Input population of cells. 
 * @param R     Cell radius.
 * @returns Maximum area covered by the given cells in the xy-plane. 
 */
template <typename T>
T getMaxArea(const Ref<const Array<T, Dynamic, Dynamic> >& cells, const T R)
{
    T caps_area = cells.rows() * boost::math::constants::pi<T>() * R * R;
    T cylinders_area = 2 * R * cells.col(4).sum(); 
    return caps_area + cylinders_area; 
}

/**
 * Get the peripheral subset of the given cells from a simply connected
 * alpha-shape built from the cell centerlines.
 *
 * It is assumed that there are 3 or more cells. 
 *
 * @param cells Input population of cells.
 * @returns An object containing the alpha-shape built from the cell
 *          centerlines. See `include/boundaries.hpp` for details. 
 */
template <typename T>
std::pair<AlphaShape2DProperties, std::vector<int> >
    getBoundaryFromCenterlines(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    std::vector<double> x, y;
    std::vector<int> idx; 
    for (int i = 0; i < cells.rows(); ++i)
    {
        T half_l = cells(i, 5);
        T quarter_l = half_l / 2; 
        x.push_back(static_cast<double>(cells(i, 0) - half_l * cells(i, 2)));
        y.push_back(static_cast<double>(cells(i, 1) - half_l * cells(i, 3)));
        x.push_back(static_cast<double>(cells(i, 0) - quarter_l * cells(i, 2)));
        y.push_back(static_cast<double>(cells(i, 1) - quarter_l * cells(i, 3)));
        x.push_back(static_cast<double>(cells(i, 0))); 
        y.push_back(static_cast<double>(cells(i, 1)));
        x.push_back(static_cast<double>(cells(i, 0) + quarter_l * cells(i, 2)));
        y.push_back(static_cast<double>(cells(i, 1) + quarter_l * cells(i, 3)));
        x.push_back(static_cast<double>(cells(i, 0) + half_l * cells(i, 2))); 
        y.push_back(static_cast<double>(cells(i, 1) + half_l * cells(i, 3)));
        for (int j = 0; j < 5; ++j)
            idx.push_back(i); 
    }
    return std::make_pair(Boundary2D(x, y).getSimplyConnectedBoundary(), idx); 
}

/**
 * Get the peripheral subset of the given cells from a simply connected
 * alpha-shape built from 2-D cross-sectional outlines of the cells.
 *
 * It is assumed that there are 3 or more cells. 
 *
 * @param cells            Input population of cells. 
 * @param R                Cell radius. 
 * @param outline_meshsize Approximate meshsize with which to obtain points
 *                         from each cell outline.
 * @returns An object containing the alpha-shape built from the cell outlines,
 *          together with a vector of indices that assigns each outline point
 *          to the cell from which it originates. 
 */
template <typename T>
std::pair<AlphaShape2DProperties, std::vector<int> >
    getBoundaryFromOutlines(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                            const T R, const T outline_meshsize)
{
    std::vector<double> x, y;
    std::vector<int> idx;

    // For each cell ... 
    for (int i = 0; i < cells.rows(); ++i)
    {
        Array<T, 2, 1> ri = cells(i, Eigen::seq(0, 1)).transpose();
        Array<T, 2, 1> ni = cells(i, Eigen::seq(2, 3)).transpose(); 
        T li = cells(i, 4);
        T half_li = cells(i, 5);
        Array<T, 2, 1> ni_rot; 
        ni_rot << R * (-ni(1)), R * ni(0); 

        // Generate a 2-D outline with approximately the given meshsize
        //
        // First generate the cylinder ...
        int m = static_cast<int>(li / outline_meshsize) + 1;
        Array<T, 2, 1> mesh1 = Array<T, 2, 1>::LinSpaced(m, -half_li, half_li);
        for (int j = 0; j < m; ++j)
        {
            Array<T, 2, 1> p = ri + mesh1(j) * ni;
            Array<T, 2, 1> q = p + ni_rot;
            Array<T, 2, 1> s = p - ni_rot;

            // For each point, keep track of the x- and y-coordinates and the 
            // cell from which it originates 
            x.push_back(static_cast<double>(p(0)));
            y.push_back(static_cast<double>(p(1))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(q(0))); 
            y.push_back(static_cast<double>(q(1))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(s(0))); 
            y.push_back(static_cast<double>(s(1))); 
            idx.push_back(i); 
        }

        // ... then generate the hemispherical caps
        m = static_cast<int>(boost::math::constants::pi<T>() * R / outline_meshsize) + 1;
        Array<T, 2, 1> mesh2 = Array<T, 2, 1>::LinSpaced(
            m, -boost::math::constants::half_pi<T>(), boost::math::constants::half_pi<T>()
        );
        Array<T, 2, 1> pi = ri - half_li * ni;
        Array<T, 2, 1> qi = ri + half_li * ni;
        for (int j = 0; j < m; ++j)
        {
            Matrix<T, 2, 2> rot; 
            T cos_theta = cos(mesh2(j)); 
            T sin_theta = sin(mesh2(j)); 
            rot << cos_theta, -sin_theta,
                   sin_theta,  cos_theta; 
            Matrix<T, 2, 1> v = pi.matrix() + R * rot * (-ni).matrix();
            Matrix<T, 2, 1> w = qi.matrix() + R * rot * ni.matrix();

            // For each point, keep track of the x- and y-coordinates and the 
            // cell from which it originates 
            x.push_back(static_cast<double>(v(0))); 
            y.push_back(static_cast<double>(v(1))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(w(0))); 
            y.push_back(static_cast<double>(w(1))); 
            idx.push_back(i);
        }
    }
    
    return std::make_pair(Boundary2D(x, y).getSimplyConnectedBoundary(), idx); 
}

/**
 * Get the peripheral subset of the given population of cells.
 *
 * This is done via a two-step process:
 * - First, a simply connected alpha-shape is built from the cell centerlines.
 * - If the area enclosed by the alpha-shape far exceeds the maximum area of
 *   the given set of cells in the xy-plane (i.e., exceeds a given factor > 1
 *   times the maximum area), then a second simply connected alpha-shape is
 *   built from the cell outlines in the xy-plane, and the cells that
 *   contribute to the new alpha-shape are chosen to lie in the periphery.
 *
 * @param cells       Input population of cells. 
 * @param R           Cell radius. 
 * @param area_factor Build the outline-based alpha-shape if the area enclosed
 *                    by the centerline-based alpha-shape exceeds this value
 *                    times the maximum area of the given cells. Should be
 *                    greater than 1.
 * @param outline_meshsize Approximate meshsize with which to obtain points 
 *                         from each cell outline, while building the outline-
 *                         based alpha-shape.
 * @param mincells_for_centerline_boundary Minimum number of cells required
 *                                         for computing a centerline-based
 *                                         alpha-shape.
 * @returns Vector of peripheral cell indices, together with the maximum area 
 *          of the given cells.  
 */
template <typename T>
std::vector<int> getBoundary(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                             const T R, const T area_factor, const T outline_meshsize,
                             const int mincells_for_centerline_boundary)
{
    // If there are fewer than 4 cells, then return all the cells 
    const int n = cells.rows();
    if (n < 4)
    {
        std::vector<int> idx;
        for (int i = 0; i < n; ++i)
            idx.push_back(i); 
        return idx;
    }

    // Get the maximum area of the cells in the xy-plane  
    const T max_area = getMaxArea<T>(cells, R);

    // If there are more than the minimum number of cells for calculating the
    // centerline-based alpha-shape, calculate the centerline-based alpha-shape
    bool success = false; 
    AlphaShape2DProperties shape1, shape2;
    std::vector<int> idx1, idx2;
    if (n >= mincells_for_centerline_boundary)
    {
        // Get the centerline-based alpha-shape
        auto result = getBoundaryFromCenterlines<T>(cells);
        shape1 = result.first; 
        idx1 = result.second; 

        // Does the area enclosed by the centerline-based alpha-shape far
        // exceed the maximum area of the cells? 
        success = (shape1.area < area_factor * max_area);
    }
    
    // Compute the outline-based alpha-shape if necessary
    if (!success)
    {
        // Get the outline-based alpha-shape 
        auto result = getBoundaryFromOutlines<T>(cells, R, outline_meshsize);
        shape2 = result.first; 
        idx2 = result.second; 

        // Get the cells corresponding to the points in the alpha-shape 
        std::unordered_set<int> cell_idx; 
        for (const int j : shape2.vertices)
            cell_idx.insert(idx2[j]);

        #ifdef DEBUG_PRINT_BOUNDARY_STATUS
            std::cout << "... Found peripheral cells using outline-based "
                      << "alpha-shape" << std::endl;
        #endif
        return std::vector<int>(cell_idx.begin(), cell_idx.end()); 
    }
    // Otherwise, return the vertices of the centerline-based alpha-shape
    else 
    {
        // Get the cells corresponding to the points in the alpha-shape 
        std::unordered_set<int> cell_idx; 
        for (const int j : shape1.vertices)
            cell_idx.insert(idx1[j]);

        #ifdef DEBUG_PRINT_BOUNDARY_STATUS
            std::cout << "... Found peripheral cells using centerline-based "
                      << "alpha-shape" << std::endl;
        #endif
        return std::vector<int>(cell_idx.begin(), cell_idx.end()); 
    }
}

/**
 * Compute radial confinement forces on the given array of cells.
 *
 * @param cells              Input population of cells.
 * @param boundary_idx       Pre-computed vector of indices of peripheral cells. 
 * @param R                  Cell radius. 
 * @param center             Fixed center for the elastic membrane. 
 * @param rest_radius_factor Multiply the effective radius obtained from the
 *                           maximum area of the given cells by this value to 
 *                           obtain the rest radius of the elastic membrane. 
 * @param spring_const       Effective spring constant for the elastic 
 *                           membrane.
 * @returns Array of generalized radial confinement forces. 
 */
template <typename T>
Array<T, Dynamic, 4> radialConfinementForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             std::vector<int>& boundary_idx, 
                                             const T R,
                                             const Ref<const Matrix<T, 2, 1> >& center,
                                             const T rest_radius_factor, 
                                             const T spring_const)
{
    // Get the maximum area of the cells in the xy-plane  
    const T max_area = getMaxArea<T>(cells, R);

    // Obtain a corresponding rest radius for the confining membrane 
    T rest_radius = rest_radius_factor * sqrt(max_area / boost::math::constants::pi<T>());

    // Maintain an array of generalized forces on each cell
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(cells.rows(), 4); 

    // For each cell ... 
    for (const int j : boundary_idx)
    {
        Matrix<T, 2, 1> rj = cells(j, Eigen::seq(0, 1)).transpose().matrix(); 
        Matrix<T, 2, 1> nj = cells(j, Eigen::seq(2, 3)).transpose().matrix();
        T half_lj = cells(j, 5);
        
        // Get the two endpoints of the centerline 
        Matrix<T, 2, 1> pj = rj - half_lj * nj;
        Matrix<T, 2, 1> qj = rj + half_lj * nj;

        // Get the triangle formed by the center-to-endpoint vectors and 
        // the centerline
        T sj = 0;
        Matrix<T, 2, 1> d1 = pj - center;
        Matrix<T, 2, 1> d2 = qj - center;
        T norm_d1 = d1.norm(); 
        T norm_d2 = d2.norm();
        T theta1 = acos((-d1).dot(nj) / norm_d1);
        T theta2 = acos(d2.dot(nj) / norm_d2);
        T diff = abs(theta1 - theta2);

        // If the triangle is not isosceles, then set the contact point
        // between the cell and the membrane to be one of the endpoints
        //
        // If the triangle is (close to) isosceles, then the contact point
        // is at the cell center 
        if (diff > 0.01 && norm_d1 > norm_d2)
            sj = -half_lj; 
        else if (diff > 0.01)    // norm_d1 <= norm_d2
            sj = half_lj;

        // Get the radial displacement of the contact point from the center
        Matrix<T, 2, 1> dist = rj + sj * nj - center;
        T magnitude = dist.norm();
        Matrix<T, 2, 1> direction = dist / magnitude;
        T delta = magnitude + R - rest_radius;
        
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
