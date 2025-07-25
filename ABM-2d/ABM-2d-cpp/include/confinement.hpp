/**
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
 *     4/29/2025
 */

#ifndef BIOFILM_CONFINEMENT_HPP
#define BIOFILM_CONFINEMENT_HPP

#include <vector>
#include <utility>
#include <unordered_set>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "indices.hpp"
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
    T cylinders_area = 2 * R * cells.col(__colidx_l).sum(); 
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
        T half_l = cells(i, __colidx_half_l);
        T quarter_l = half_l / 2; 
        x.push_back(static_cast<double>(cells(i, __colidx_rx) - half_l * cells(i, __colidx_nx)));
        y.push_back(static_cast<double>(cells(i, __colidx_ry) - half_l * cells(i, __colidx_ny)));
        x.push_back(static_cast<double>(cells(i, __colidx_rx) - quarter_l * cells(i, __colidx_nx)));
        y.push_back(static_cast<double>(cells(i, __colidx_ry) - quarter_l * cells(i, __colidx_ny)));
        x.push_back(static_cast<double>(cells(i, __colidx_rx))); 
        y.push_back(static_cast<double>(cells(i, __colidx_ry)));
        x.push_back(static_cast<double>(cells(i, __colidx_rx) + quarter_l * cells(i, __colidx_nx)));
        y.push_back(static_cast<double>(cells(i, __colidx_ry) + quarter_l * cells(i, __colidx_ny)));
        x.push_back(static_cast<double>(cells(i, __colidx_rx) + half_l * cells(i, __colidx_nx))); 
        y.push_back(static_cast<double>(cells(i, __colidx_ry) + half_l * cells(i, __colidx_ny)));
        for (int j = 0; j < 5; ++j)
            idx.push_back(i); 
    }
    return std::make_pair(Boundary2D(x, y).getSimplyConnectedBoundary(), idx); 
}

/**
 * Get the peripheral subset of the given population of cells.
 *
 * This is done as follows:
 * - If there are fewer than `mincells_for_boundary` cells, then return all 
 *   the cells. 
 * - Otherwise, build a simply connected alpha-shape from the cell centerlines
 *   and identify the cells that contribute to the alpha-shape.
 *
 * @param cells                 Input population of cells. 
 * @param R                     Cell radius. 
 * @param mincells_for_boundary Minimum number of cells required for computing
 *                              an alpha-shape.
 * @returns Vector of peripheral cell indices, together with the maximum area 
 *          of the given cells.  
 */
template <typename T>
std::vector<int> getBoundary(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                             const T R, const int mincells_for_boundary)
{
    // If there are fewer than the minimum number of cells, then return all
    // the cells 
    const int n = cells.rows();
    if (n < mincells_for_boundary)
    {
        std::vector<int> idx;
        for (int i = 0; i < n; ++i)
            idx.push_back(i);
        return idx;
    }

    // Calculate the centerline-based alpha-shape
    auto result = getBoundaryFromCenterlines<T>(cells);
    AlphaShape2DProperties shape = result.first; 
    std::vector<int> idx = result.second; 

    // Get the cells corresponding to the points in the alpha-shape 
    std::unordered_set<int> cell_idx; 
    for (const int j : shape.vertices)
        cell_idx.insert(idx[j]);

    #ifdef DEBUG_PRINT_BOUNDARY_STATUS
        std::cout << "... Found peripheral cells using centerline-based "
                  << "alpha-shape" << std::endl;
    #endif
    return std::vector<int>(cell_idx.begin(), cell_idx.end()); 
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
        Array<T, 2, 1> ri = cells(i, __colseq_r).transpose(); 
        Array<T, 2, 1> ni = cells(i, __colseq_n).transpose(); 
        T li = cells(i, __colidx_l);
        T half_li = cells(i, __colidx_half_l);
        Array<T, 2, 1> ni_rot; 
        ni_rot << R * (-ni(1)), R * ni(0); 

        // Generate a 2-D outline with approximately the given meshsize
        //
        // First generate the cylinder ...
        int m = static_cast<int>(li / outline_meshsize) + 1;
        Array<T, Dynamic, 1> mesh1 = Array<T, Dynamic, 1>::LinSpaced(m, -half_li, half_li);
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
        Array<T, Dynamic, 1> mesh2 = Array<T, Dynamic, 1>::LinSpaced(
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
 * Compute radial confinement forces on the given array of cells.
 *
 * @param cells        Input population of cells.
 * @param boundary_idx Pre-computed vector of indices of peripheral cells. 
 * @param R            Cell radius. 
 * @param center       Fixed center for the elastic membrane.
 * @param rest_radius  Rest radius of the elastic membrane. 
 * @param spring_const Effective spring constant for the elastic membrane.
 * @returns Array of generalized radial confinement forces. 
 */
template <typename T>
Array<T, Dynamic, 4> radialConfinementForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                             std::vector<int>& boundary_idx, 
                                             const T R,
                                             const Ref<const Matrix<T, 2, 1> >& center,
                                             const T rest_radius, 
                                             const T spring_const)
{
    // Maintain an array of generalized forces on each cell
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(cells.rows(), 4); 

    // For each cell ... 
    for (const int j : boundary_idx)
    {
        Matrix<T, 2, 1> rj = cells(j, __colseq_r).transpose().matrix(); 
        Matrix<T, 2, 1> nj = cells(j, __colseq_n).transpose().matrix();
        T half_lj = cells(j, __colidx_half_l);
        
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

/**
 * Compute channel confinement forces on the given array of cells. 
 *
 * @param cells                Input population of cells.
 * @param boundary_idx         Pre-computed vector of indices of peripheral cells. 
 * @param R                    Cell radius. 
 * @param short_section_y      y-position of the short section of the elastic 
 *                             membrane (parallel to the x-axis).
 * @param left_long_section_x  x-position of the left-hand long section of the
 *                             elastic membrane (parallel to the y-axis).
 * @param right_long_section_x x-position of the right-hand long section of the
 *                             elastic membrane (parallel to the y-axis).
 * @param spring_const         Effective spring constant for the elastic 
 *                             membrane.
 * @returns Array of generalized channel confinement forces. 
 */
template <typename T>
Array<T, Dynamic, 4> channelConfinementForces(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                              std::vector<int>& boundary_idx, 
                                              const T R,
                                              const T short_section_y,
                                              const T left_long_section_x,
                                              const T right_long_section_x,
                                              const T spring_const)
{
    // Maintain an array of generalized forces on each cell
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(cells.rows(), 4); 

    // For each cell ... 
    for (const int j : boundary_idx)
    {
        Matrix<T, 2, 1> rj = cells(j, __colseq_r).transpose().matrix(); 
        Matrix<T, 2, 1> nj = cells(j, __colseq_n).transpose().matrix();
        T half_lj = cells(j, __colidx_half_l);

        // Get the two endpoints of the centerline 
        Matrix<T, 2, 1> pj = rj - half_lj * nj;
        Matrix<T, 2, 1> qj = rj + half_lj * nj;

        // Identify the contact point with the short section of the membrane 
        //
        // If the cell is exactly parallel to the x-axis, then choose the
        // cell center as the contact point; otherwise, choose the endpoint
        // with the lesser y-coordinate
        Matrix<T, 2, 1> contact;
        T sj; 
        if (nj(1) != 0)           // Cell is not parallel to x-axis
        {
            if (pj(1) < qj(1))    // pj has lesser y-coordinate 
            {
                contact = pj;
                sj = -half_lj; 
            }
            else                  // qj has lesser y-coordinate
            {
                contact = qj;
                sj = half_lj;
            }
        }
        else                      // Cell is parallel to x-axis
        {
            contact = rj;
            sj = 0.0;
        }
        
        // Does the cell penetrate the short section of the membrane?
        if (contact(1) - R < short_section_y) 
        {
            // If so, determine the generalized forces acting on the cell 
            T delta = short_section_y - contact(1) + R;
            T prefactor = spring_const * delta;
            Array<T, 1, 2> direction;    // The force acts upward 
            direction << 0, -1;
            dEdq(j, Eigen::seq(0, 1)) = prefactor * direction;
            dEdq(j, Eigen::seq(2, 3)) = prefactor * sj * direction;
        }

        // Identify the contact point with the left-hand long section of the 
        // membrane 
        //
        // If the cell is exactly parallel to the y-axis, then choose the
        // cell center as the contact point; otherwise, choose the endpoint
        // with the lesser x-coordinate
        if (nj(0) != 0)           // Cell is not parallel to y-axis
        {
            if (pj(0) < qj(0))    // pj has lesser x-coordinate 
            {
                contact = pj;
                sj = -half_lj; 
            }
            else                  // qj has lesser x-coordinate 
            {
                contact = qj;
                sj = half_lj;
            }
        }
        else                      // Cell is parallel to y-axis
        {
            contact = rj;
            sj = 0.0;
        }

        // Does the cell penetrate this section of the membrane? 
        if (contact(0) - R < left_long_section_x) 
        {
            // If so, determine the generalized forces acting on the cell 
            T delta = left_long_section_x - contact(0) + R; 
            T prefactor = spring_const * delta;
            Array<T, 1, 2> direction;    // The force acts to the right  
            direction << -1, 0;
            dEdq(j, Eigen::seq(0, 1)) += prefactor * direction;
            dEdq(j, Eigen::seq(2, 3)) += prefactor * sj * direction;
        }

        // Identify the contact point with the right-hand long section of the 
        // membrane 
        //
        // If the cell is exactly parallel to the y-axis, then choose the
        // cell center as the contact point; otherwise, choose the endpoint
        // with the greater x-coordinate
        if (nj(0) != 0)           // Cell is not parallel to y-axis
        {
            if (pj(0) > qj(0))    // pj has greater x-coordinate 
            {
                contact = pj;
                sj = -half_lj; 
            }
            else                  // qj has greater x-coordinate 
            {
                contact = qj;
                sj = half_lj;
            }
        }
        else                      // Cell is parallel to y-axis
        {
            contact = rj;
            sj = 0.0;
        }

        // Does the cell penetrate this section of the membrane?
        if (contact(0) + R > right_long_section_x)
        {
            // If so, determine the generalized forces acting on the cell 
            T delta = contact(0) + R - right_long_section_x; 
            T prefactor = spring_const * delta;
            Array<T, 1, 2> direction;    // The force acts to the left 
            direction << 1, 0;
            dEdq(j, Eigen::seq(0, 1)) += prefactor * direction; 
            dEdq(j, Eigen::seq(2, 3)) += prefactor * sj * direction;  
        }
    }

    return dEdq;
}

#endif
