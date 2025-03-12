/**
 * Distance calculations in three dimensions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/5/2025
 */

#ifndef DISTANCES_3D_HPP
#define DISTANCES_3D_HPP

#include <stdexcept>
#include <utility>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/squared_distance_3.h>
#include "indices.hpp"

using namespace Eigen; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;

/**
 * Generate a Segment_3 instance for the given cell. 
 *
 * @param r Cell center.
 * @param n Cell orientation. Assumed to be normalized. 
 * @param half_l Cell half-length.
 * @returns Segment_3 instance for the given cell. 
 */
template <typename T>
Segment_3 generateSegment(const Ref<const Matrix<T, 3, 1> >& r, 
                          const Ref<const Matrix<T, 3, 1> >& n, const T half_l)
{
    // Define the Segment_3 instance from the cell's two endpoints 
    Point_3 p(r(0) - half_l * n(0), r(1) - half_l * n(1), r(2) - half_l * n(2)); 
    Point_3 q(r(0) + half_l * n(0), r(1) + half_l * n(1), r(2) + half_l * n(2));
    return Segment_3(p, q); 
}

/**
 * Generate a vector of Segment_3 instances for the given population of cells.
 *
 * Note that Segment_3 is not mutable, and therefore a new vector must be 
 * generated every time the population is updated in some way. 
 *
 * @param cells Existing population of cells.
 * @returns Vector of Segment_3 instances for each cell. 
 */
template <typename T>
std::vector<Segment_3> generateSegments(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    std::vector<Segment_3> segments;
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Define the Segment_3 instance from the cell's two endpoints 
        Point_3 p(
            cells(i, __colidx_rx) - cells(i, __colidx_half_l) * cells(i, __colidx_nx),
            cells(i, __colidx_ry) - cells(i, __colidx_half_l) * cells(i, __colidx_ny),
            cells(i, __colidx_rz) - cells(i, __colidx_half_l) * cells(i, __colidx_nz)
        );
        Point_3 q(
            cells(i, __colidx_rx) + cells(i, __colidx_half_l) * cells(i, __colidx_nx),
            cells(i, __colidx_ry) + cells(i, __colidx_half_l) * cells(i, __colidx_ny),
            cells(i, __colidx_rz) + cells(i, __colidx_half_l) * cells(i, __colidx_nz)
        );
        segments.push_back(Segment_3(p, q));
    }

    return segments;
}

/** 
 * Return the cell-body coordinate along a cell centerline that is nearest to
 * a given point.
 *
 * @param r Cell center.
 * @param n Cell orientation. Assumed to be normalized. 
 * @param half_l Cell half-length.
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
 * Output an error message pertaining to the given cell-cell configuration. 
 *
 * @param id1 ID of cell 1.
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1. Assumed to be normalized. 
 * @param half_l1 Half-length of cell 1.
 * @param id2 ID of cell 2. 
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2. Assumed to be normalized. 
 * @param half_l2 Half-length of cell 2.
 */
template <typename T>
void pairConfigSummary(const int id1, const Ref<const Matrix<T, 3, 1> >& r1,
                       const Ref<const Matrix<T, 3, 1> >& n1, const T half_l1,
                       const int id2, const Ref<const Matrix<T, 3, 1> >& r2,
                       const Ref<const Matrix<T, 3, 1> >& n2, const T half_l2)
{
    std::cerr << "Cell 1 ID = " << id1 << std::endl
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ", "
                                     << r1(2) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ", "
                                          << n1(2) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 2 ID = " << id2 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ", "
                                     << r2(2) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ", "
                                          << n2(2) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl;
}

/**
 * Output an error message pertaining to the given cell-cell configuration.
 *
 * This error message contains the velocities of the cells as well.  
 *
 * @param id1 ID of cell 1.
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1. Assumed to be normalized. 
 * @param half_l1 Half-length of cell 1.
 * @param dr1 Translational velocity of cell 1. 
 * @param id2 ID of cell 2. 
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2. Assumed to be normalized.
 * @param half_l2 Half-length of cell 2.
 * @param dr2 Translational velocity of cell 2. 
 */
template <typename T>
void pairConfigSummaryWithVelocities(const int id1,
                                     const Ref<const Matrix<T, 3, 1> >& r1,
                                     const Ref<const Matrix<T, 3, 1> >& n1,
                                     const T half_l1,
                                     const Ref<const Matrix<T, 3, 1> >& dr1,
                                     const int id2,
                                     const Ref<const Matrix<T, 3, 1> >& r2,
                                     const Ref<const Matrix<T, 3, 1> >& n2,
                                     const T half_l2,
                                     const Ref<const Matrix<T, 3, 1> >& dr2)
{
    std::cerr << "Cell 1 ID = " << id1 << std::endl
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ", "
                                     << r1(2) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ", "
                                          << n1(2) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 1 velocity = (" << dr1(0) << ", " << dr1(1) << ", "
                                       << dr1(2) << ")" << std::endl
              << "Cell 2 ID = " << id2 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ", "
                                     << r2(2) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ", "
                                          << n2(2) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl
              << "Cell 2 velocity = (" << dr2(0) << ", " << dr2(1) << ", "
                                       << dr2(2) << ")" << std::endl;
}

/**
 * Return the shortest distance between the centerlines of two cells, along
 * with the cell-body coordinates at which the shortest distance is achieved.
 *
 * The distance vector returned by this function runs from cell 1 to cell 2.
 *
 * @param cell1 Segment_3 instance for cell 1.
 * @param cell2 Segment_3 instance for cell 2.
 * @param id1 ID of cell 1. Only used for debugging output. 
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1. Assumed to be normalized. 
 * @param half_l1 Half of length of cell 1.
 * @param id2 ID of cell 2. Only used for debugging output. 
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2. Assumed to be normalized.
 * @param half_l2 Half of length of cell 2.
 * @param kernel CGAL kernel instance to be passed to CGAL::...::squared_distance().
 * @returns Shortest distance between the two cells, along with the cell-body
 *          coordinates at which the shortest distance is achieved. The
 *          distance is returned as a vector running from cell 1 to cell 2.
 */
template <typename T>
std::tuple<Matrix<T, 3, 1>, T, T> distBetweenCells(const Segment_3& cell1,
                                                   const Segment_3& cell2,
                                                   const int id1, 
                                                   const Ref<const Matrix<T, 3, 1> >& r1,
                                                   const Ref<const Matrix<T, 3, 1> >& n1,
                                                   T half_l1,
                                                   const int id2, 
                                                   const Ref<const Matrix<T, 3, 1> >& r2,
                                                   const Ref<const Matrix<T, 3, 1> >& n2,
                                                   T half_l2,
                                                   const K& kernel)
{
    Matrix<T, 3, 1> d = Matrix<T, 3, 1>::Zero(); 
    T s = 0;
    T t = 0;

    // Are the two cells (nearly) parallel?
    //
    // We say that two cells are nearly parallel if they are at an angle of
    // theta <= 0.01 radians, which translates to cos(theta) >= 0.9999
    T cos_theta = n1.dot(n2);
    if (cos_theta >= 0.9999 || cos_theta <= -0.9999)
    {
        // Identify the four endpoint vectors 
        Matrix<T, 3, 1> p1 = r1 - half_l1 * n1; 
        Matrix<T, 3, 1> q1 = r1 + half_l1 * n1; 
        Matrix<T, 3, 1> p2 = r2 - half_l2 * n2;  
        Matrix<T, 3, 1> q2 = r2 + half_l2 * n2; 

        // Get the distance vectors between the endpoints of cell 1 and the
        // body of cell 2  
        T s_p1_to_cell2 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, p1); 
        T s_q1_to_cell2 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q1);
        Matrix<T, 3, 1> d_p1_to_cell2 = r2 + s_p1_to_cell2 * n2 - p1; 
        Matrix<T, 3, 1> d_q1_to_cell2 = r2 + s_q1_to_cell2 * n2 - q1;
        T dist_p1_to_cell2 = d_p1_to_cell2.norm(); 
        T dist_q1_to_cell2 = d_q1_to_cell2.norm(); 

        // If the two distance vectors point to the same point along cell 2
        // (which should be an endpoint), return the shorter of the two
        if (s_p1_to_cell2 == s_q1_to_cell2)
        {
            if (dist_p1_to_cell2 < dist_q1_to_cell2)
            {
                s = -half_l1; 
                t = s_p1_to_cell2; 
                d = d_p1_to_cell2;
            }
            else    // dist_p1_to_cell2 >= dist_q1_to_cell2
            {
                s = half_l1; 
                t = s_q1_to_cell2; 
                d = d_q1_to_cell2; 
            }
        }
        // Otherwise, get the distance vectors between the endpoints of cell 2
        // and the body of cell 1
        else 
        {
            T s_p2_to_cell1 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, p2); 
            T s_q2_to_cell1 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q2); 
            Matrix<T, 3, 1> d_p2_to_cell1 = r1 + s_p2_to_cell1 * n1 - p2; 
            Matrix<T, 3, 1> d_q2_to_cell1 = r1 + s_q2_to_cell1 * n1 - q2;
            T dist_p2_to_cell1 = d_p2_to_cell1.norm(); 
            T dist_q2_to_cell1 = d_q2_to_cell1.norm(); 

            // Get the two shortest distance vectors among the four
            std::vector<std::pair<T, int> > dists {
                std::make_pair(dist_p1_to_cell2, 0),
                std::make_pair(dist_q1_to_cell2, 1),
                std::make_pair(dist_p2_to_cell1, 2),
                std::make_pair(dist_q2_to_cell1, 3)
            }; 
            std::sort(
                dists.begin(), dists.end(),
                [](std::pair<T, int>& left, std::pair<T, int>& right)
                {
                    return left.first < right.first; 
                }
            );
            std::unordered_set<int> min_idx { dists[0].second, dists[1].second };
            if (min_idx.find(0) != min_idx.end())
            {
                if (min_idx.find(1) != min_idx.end())
                {
                    // Average between d_p1_to_cell2 and d_q1_to_cell2
                    s = 0.0; 
                    t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, r1);
                    d = r2 + t * n2 - r1;
                }
                else if (min_idx.find(2) != min_idx.end())
                {
                    // Average between d_p1_to_cell2 and d_p2_to_cell1
                    s = (-half_l1 + s_p2_to_cell1) / 2; 
                    t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, r1 + s * n1);
                    d = r2 + t * n2 - r1 - s * n1;
                }
                else    // min_idx.find(3) != min_idx.end()
                {
                    // Average between d_p1_to_cell2 and d_q2_to_cell1
                    s = (-half_l1 + s_q2_to_cell1) / 2; 
                    t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, r1 + s * n1);
                    d = r2 + t * n2 - r1 - s * n1;
                }
            }
            else if (min_idx.find(1) != min_idx.end())
            {
                if (min_idx.find(2) != min_idx.end())
                {
                    // Average between d_q1_to_cell2 and d_p2_to_cell1
                    s = (half_l1 + s_p2_to_cell1) / 2; 
                    t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, r1 + s * n1);
                    d = r2 + t * n2 - r1 - s * n1;
                }
                else    // min_idx.find(3) != min_idx.end()
                {
                    // Average between d_q1_to_cell2 and d_q2_to_cell1
                    s = (half_l1 + s_q2_to_cell1) / 2; 
                    t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, r1 + s * n1);
                    d = r2 + t * n2 - r1 - s * n1;
                }
            }
            else    // min_idx.find(2) != min_idx.end() && min_idx.find(3) != min_idx.end()
            {
                // Average between d_p2_to_cell1 and d_q2_to_cell1
                t = 0.0;
                s = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, r2); 
                d = r2 - r1 - s * n1;
            }
        }
    }
    else 
    {
        // Otherwise, compute the distance vector 
        auto result = CGAL::Distance_3::internal::squared_distance(cell1, cell2, kernel);
        s = static_cast<T>(CGAL::to_double(result.x)) * 2 * half_l1 - half_l1;
        t = static_cast<T>(CGAL::to_double(result.y)) * 2 * half_l2 - half_l2;
        d = r2 + t * n2 - r1 - s * n1;
    }
    
    #ifdef DEBUG_CHECK_DISTANCE_NAN
        if (d.array().isNaN().any())
        {
            std::cerr << "Found nan in distance vector:" << std::endl;
            pairConfigSummary<T>(id1, r1, n1, half_l1, id2, r2, n2, half_l2); 
            throw std::runtime_error("Found nan in distance vector");
        }
    #endif
    
    return std::make_tuple(d, s, t); 
}

#endif
