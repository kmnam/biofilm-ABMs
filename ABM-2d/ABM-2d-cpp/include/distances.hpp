/**
 * Distance calculations in two dimensions.
 *
 * Note that the CGAL functions used here pertain to points and objects
 * in 3-D, for which there is the correct functionality.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/31/2025
 */

#ifndef DISTANCES_2D_HPP
#define DISTANCES_2D_HPP

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
Segment_3 generateSegment(const Ref<const Matrix<T, 2, 1> >& r, 
                          const Ref<const Matrix<T, 2, 1> >& n, const T half_l)
{
    // Define the Segment_3 instance from the cell's two endpoints 
    Point_3 p(r(0) - half_l * n(0), r(1) - half_l * n(1), 0.0); 
    Point_3 q(r(0) + half_l * n(0), r(1) + half_l * n(1), 0.0);
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
            0.0
        );
        Point_3 q(
            cells(i, __colidx_rx) + cells(i, __colidx_half_l) * cells(i, __colidx_nx),
            cells(i, __colidx_ry) + cells(i, __colidx_half_l) * cells(i, __colidx_ny),
            0.0
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
T nearestCellBodyCoordToPoint(const Ref<const Matrix<T, 2, 1> >& r, 
                              const Ref<const Matrix<T, 2, 1> >& n,
                              const T half_l,
                              const Ref<const Matrix<T, 2, 1> >& q)
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
void pairConfigSummary(const int id1, const Ref<const Matrix<T, 2, 1> >& r1,
                       const Ref<const Matrix<T, 2, 1> >& n1, const T half_l1,
                       const int id2, const Ref<const Matrix<T, 2, 1> >& r2,
                       const Ref<const Matrix<T, 2, 1> >& n2, const T half_l2)
{
    std::cerr << "Cell 1 ID = " << id1 << std::endl
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 2 ID = " << id2 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ")" << std::endl
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
                                     const Ref<const Matrix<T, 2, 1> >& r1,
                                     const Ref<const Matrix<T, 2, 1> >& n1,
                                     const T half_l1,
                                     const Ref<const Matrix<T, 2, 1> >& dr1,
                                     const int id2,
                                     const Ref<const Matrix<T, 2, 1> >& r2,
                                     const Ref<const Matrix<T, 2, 1> >& n2,
                                     const T half_l2,
                                     const Ref<const Matrix<T, 2, 1> >& dr2)
{
    std::cerr << "Cell 1 ID = " << id1 << std::endl
              << "Cell 1 center = (" << r1(0) << ", " << r1(1) << ")" << std::endl
              << "Cell 1 orientation = (" << n1(0) << ", " << n1(1) << ")" << std::endl
              << "Cell 1 half-length = " << half_l1 << std::endl
              << "Cell 1 velocity = (" << dr1(0) << ", " << dr1(1) << ")" << std::endl
              << "Cell 2 ID = " << id2 << std::endl
              << "Cell 2 center = (" << r2(0) << ", " << r2(1) << ")" << std::endl
              << "Cell 2 orientation = (" << n2(0) << ", " << n2(1) << ")" << std::endl
              << "Cell 2 half-length = " << half_l2 << std::endl
              << "Cell 2 velocity = (" << dr2(0) << ", " << dr2(1) << ")" << std::endl;
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
std::tuple<Matrix<T, 2, 1>, T, T> distBetweenCells(const Segment_3& cell1,
                                                   const Segment_3& cell2,
                                                   const int id1, 
                                                   const Ref<const Matrix<T, 2, 1> >& r1,
                                                   const Ref<const Matrix<T, 2, 1> >& n1,
                                                   T half_l1,
                                                   const int id2, 
                                                   const Ref<const Matrix<T, 2, 1> >& r2,
                                                   const Ref<const Matrix<T, 2, 1> >& n2,
                                                   T half_l2,
                                                   const K& kernel)
{
    Matrix<T, 2, 1> d = Matrix<T, 2, 1>::Zero(); 
    T s = 0;
    T t = 0;

    // Are the two cells (nearly) parallel?
    //
    // We say that two cells are nearly parallel if they have are at an angle
    // of theta <= 0.01 radians, which translates to cos(theta) >= 0.9999
    T cos_theta = n1.dot(n2);
    if (cos_theta >= 0.9999 || cos_theta <= -0.9999)
    {
        // Identify the four endpoint vectors 
        Matrix<T, 2, 1> p1 = r1 - half_l1 * n1; 
        Matrix<T, 2, 1> q1 = r1 + half_l1 * n1; 
        Matrix<T, 2, 1> p2 = r2 - half_l2 * n2;  
        Matrix<T, 2, 1> q2 = r2 + half_l2 * n2; 

        // Get the distance vectors between the endpoints of cell 1 and the
        // body of cell 2  
        T s_p1_to_cell2 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, p1); 
        T s_q1_to_cell2 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q1);
        Matrix<T, 2, 1> d_p1_to_cell2 = r2 + s_p1_to_cell2 * n2 - p1; 
        Matrix<T, 2, 1> d_q1_to_cell2 = r2 + s_q1_to_cell2 * n2 - q1;
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
            Matrix<T, 2, 1> d_p2_to_cell1 = r1 + s_p2_to_cell1 * n1 - p2; 
            Matrix<T, 2, 1> d_q2_to_cell1 = r1 + s_q2_to_cell1 * n1 - q2;
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
        d = (r2 + t * n2 - r1 - s * n1)(Eigen::seq(0, 1));
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

// ----------------------------------------------------------------------- //
//                   CUSTOM DISTANCE FUNCTION (DEPRECATED)                 //
// ----------------------------------------------------------------------- //
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
std::tuple<Matrix<T, 2, 1>, T, T> distBetweenCells(const Ref<const Matrix<T, 2, 1> >& r1,
                                                   const Ref<const Matrix<T, 2, 1> >& n1, 
                                                   const T half_l1,
                                                   const Ref<const Matrix<T, 2, 1> >& r2,
                                                   const Ref<const Matrix<T, 2, 1> >& n2,
                                                   const T half_l2)
{
    // Vector running from r1 to r2
    Matrix<T, 2, 1> r12 = r2 - r1;

    // We are looking for the values of s in [-l1/2, l1/2] and 
    // t in [-l2/2, l2/2] such that the norm of r12 + t*n2 - s*n1
    // is minimized
    T r12_dot_n1 = r12.dot(n1);
    T r12_dot_n2 = r12.dot(n2);
    T n1_dot_n2 = n1.dot(n2);
    T s_numer = r12_dot_n1 - n1_dot_n2 * r12_dot_n2;
    T t_numer = n1_dot_n2 * r12_dot_n1 - r12_dot_n2;
    T denom = 1 - n1_dot_n2 * n1_dot_n2;
    Matrix<T, 2, 1> dist = Matrix<T, 2, 1>::Zero(2);
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
                Matrix<T, 2, 1> q = r2 + half_l2 * n2; 
                s = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q);
                t = half_l2;
            }
            else if (t < Z && t >= -Z)      // In region 2
            {
                // In this case, set s = l1 / 2 and find t
                Matrix<T, 2, 1> q = r1 + half_l1 * n1;
                t = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q); 
                s = half_l1;
            }
            else if (t < -Z && t < Y)      // In region 3
            {
                // In this case, set t = -l2 / 2 and find s
                Matrix<T, 2, 1> q = r2 - half_l2 * n2;
                s = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q);
                t = -half_l2;
            }
            else    // t >= s + X and t < -s - X, in region 4
            {
                // In this case, set s = -l1 / 2 and find t
                Matrix<T, 2, 1> q = r1 - half_l1 * n1;
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
        Matrix<T, 2, 1> p1 = r1 - half_l1 * n1;                 // Endpoint of cell 1 for s = -l1 / 2
        Matrix<T, 2, 1> q1 = r1 + half_l1 * n1;                 // Endpoint of cell 1 for s = l1 / 2
        T t_p1 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, p1);   // Distance from cell 2 to p1
        T t_q1 = nearestCellBodyCoordToPoint<T>(r2, n2, half_l2, q1);   // Distance from cell 2 to q1
        Matrix<T, 2, 1> dist_from_p1 = (r2 + t_p1 * n2) - p1;   // Vector running towards cell 2 from p1
        Matrix<T, 2, 1> dist_from_q1 = (r2 + t_q1 * n2) - q1;   // Vector running towards cell 2 from q1
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
            Matrix<T, 2, 1> p2 = r2 - half_l2 * n2;                 // Endpoint of cell 2 for t = -l2 / 2
            Matrix<T, 2, 1> q2 = r2 + half_l2 * n2;                 // Endpoint of cell 2 for t = l2 / 2
            T s_p2 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, p2);   // Distance from cell 1 to p2
            T s_q2 = nearestCellBodyCoordToPoint<T>(r1, n1, half_l1, q2);   // Distance from cell 1 to q2
            Matrix<T, 2, 1> dist_from_p2 = (r1 + s_p2 * n1) - p2;   // Vector from p2 running towards cell 1
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

#endif
