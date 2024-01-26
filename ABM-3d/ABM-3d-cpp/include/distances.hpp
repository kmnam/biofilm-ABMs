/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/25/2024
 */

#ifndef DISTANCES_3D_HPP
#define DISTANCES_3D_HPP

#include <tuple>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/squared_distance_3.h>

using namespace Eigen; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;

/**
 *
 */
Segment_3 cellSegment(const Ref<const Array<double, 3, 1> >& r,
                      const Ref<const Array<double, 3, 1> >& n, const double half_l)
{
    Array<double, 3, 1> p_ = r - half_l * n;
    Array<double, 3, 1> q_ = r + half_l * n;
    Point_3 p(p_(0), p_(1), p_(2));
    Point_3 q(q_(0), q_(1), q_(2));

    return Segment_3(p, q);
}

/**
 */
std::tuple<Matrix<double, 3, 1>, double, double> distBetweenCells(const Segment_3& cell1,
                                                                  const Segment_3& cell2,
                                                                  const Ref<const Matrix<double, 3, 1> >& r1,
                                                                  const Ref<const Matrix<double, 3, 1> >& n1,
                                                                  const double half_l1,
                                                                  const Ref<const Matrix<double, 3, 1> >& r2,
                                                                  const Ref<const Matrix<double, 3, 1> >& n2,
                                                                  const double half_l2,
                                                                  const K& k)
{
    auto result = CGAL::Distance_3::internal::squared_distance(cell1, cell2, k);
    double s = CGAL::to_double(result.x) * 2 * half_l1 - half_l1;
    double t = CGAL::to_double(result.y) * 2 * half_l2 - half_l2;
    Matrix<double, 3, 1> dist = r2 + t * n2 - r1 - s * n1;

    return std::make_tuple(dist, s, t); 
}

// ----------------------------------------------------------------------- //
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
    Matrix<T, 3, 1> dist = Matrix<T, 3, 1>::Zero();
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

#endif
