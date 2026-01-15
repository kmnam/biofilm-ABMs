/**
 * Functions for alpha-wrapping in three dimensions.  
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/15/2026
 */

#ifndef ALPHA_WRAPPING_3D_HPP
#define ALPHA_WRAPPING_3D_HPP

#include <iostream>
#include <utility>
#include <tuple>
#include <vector>
#include <variant>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/boost/graph/iterator.h>
#include "indices.hpp"
#include "utils.hpp"

using std::sin;
using boost::multiprecision::sin; 
using std::cos; 
using boost::multiprecision::cos;
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::ceil; 
using boost::multiprecision::ceil; 

using namespace Eigen;
using K = CGAL::Exact_predicates_inexact_constructions_kernel; 
using Point_3 = K::Point_3;
using Ray_3 = K::Ray_3; 
using Segment_3 = K::Segment_3;
using Vector_3 = K::Vector_3; 
using Mesh = CGAL::Surface_mesh<Point_3>;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Mesh>;
using AABBTraits = CGAL::AABB_traits<K, Primitive>; 
using AABBTree = CGAL::AABB_tree<AABBTraits>; 

/**
 * Generate a Fibonacci lattice of n points on the unit disk. 
 *
 * @param n Number of points. 
 * @returns Corresponding Fibonacci lattice on the unit disk. 
 */
template <typename T>
Array<T, Dynamic, 2> fibonacciLatticeDisk(const int n)
{
    Array<T, Dynamic, 2> points(n, 2);
    const T three(3), five(5); 
    const T golden = boost::math::constants::pi<T>() * (three - sqrt(five)); 
     
    for (int i = 0; i < n; ++i)
    {
        T radius = sqrt((i * 0.5) / n); 
        T angle = i * golden; 
        while (angle >= boost::math::constants::two_pi<T>())
            angle -= boost::math::constants::two_pi<T>(); 
        points(i, 0) = radius * cos(angle); 
        points(i, 1) = radius * sin(angle); 
    }

    return points;  
}

/**
 * Generate a Fibonacci lattice of n points on a given rectangle of given
 * width (and length 2 * pi). 
 *
 * @param n Number of points.
 * @param l Rectangle width.
 * @returns Corresponding Fibonacci lattice on the given rectangle.  
 */
template <typename T>
Array<T, Dynamic, 2> fibonacciLatticeRectangle(const int n, const T l)
{
    Array<T, Dynamic, 2> points(n, 2);
    const T three(3), five(5); 
    const T golden = boost::math::constants::pi<T>() * (three - sqrt(five)); 
     
    for (int i = 0; i < n; ++i)
    {
        points(i, 0) = -0.5 * l + (i + 0.5) * l / n;
        T angle = i * golden;  
        while (angle >= boost::math::constants::two_pi<T>())
            angle -= boost::math::constants::two_pi<T>(); 
        points(i, 1) = angle; 
    }

    return points;  
}

/**
 * Generate a roughly equally spaced lattice of n points along the surface
 * of a given spherocylinder.
 *
 * @param r Spherocylinder center.
 * @param n Spherocylinder orientation.
 * @param l Spherocylinder length. 
 * @param R Spherocylinder radius.
 * @param L0 Shortest possible spherocylinder length. 
 * @param n_points_L0 Number of points to sample from a spherocylinder of 
 *                    length L0.
 * @returns Corresponding lattice of points on the spherocylinder surface.
 */
template <typename T>
Array<T, Dynamic, 3> fibonacciLatticeSpherocylinder(const Ref<const Matrix<T, 3, 1> >& r,
                                                    const Ref<const Matrix<T, 3, 1> >& n,
                                                    const T l, const T R,
                                                    const T L0, const int n_points_L0)
{
    // Get the endpoints of the cylinder
    T half_l = 0.5 * l;  
    Matrix<T, 3, 1> p = r - half_l * n; 
    Matrix<T, 3, 1> q = r + half_l * n; 

    // Get the number of points to sample from the spherocylinder ... 
    //
    // First divide the number of points for the unit-length spherocylinder 
    // between the caps and the body 
    int n_body_L0 = static_cast<int>(ceil((l / (l + 2 * R)) * n_points_L0));
    int n_caps_L0 = n_points_L0 - n_body_L0;
    
    // Ensure that the number of points on the caps is even 
    if (n_caps_L0 % 2 == 1)
    {
        n_caps_L0--;
        n_body_L0++;
    }

    // Increase the number of points along the body to match the given length
    const int n_caps = n_caps_L0; 
    const int n_body = static_cast<int>(ceil(n_body_L0 * l / L0));
    const int n_total = n_caps + n_body;

    // Generate an orthonormal basis in the spherocylinder's body frame
    Matrix<T, 3, 1> a, u, v;  
    if (n(1) == 1)
        a << 1, 0, 0; 
    else 
        a << 0, 1, 0; 
    u = n.cross(a);
    u /= u.norm();
    v = n.cross(u);
    Matrix<T, 3, 3> A; 
    A.col(0) = u; 
    A.col(1) = v; 
    A.col(2) = n; 

    // Generate points on the two caps ...
    //
    // This is done by projecting a Fibonacci lattice of points on the unit 
    // disk onto each cap (and translating/rotating/scaling the points
    // appropriately) 
    Array<T, Dynamic, 2> points_disk = fibonacciLatticeDisk<T>(n_caps / 2); 
    Array<T, Dynamic, 3> points_cap1(n_caps / 2, 3), points_cap2(n_caps / 2, 3);
    for (int i = 0; i < n_caps / 2; ++i)
    {
        T radius = sqrt(static_cast<T>(i + 0.5) / (n_caps / 2)); 
        Matrix<T, 3, 1> w; 
        w << points_disk(i, 0), points_disk(i, 1), -sqrt(1 - radius * radius); 
        points_cap1.row(i) = (p + R * A * w).transpose();
        w(2) *= -1;  
        points_cap2.row(i) = (q + R * A * w).transpose();  
    } 

    // Generate points along the cylindrical body ...
    //
    // This is done by wrapping a Fibonacci lattice of points on the rectangle
    // with the appropriate dimensions around the cylinder (and translating/
    // rotating/scaling the points appropriately) 
    Array<T, Dynamic, 2> points_rect = fibonacciLatticeRectangle<T>(n_body, l);
    Array<T, Dynamic, 3> points_body(n_body, 3);
    for (int i = 0; i < n_body; ++i)
    {
        Matrix<T, 3, 1> w; 
        w << R * cos(points_rect(i, 1)), R * sin(points_rect(i, 1)), points_rect(i, 0);  
        points_body.row(i) = (r + A * w).transpose(); 
    } 

    // Concatenate the three arrays and return 
    Array<T, Dynamic, 3> points(n_total, 3); 
    points(Eigen::seqN(0, n_caps / 2), Eigen::all) = points_cap1; 
    points(Eigen::seqN(n_caps / 2, n_caps / 2), Eigen::all) = points_cap2; 
    points(Eigen::seqN(n_caps, n_body), Eigen::all) = points_body; 
    return points;  
}

/**
 * Sample points along the surface of each cell in the given population. 
 */
template <typename T>
Array<T, Dynamic, 3> getSurfacePoints(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                      const T R, const T L0, const int n_points_L0)
{
    Array<T, Dynamic, 3> points(0, 3);
    int n_total = 0; 

    // For each cell ...
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Sample the corresponding surface points
        Matrix<T, 3, 1> r, n; 
        r << cells(i, __colidx_rx), cells(i, __colidx_ry), cells(i, __colidx_rz); 
        n << cells(i, __colidx_nx), cells(i, __colidx_ny), cells(i, __colidx_nz); 
        Array<T, Dynamic, 3> points_surface = fibonacciLatticeSpherocylinder<T>(
            r, n, cells(i, __colidx_l), R, L0, n_points_L0
        );
        n_total += points_surface.rows();
        points.conservativeResize(n_total, 3); 
        points(Eigen::seq(n_total - points_surface.rows(), n_total - 1), Eigen::all) = points_surface; 
    }

    return points;  
}

/**
 * Alpha-wrapping. 
 */
template <typename T>
Mesh alphaWrap(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
               const T R, const T L0, const int n_points_L0, const double alpha,
               const double offset)
{
    // Sample a selection of points along the surface and interior of each cell 
    Array<T, Dynamic, 3> points = getSurfacePoints<T>(cells, R, L0, n_points_L0);
    std::cout << "Sampled " << points.rows() << " points" << std::endl; 

    // Prepare the points to be processed by CGAL::alpha_wrap_3()
    std::vector<Point_3> points_vec; 
    for (int i = 0; i < points.rows(); ++i)
    {
        Point_3 p(
            static_cast<double>(points(i, 0)),
            static_cast<double>(points(i, 1)),
            static_cast<double>(points(i, 2))
        ); 
        points_vec.push_back(p); 
    }

    // Wrap the points 
    Mesh wrap; 
    CGAL::alpha_wrap_3(points_vec, alpha, offset, wrap);
    std::cout << "Done with alpha-wrapping: " 
              << CGAL::num_vertices(wrap) << " vertices, " 
              << CGAL::num_faces(wrap) << " faces\n";

    return wrap;  
}

/**
 * Given an alpha-wrapping of a population of cells, identify the basal 
 * layer of points in the mesh.
 */
template <typename T>
Array<T, Dynamic, 3> getBasalPoints(const Mesh& wrap, const T R,
                                    boost::random::mt19937& rng)
{
    const double eps1 = 1e-3 * static_cast<double>(R);
    const double eps2 = 1e-6; 
    boost::random::uniform_01<> dist; 

    // Build the AABB tree 
    AABBTree tree(CGAL::faces(wrap).first, CGAL::faces(wrap).second, wrap);
    tree.accelerate_distance_queries(); 

    // Get a bounding box for the mesh 
    auto bbox = CGAL::Polygon_mesh_processing::bbox(wrap);
    double z0 = bbox.zmin() - (bbox.zmax() - bbox.zmin());

    // For each vertex in the mesh ...
    Array<T, Dynamic, 3> basal_points(0, 3); 
    int n_basal = 0;   
    for (const auto& vi : CGAL::vertices(wrap))
    {
        // Define the ray pointing straight upwards from under the vertex 
        const Point_3 p = wrap.point(vi);
        const Point_3 q(
            p.x() + 0.5 * eps2 * (2 * dist(rng) - 1),
            p.y() + 0.5 * eps2 * (2 * dist(rng) - 1),
            z0
        ); 
        const Ray_3 ray(q, Vector_3(0, 0, 1)); 

        // Get the intersection between the mesh and the ray 
        auto hit = tree.first_intersection(ray); 

        // If the hit is ill-defined, then skip the vertex 
        if (!hit)
        {
            continue; 
        }
        else    // Otherwise, process the intersection 
        {
            const auto& intersect = hit->first; 

            // Is the intersection a point? 
            if (const Point_3* x = std::get_if<Point_3>(&intersect))
            {
                // Is the intersection point close to the vertex itself?
                double dist = CGAL::squared_distance(*x, p);
                if (dist < eps1 * eps1)
                {
                    // If so, the vertex is in the basal layer
                    n_basal++; 
                    basal_points.conservativeResize(n_basal, 3); 
                    basal_points(n_basal - 1, 0) = static_cast<T>(p.x()); 
                    basal_points(n_basal - 1, 1) = static_cast<T>(p.y()); 
                    basal_points(n_basal - 1, 2) = static_cast<T>(p.z()); 
                }
            }
            // Otherwise, the intersection must be a segment
            else
            {
                const Segment_3* s = std::get_if<Segment_3>(&intersect);

                // Is the segment close to the vertex itself?
                double dist = CGAL::squared_distance(p, *s);
                if (dist < eps1 * eps1)
                {
                    // If so, the vertex is in the basal layer
                    n_basal++; 
                    basal_points.conservativeResize(n_basal, 3); 
                    basal_points(n_basal - 1, 0) = static_cast<T>(p.x()); 
                    basal_points(n_basal - 1, 1) = static_cast<T>(p.y()); 
                    basal_points(n_basal - 1, 2) = static_cast<T>(p.z()); 
                }
            } 
        }
    }

    return basal_points;  
}

#endif
