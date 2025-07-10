/**
 * Test module for the `SimplicialComplex3D` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/27/2025
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/topology.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T;

using std::sin; 
using boost::multiprecision::sin;
using std::cos; 
using boost::multiprecision::cos; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::abs; 
using boost::multiprecision::abs;

/**
 * Return true if there is no solution to the linear system A * x = b, by 
 * checking whether there is an inconsistency in the row echelon form. 
 *
 * @param A Input matrix. 
 * @param b Input vector. 
 * @returns True if there is no solution to the linear system A * x = b. 
 */
template <typename T>
bool containsInconsistency(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                           const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    Matrix<T, Dynamic, Dynamic> system(A.rows(), A.cols() + 1); 
    system(Eigen::all, Eigen::seq(0, A.cols() - 1)) = A; 
    system.col(A.cols()) = b; 
    system = ::rowEchelonForm<T>(system); 
    bool found_inconsistency = false; 
    for (int i = 0; i < A.rows(); ++i)
    {
        if ((system.row(i).head(A.cols()).array() == 0).all() && system(i, A.cols()) != 0)
        {
            found_inconsistency = true;
            break; 
        }
    }

    return found_inconsistency; 
}

/**
 * Generate a collection of 0-simplices. 
 */
SimplicialComplex3D<T> complex_points()
{
    Array<T, Dynamic, 3> points(3, 3); 
    Array<int, Dynamic, 2> edges(0, 2); 
    Array<int, Dynamic, 3> triangles(0, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 1, 0, 0,
              0, 1, 0,
              0, 0, 1; 
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a single 2-simplex. 
 */
SimplicialComplex3D<T> complex_triangle()
{
    Array<T, Dynamic, 3> points(3, 3); 
    Array<int, Dynamic, 2> edges(3, 2); 
    Array<int, Dynamic, 3> triangles(1, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 1, 0, 0,
              0, 1, 0,
              0, 0, 1; 
    edges << 0, 1,
             0, 2,
             1, 2;
    triangles << 0, 1, 2;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate the boundary of a 2-simplex (i.e., a cycle). 
 */
SimplicialComplex3D<T> complex_cycle()
{
    Array<T, Dynamic, 3> points(3, 3); 
    Array<int, Dynamic, 2> edges(3, 2); 
    Array<int, Dynamic, 3> triangles(0, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 1, 0, 0,
              0, 1, 0,
              0, 0, 1; 
    edges << 0, 1,
             0, 2,
             1, 2;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a single 2-simplex with 3 additional 1-simplices.  
 */
SimplicialComplex3D<T> complex_triangles_with_appendages()
{
    Array<T, Dynamic, 3> points(7, 7); 
    Array<int, Dynamic, 2> edges(7, 2); 
    Array<int, Dynamic, 3> triangles(1, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 1, 0, 0,
              0, 1, 0,
              0, 0, 1,
              2, 0, 0,
              0, 2, 0,
              0, 0, 2,
              1, 1, 3;
    edges << 0, 1,
             0, 2,
             1, 2,
             0, 3,
             1, 4,
             2, 5,
             5, 6;
    triangles << 0, 1, 2;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a two-dimensional mesh of 2-simplices. 
 */
SimplicialComplex3D<T> complex_2d_mesh()
{
    Array<T, Dynamic, 3> points(9, 3); 
    Array<int, Dynamic, 2> edges(16, 2); 
    Array<int, Dynamic, 3> triangles(8, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    T cos60 = cos(boost::math::constants::third_pi<T>()); 
    T sin60 = sin(boost::math::constants::third_pi<T>()); 
    points << 0, 0, 0,
              1, 0, 0,
              cos60, sin60, 0,
              2, 0, 0,
              1 + cos60, sin60, 0,
              1, 2 * sin60, 0,
              2 + cos60, sin60, 0,
              cos60, -sin60, 0,
              1 + cos60, -sin60, 0;
    edges << 0, 1,
             0, 2,
             1, 2,
             1, 3,
             1, 4,
             3, 4,
             2, 4,
             2, 5,
             4, 5,
             3, 6,
             4, 6,
             0, 7,
             1, 7,
             1, 8,
             3, 8,
             7, 8;
    triangles << 0, 1, 2,
                 1, 3, 4,
                 1, 2, 4,
                 2, 4, 5,
                 3, 4, 6,
                 0, 1, 7,
                 1, 3, 8,
                 1, 7, 8;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a two-dimensional mesh of 2-simplices with a hole in the middle. 
 */
SimplicialComplex3D<T> complex_2d_mesh_with_hole()
{
    Array<T, Dynamic, 3> points(9, 3);
    Array<int, Dynamic, 2> edges(15, 2); 
    Array<int, Dynamic, 3> triangles(6, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    T cos60 = cos(boost::math::constants::third_pi<T>()); 
    T sin60 = sin(boost::math::constants::third_pi<T>()); 
    points << 0, 0, 0,
              1, 0, 0,
              cos60, sin60, 0,
              2, 0, 0,
              1 + cos60, sin60, 0,
              1, 2 * sin60, 0,
              2 + cos60, sin60, 0,
              cos60, -sin60, 0,
              1 + cos60, -sin60, 0;
    edges << 0, 1,
             0, 2,
             1, 2,
             1, 3,
             3, 4,
             2, 4,
             2, 5,
             4, 5,
             3, 6,
             4, 6,
             0, 7,
             1, 7,
             1, 8,
             3, 8,
             7, 8;
    triangles << 0, 1, 2,
                 2, 4, 5,
                 3, 4, 6,
                 0, 1, 7,
                 1, 3, 8,
                 1, 7, 8;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a two-dimensional triangulation of an annulus. 
 */
SimplicialComplex3D<T> complex_annulus()
{
    Array<T, Dynamic, 3> points(12, 3); 
    Array<int, Dynamic, 2> edges(24, 2); 
    Array<int, Dynamic, 3> triangles(12, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    const T r1 = 1.0; 
    const T r2 = 2.0; 
    for (int i = 0; i < 6; ++i)
    { 
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6) * boost::math::constants::two_pi<T>()
        ); 
        points(i, 0) = r1 * cos(theta); 
        points(i, 1) = r1 * sin(theta);
        points(6 + i, 0) = r2 * cos(theta); 
        points(6 + i, 1) = r2 * sin(theta);  
    }
    edges <<  0,  1,
              0,  5,
              0,  6,
              0, 11,
              1,  2,
              1,  6, 
              1,  7, 
              2,  3, 
              2,  7,
              2,  8,
              3,  4, 
              3,  8,
              3,  9,
              4,  5,
              4,  9,
              4, 10,
              5, 10,
              5, 11,
              6,  7,
              6, 11,
              7,  8,
              8,  9,
              9, 10,
             10, 11;
    triangles << 0,  1,  6,
                 0,  5, 11,
                 0,  6, 11,
                 1,  2,  7,
                 1,  6,  7,
                 2,  3,  8,
                 2,  7,  8,
                 3,  4,  9,
                 3,  8,  9,
                 4,  5, 10,
                 4,  9, 10,
                 5, 10, 11;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a single 3-simplex. 
 */
SimplicialComplex3D<T> complex_tetrahedron()
{
    Array<T, Dynamic, 3> points(4, 3);
    Array<int, Dynamic, 2> edges(6, 2); 
    Array<int, Dynamic, 3> triangles(4, 3); 
    Array<int, Dynamic, 4> tetrahedra(1, 4);
    points <<  1,  0, -1.0 / sqrt(2.0),
              -1,  0, -1.0 / sqrt(2.0),
               0,  1,  1.0 / sqrt(2.0),
               0, -1,  1.0 / sqrt(2.0);
    edges << 0, 1,
             0, 2,
             0, 3,
             1, 2,
             1, 3,
             2, 3;
    triangles << 0, 1, 2,
                 0, 1, 3,
                 0, 2, 3,
                 1, 2, 3;
    tetrahedra << 0, 1, 2, 3;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

TEST_CASE("Tests for basic methods", "[SimplicialComplex3D]")
{
    // ------------------------------------------------------------- // 
    // Test for discrete set of points
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 0);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 3);  
    REQUIRE(cplex.getNumSimplices() == 3); 
    REQUIRE(cplex.getNumSimplices(0) == 3);
    REQUIRE_THROWS(cplex.getNumSimplices(1)); 
    REQUIRE_THROWS(cplex.getNumSimplices(2)); 
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex 
    Array<int, Dynamic, 1> simplices = cplex.getSimplices<0>();
    REQUIRE(simplices.size() == 3);
    for (int i = 0; i < 3; ++i)
        REQUIRE(simplices(i) == i);

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 3);  
    REQUIRE(cplex.getNumSimplices() == 7); 
    REQUIRE(cplex.getNumSimplices(0) == 3);
    REQUIRE(cplex.getNumSimplices(1) == 3);
    REQUIRE(cplex.getNumSimplices(2) == 1);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    Array<int, Dynamic, 1> simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 3);
    for (int i = 0; i < 3; ++i)
        REQUIRE(simplices0(i) == i);
    Array<int, Dynamic, 2> simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 3);
    REQUIRE(simplices1(0, 0) == 0);
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);
    REQUIRE(simplices1(1, 1) == 2); 
    REQUIRE(simplices1(2, 0) == 1);
    REQUIRE(simplices1(2, 1) == 2);
    Array<int, Dynamic, 3> simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 1);
    REQUIRE(simplices2(0, 0) == 0);
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 2);

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 1);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 3);  
    REQUIRE(cplex.getNumSimplices() == 6); 
    REQUIRE(cplex.getNumSimplices(0) == 3);
    REQUIRE(cplex.getNumSimplices(1) == 3);
    REQUIRE_THROWS(cplex.getNumSimplices(2));
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 3);
    for (int i = 0; i < 3; ++i)
        REQUIRE(simplices0(i) == i);
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 3);
    REQUIRE(simplices1(0, 0) == 0);
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);
    REQUIRE(simplices1(1, 1) == 2); 
    REQUIRE(simplices1(2, 0) == 1);
    REQUIRE(simplices1(2, 1) == 2);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- // 
    cplex = complex_triangles_with_appendages(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 7);  
    REQUIRE(cplex.getNumSimplices() == 15); 
    REQUIRE(cplex.getNumSimplices(0) == 7);
    REQUIRE(cplex.getNumSimplices(1) == 7);
    REQUIRE(cplex.getNumSimplices(2) == 1);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 7);
    for (int i = 0; i < 7; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 6
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 7);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 2)
    REQUIRE(simplices1(1, 1) == 2); 
    REQUIRE(simplices1(2, 0) == 0);     // (0, 3)
    REQUIRE(simplices1(2, 1) == 3);
    REQUIRE(simplices1(3, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(3, 1) == 2);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 4)
    REQUIRE(simplices1(4, 1) == 4);
    REQUIRE(simplices1(5, 0) == 2);     // (2, 5)
    REQUIRE(simplices1(5, 1) == 5);
    REQUIRE(simplices1(6, 0) == 5);     // (5, 6)
    REQUIRE(simplices1(6, 1) == 6);
    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 1);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 2)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 2);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 9);  
    REQUIRE(cplex.getNumSimplices() == 33); 
    REQUIRE(cplex.getNumSimplices(0) == 9);
    REQUIRE(cplex.getNumSimplices(1) == 16);
    REQUIRE(cplex.getNumSimplices(2) == 8);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 9);
    for (int i = 0; i < 9; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 8
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 16);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 2)
    REQUIRE(simplices1(1, 1) == 2);
    REQUIRE(simplices1(2, 0) == 0);     // (0, 7)
    REQUIRE(simplices1(2, 1) == 7); 
    REQUIRE(simplices1(3, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(3, 1) == 2);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 3)
    REQUIRE(simplices1(4, 1) == 3);
    REQUIRE(simplices1(5, 0) == 1);     // (1, 4)
    REQUIRE(simplices1(5, 1) == 4);
    REQUIRE(simplices1(6, 0) == 1);     // (1, 7)
    REQUIRE(simplices1(6, 1) == 7);
    REQUIRE(simplices1(7, 0) == 1);     // (1, 8)
    REQUIRE(simplices1(7, 1) == 8);
    REQUIRE(simplices1(8, 0) == 2);     // (2, 4)
    REQUIRE(simplices1(8, 1) == 4);
    REQUIRE(simplices1(9, 0) == 2);     // (2, 5)
    REQUIRE(simplices1(9, 1) == 5);
    REQUIRE(simplices1(10, 0) == 3);    // (3, 4)
    REQUIRE(simplices1(10, 1) == 4);
    REQUIRE(simplices1(11, 0) == 3);    // (3, 6)
    REQUIRE(simplices1(11, 1) == 6);
    REQUIRE(simplices1(12, 0) == 3);    // (3, 8)
    REQUIRE(simplices1(12, 1) == 8);
    REQUIRE(simplices1(13, 0) == 4);    // (4, 5)
    REQUIRE(simplices1(13, 1) == 5);
    REQUIRE(simplices1(14, 0) == 4);    // (4, 6)
    REQUIRE(simplices1(14, 1) == 6);
    REQUIRE(simplices1(15, 0) == 7);    // (7, 8)
    REQUIRE(simplices1(15, 1) == 8);
    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 8);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 2)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 2);
    REQUIRE(simplices2(1, 0) == 0);     // (0, 1, 7)
    REQUIRE(simplices2(1, 1) == 1);
    REQUIRE(simplices2(1, 2) == 7);
    REQUIRE(simplices2(2, 0) == 1);     // (1, 2, 4)
    REQUIRE(simplices2(2, 1) == 2);
    REQUIRE(simplices2(2, 2) == 4); 
    REQUIRE(simplices2(3, 0) == 1);     // (1, 3, 4)
    REQUIRE(simplices2(3, 1) == 3);
    REQUIRE(simplices2(3, 2) == 4);
    REQUIRE(simplices2(4, 0) == 1);     // (1, 3, 8)
    REQUIRE(simplices2(4, 1) == 3);
    REQUIRE(simplices2(4, 2) == 8); 
    REQUIRE(simplices2(5, 0) == 1);     // (1, 7, 8)
    REQUIRE(simplices2(5, 1) == 7);
    REQUIRE(simplices2(5, 2) == 8); 
    REQUIRE(simplices2(6, 0) == 2);     // (2, 4, 5)
    REQUIRE(simplices2(6, 1) == 4);
    REQUIRE(simplices2(6, 2) == 5); 
    REQUIRE(simplices2(7, 0) == 3);     // (3, 4, 6)
    REQUIRE(simplices2(7, 1) == 4);
    REQUIRE(simplices2(7, 2) == 6);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 9);  
    REQUIRE(cplex.getNumSimplices() == 30); 
    REQUIRE(cplex.getNumSimplices(0) == 9);
    REQUIRE(cplex.getNumSimplices(1) == 15);
    REQUIRE(cplex.getNumSimplices(2) == 6);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 9);
    for (int i = 0; i < 9; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 8
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 15);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 2)
    REQUIRE(simplices1(1, 1) == 2);
    REQUIRE(simplices1(2, 0) == 0);     // (0, 7)
    REQUIRE(simplices1(2, 1) == 7); 
    REQUIRE(simplices1(3, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(3, 1) == 2);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 3)
    REQUIRE(simplices1(4, 1) == 3);
    REQUIRE(simplices1(5, 0) == 1);     // (1, 7)
    REQUIRE(simplices1(5, 1) == 7);
    REQUIRE(simplices1(6, 0) == 1);     // (1, 8)
    REQUIRE(simplices1(6, 1) == 8);
    REQUIRE(simplices1(7, 0) == 2);     // (2, 4)
    REQUIRE(simplices1(7, 1) == 4);
    REQUIRE(simplices1(8, 0) == 2);     // (2, 5)
    REQUIRE(simplices1(8, 1) == 5);
    REQUIRE(simplices1(9, 0) == 3);     // (3, 4)
    REQUIRE(simplices1(9, 1) == 4);
    REQUIRE(simplices1(10, 0) == 3);    // (3, 6)
    REQUIRE(simplices1(10, 1) == 6);
    REQUIRE(simplices1(11, 0) == 3);    // (3, 8)
    REQUIRE(simplices1(11, 1) == 8);
    REQUIRE(simplices1(12, 0) == 4);    // (4, 5)
    REQUIRE(simplices1(12, 1) == 5);
    REQUIRE(simplices1(13, 0) == 4);    // (4, 6)
    REQUIRE(simplices1(13, 1) == 6);
    REQUIRE(simplices1(14, 0) == 7);    // (7, 8)
    REQUIRE(simplices1(14, 1) == 8);
    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 6);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 2)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 2);
    REQUIRE(simplices2(1, 0) == 0);     // (0, 1, 7)
    REQUIRE(simplices2(1, 1) == 1);
    REQUIRE(simplices2(1, 2) == 7);
    REQUIRE(simplices2(2, 0) == 1);     // (1, 3, 8)
    REQUIRE(simplices2(2, 1) == 3);
    REQUIRE(simplices2(2, 2) == 8); 
    REQUIRE(simplices2(3, 0) == 1);     // (1, 7, 8)
    REQUIRE(simplices2(3, 1) == 7);
    REQUIRE(simplices2(3, 2) == 8); 
    REQUIRE(simplices2(4, 0) == 2);     // (2, 4, 5)
    REQUIRE(simplices2(4, 1) == 4);
    REQUIRE(simplices2(4, 2) == 5); 
    REQUIRE(simplices2(5, 0) == 3);     // (3, 4, 6)
    REQUIRE(simplices2(5, 1) == 4);
    REQUIRE(simplices2(5, 2) == 6);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron 
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 3);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 4);  
    REQUIRE(cplex.getNumSimplices() == 15); 
    REQUIRE(cplex.getNumSimplices(0) == 4);
    REQUIRE(cplex.getNumSimplices(1) == 6);
    REQUIRE(cplex.getNumSimplices(2) == 4);
    REQUIRE(cplex.getNumSimplices(3) == 1); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 4);
    for (int i = 0; i < 4; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 3
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 6);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 2)
    REQUIRE(simplices1(1, 1) == 2); 
    REQUIRE(simplices1(2, 0) == 0);     // (0, 3)
    REQUIRE(simplices1(2, 1) == 3);
    REQUIRE(simplices1(3, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(3, 1) == 2);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 3)
    REQUIRE(simplices1(4, 1) == 3);
    REQUIRE(simplices1(5, 0) == 2);     // (2, 3)
    REQUIRE(simplices1(5, 1) == 3);
    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 4);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 2)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 2);
    REQUIRE(simplices2(1, 0) == 0);     // (0, 1, 3)
    REQUIRE(simplices2(1, 1) == 1);
    REQUIRE(simplices2(1, 2) == 3);
    REQUIRE(simplices2(2, 0) == 0);     // (0, 2, 3)
    REQUIRE(simplices2(2, 1) == 2);
    REQUIRE(simplices2(2, 2) == 3);
    REQUIRE(simplices2(3, 0) == 1);     // (1, 2, 3)
    REQUIRE(simplices2(3, 1) == 2);
    REQUIRE(simplices2(3, 2) == 3);
    Array<int, Dynamic, 4> simplices3 = cplex.getSimplices<3>();
    REQUIRE(simplices3.rows() == 1);
    REQUIRE(simplices3(0, 0) == 0);     // (0, 1, 2, 3)
    REQUIRE(simplices3(0, 1) == 1);
    REQUIRE(simplices3(0, 2) == 2);
    REQUIRE(simplices3(0, 3) == 3); 
}

TEST_CASE("Tests for boundary calculations", "[SimplicialComplex3D::getBoundary()]")
{
    // ------------------------------------------------------------- // 
    // Test for discrete set of points
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();
    SimplicialComplex3D<T> boundary = cplex.getBoundary();
    Array<int, Dynamic, 1> boundary_points = boundary.getSimplices<0>(); 
    Array<int, Dynamic, 2> boundary_edges = boundary.getSimplices<1>(); 
    Array<int, Dynamic, 3> boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 3); 
    REQUIRE(boundary_edges.rows() == 0); 
    REQUIRE(boundary_triangles.rows() == 0);
    REQUIRE(boundary_points(0) == 0); 
    REQUIRE(boundary_points(1) == 1); 
    REQUIRE(boundary_points(2) == 2); 

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>(); 
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 3); 
    REQUIRE(boundary_edges.rows() == 3); 
    REQUIRE(boundary_triangles.rows() == 0);
    REQUIRE(boundary_points(0) == 0);
    REQUIRE(boundary_points(1) == 1); 
    REQUIRE(boundary_points(2) == 2);
    REQUIRE(boundary_edges(0, 0) == 0); 
    REQUIRE(boundary_edges(0, 1) == 1); 
    REQUIRE(boundary_edges(1, 0) == 0); 
    REQUIRE(boundary_edges(1, 1) == 2);  
    REQUIRE(boundary_edges(2, 0) == 1); 
    REQUIRE(boundary_edges(2, 1) == 2); 

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>();
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 0); 
    REQUIRE(boundary_edges.rows() == 0);
    REQUIRE(boundary_triangles.rows() == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- // 
    cplex = complex_triangles_with_appendages();
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>(); 
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 7); 
    REQUIRE(boundary_edges.rows() == 7); 
    REQUIRE(boundary_triangles.rows() == 0);
    for (int i = 0; i < 7; ++i)
        REQUIRE(boundary_points(i) == i);
    REQUIRE(boundary_edges(0, 0) == 0);     // (0, 1)
    REQUIRE(boundary_edges(0, 1) == 1); 
    REQUIRE(boundary_edges(1, 0) == 0);     // (0, 2)
    REQUIRE(boundary_edges(1, 1) == 2); 
    REQUIRE(boundary_edges(2, 0) == 0);     // (0, 3)
    REQUIRE(boundary_edges(2, 1) == 3);
    REQUIRE(boundary_edges(3, 0) == 1);     // (1, 2)
    REQUIRE(boundary_edges(3, 1) == 2);
    REQUIRE(boundary_edges(4, 0) == 1);     // (1, 4)
    REQUIRE(boundary_edges(4, 1) == 4);
    REQUIRE(boundary_edges(5, 0) == 2);     // (2, 5)
    REQUIRE(boundary_edges(5, 1) == 5);
    REQUIRE(boundary_edges(6, 0) == 5);     // (5, 6)
    REQUIRE(boundary_edges(6, 1) == 6);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh(); 
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>(); 
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 8); 
    REQUIRE(boundary_edges.rows() == 8); 
    REQUIRE(boundary_triangles.rows() == 0);
    REQUIRE(boundary_points(0) == 0);
    REQUIRE(boundary_points(1) == 2); 
    REQUIRE(boundary_points(2) == 3);
    REQUIRE(boundary_points(3) == 4);
    REQUIRE(boundary_points(4) == 5); 
    REQUIRE(boundary_points(5) == 6);
    REQUIRE(boundary_points(6) == 7);
    REQUIRE(boundary_points(7) == 8);
    REQUIRE(boundary_edges(0, 0) == 0);     // (0, 2)
    REQUIRE(boundary_edges(0, 1) == 2);
    REQUIRE(boundary_edges(1, 0) == 0);     // (0, 7)
    REQUIRE(boundary_edges(1, 1) == 7); 
    REQUIRE(boundary_edges(2, 0) == 2);     // (2, 5)
    REQUIRE(boundary_edges(2, 1) == 5);
    REQUIRE(boundary_edges(3, 0) == 3);     // (3, 6)
    REQUIRE(boundary_edges(3, 1) == 6);
    REQUIRE(boundary_edges(4, 0) == 3);     // (3, 8)
    REQUIRE(boundary_edges(4, 1) == 8);
    REQUIRE(boundary_edges(5, 0) == 4);     // (4, 5)
    REQUIRE(boundary_edges(5, 1) == 5);
    REQUIRE(boundary_edges(6, 0) == 4);     // (4, 6)
    REQUIRE(boundary_edges(6, 1) == 6);
    REQUIRE(boundary_edges(7, 0) == 7);     // (7, 8)
    REQUIRE(boundary_edges(7, 1) == 8);
    
    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>(); 
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 9); 
    REQUIRE(boundary_edges.rows() == 12); 
    REQUIRE(boundary_triangles.rows() == 0);
    for (int i = 0; i < 9; ++i)
        REQUIRE(boundary_points(i) == i);    // 0, ..., 8
    REQUIRE(boundary_edges(0, 0) == 0);      // (0, 2)
    REQUIRE(boundary_edges(0, 1) == 2);
    REQUIRE(boundary_edges(1, 0) == 0);      // (0, 7)
    REQUIRE(boundary_edges(1, 1) == 7); 
    REQUIRE(boundary_edges(2, 0) == 1);      // (1, 2)
    REQUIRE(boundary_edges(2, 1) == 2);
    REQUIRE(boundary_edges(3, 0) == 1);      // (1, 3)
    REQUIRE(boundary_edges(3, 1) == 3);
    REQUIRE(boundary_edges(4, 0) == 2);      // (2, 4)
    REQUIRE(boundary_edges(4, 1) == 4);
    REQUIRE(boundary_edges(5, 0) == 2);      // (2, 5)
    REQUIRE(boundary_edges(5, 1) == 5);
    REQUIRE(boundary_edges(6, 0) == 3);      // (3, 4)
    REQUIRE(boundary_edges(6, 1) == 4);
    REQUIRE(boundary_edges(7, 0) == 3);      // (3, 6)
    REQUIRE(boundary_edges(7, 1) == 6);
    REQUIRE(boundary_edges(8, 0) == 3);      // (3, 8)
    REQUIRE(boundary_edges(8, 1) == 8);
    REQUIRE(boundary_edges(9, 0) == 4);      // (4, 5)
    REQUIRE(boundary_edges(9, 1) == 5);
    REQUIRE(boundary_edges(10, 0) == 4);     // (4, 6)
    REQUIRE(boundary_edges(10, 1) == 6);
    REQUIRE(boundary_edges(11, 0) == 7);     // (7, 8)
    REQUIRE(boundary_edges(11, 1) == 8);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();
    boundary = cplex.getBoundary();
    boundary_points = boundary.getSimplices<0>(); 
    boundary_edges = boundary.getSimplices<1>(); 
    boundary_triangles = boundary.getSimplices<2>();
    REQUIRE(boundary_points.size() == 4); 
    REQUIRE(boundary_edges.rows() == 6); 
    REQUIRE(boundary_triangles.rows() == 4);
    for (int i = 0; i < 4; ++i)
        REQUIRE(boundary_points(i) == i);    // 0, ..., 3
    REQUIRE(boundary_edges(0, 0) == 0);      // (0, 1)
    REQUIRE(boundary_edges(0, 1) == 1); 
    REQUIRE(boundary_edges(1, 0) == 0);      // (0, 2)
    REQUIRE(boundary_edges(1, 1) == 2); 
    REQUIRE(boundary_edges(2, 0) == 0);      // (0, 3)
    REQUIRE(boundary_edges(2, 1) == 3);
    REQUIRE(boundary_edges(3, 0) == 1);      // (1, 2)
    REQUIRE(boundary_edges(3, 1) == 2);
    REQUIRE(boundary_edges(4, 0) == 1);      // (1, 3)
    REQUIRE(boundary_edges(4, 1) == 3);
    REQUIRE(boundary_edges(5, 0) == 2);      // (2, 3)
    REQUIRE(boundary_edges(5, 1) == 3);
    REQUIRE(boundary_triangles(0, 0) == 0);     // (0, 1, 2)
    REQUIRE(boundary_triangles(0, 1) == 1);
    REQUIRE(boundary_triangles(0, 2) == 2);
    REQUIRE(boundary_triangles(1, 0) == 0);     // (0, 1, 3)
    REQUIRE(boundary_triangles(1, 1) == 1);
    REQUIRE(boundary_triangles(1, 2) == 3);
    REQUIRE(boundary_triangles(2, 0) == 0);     // (0, 2, 3)
    REQUIRE(boundary_triangles(2, 1) == 2);
    REQUIRE(boundary_triangles(2, 2) == 3);
    REQUIRE(boundary_triangles(3, 0) == 1);     // (1, 2, 3)
    REQUIRE(boundary_triangles(3, 1) == 2);
    REQUIRE(boundary_triangles(3, 2) == 3);
}

TEST_CASE(
    "Tests for boundary homomorphism calculations",
    "[SimplicialComplex3D::getBoundaryHomomorphism()]"
)
{
    Matrix<Fp<0>, Dynamic, Dynamic> del1, del2, del3; 
    Matrix<Fp<2>, Dynamic, Dynamic> del1_p2, del2_p2, del3_p2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points(); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism<0>(0));
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism<0>(1)); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism<0>(2)); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism<0>(3));

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [0, 1, 2] to [1, 2] - [0, 2] + [0, 1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 3 
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 1
    REQUIRE(del2(0) == 1); 
    REQUIRE(del2(1) == -1); 
    REQUIRE(del2(2) == 1);

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send: 
    // -> [0, 1] to [1] - [0]
    // -> [0, 2] to [2] - [0]
    // -> [1, 2] to [2] - [1]
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 3 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 3
    REQUIRE(del1(0, 0) == -1);    // [1] - [0]
    REQUIRE(del1(1, 0) == 1); 
    REQUIRE(del1(2, 0) == 0); 
    REQUIRE(del1(0, 1) == -1);    // [2] - [0]
    REQUIRE(del1(1, 1) == 0);
    REQUIRE(del1(2, 1) == 1); 
    REQUIRE(del1(0, 2) == 0);     // [2] - [1]
    REQUIRE(del1(1, 2) == -1);
    REQUIRE(del1(2, 2) == 1); 

    // The homomorphism should map 4 * [0, 1] + 2 * [0, 2] - 3 * [1, 2]
    // to 4 * ([1] - [0]) + 2 * ([2] - [0]) - 3 * ([2] - [1]), which is 
    // -6 * [0] + 7 * [1] - [2]
    Matrix<Fp<0>, Dynamic, 1> x(3), y(3);
    x << 4, 2, -3;
    y = del1 * x;
    REQUIRE(y(0) == -6); 
    REQUIRE(y(1) == 7); 
    REQUIRE(y(2) == -1);

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send: 
    // -> [0, 1] to [1] - [0]
    // -> [0, 2] to [2] - [0]
    // -> [1, 2] to [2] - [1]
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 3 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 3
    REQUIRE(del1(0, 0) == -1);    // [1] - [0]
    REQUIRE(del1(1, 0) == 1); 
    REQUIRE(del1(2, 0) == 0); 
    REQUIRE(del1(0, 1) == -1);    // [2] - [0]
    REQUIRE(del1(1, 1) == 0);
    REQUIRE(del1(2, 1) == 1); 
    REQUIRE(del1(0, 2) == 0);     // [2] - [1]
    REQUIRE(del1(1, 2) == -1);
    REQUIRE(del1(2, 2) == 1);

    // Compare against the boundary homomorphism mod 2
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    
    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- // 
    cplex = complex_triangles_with_appendages();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [0, 1, 2] to [1, 2] - [0, 2] + [0, 1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 7
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 1
    REQUIRE(del2(0) == 1);     // [0, 1]
    REQUIRE(del2(1) == -1);    // [0, 2]
    REQUIRE(del2(2) == 0);     // [0, 3]
    REQUIRE(del2(3) == 1);     // [1, 2]
    REQUIRE(del2(4) == 0);     // [1, 4]
    REQUIRE(del2(5) == 0);     // [2, 5]
    REQUIRE(del2(6) == 0);     // [5, 6]

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send each [v0, v1] to [v1] - [v0] 
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 7 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 7
    Matrix<int, Dynamic, 2> edges(7, 2);  
    edges << 0, 1,    // Order edges lexicographically 
             0, 2,
             0, 3,
             1, 2,
             1, 4,
             2, 5,
             5, 6;
    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            // The (i, j)-th entry is the coefficient of the i-th 0-simplex
            // in the image of the j-th 1-simplex
            if (i == edges(j, 0))
                REQUIRE(del1(i, j) == -1); 
            else if (i == edges(j, 1))
                REQUIRE(del1(i, j) == 1);
            else 
                REQUIRE(del1(i, j) == 0);  
        }
    }

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 16
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 8
    edges.resize(16, 2);
    edges << 0, 1,    // Order edges lexicographically 
             0, 2,
             0, 7,
             1, 2,
             1, 3,
             1, 4,
             1, 7,
             1, 8,
             2, 4,
             2, 5,
             3, 4,
             3, 6,
             3, 8,
             4, 5,
             4, 6,
             7, 8;
    Matrix<int, Dynamic, 3> triangles(8, 3);
    triangles << 0, 1, 2,    // Order triangles lexicographically
                 0, 1, 7,
                 1, 2, 4,
                 1, 3, 4,
                 1, 3, 8,
                 1, 7, 8,
                 2, 4, 5,
                 3, 4, 6;
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            // The (i, j)-th entry is the coefficient of the i-th 1-simplex
            // in the image of the j-th 2-simplex
            int v0 = triangles(j, 0); 
            int v1 = triangles(j, 1); 
            int v2 = triangles(j, 2); 
            if (edges(i, 0) == v1 && edges(i, 1) == v2)
                REQUIRE(del2(i, j) == 1); 
            else if (edges(i, 0) == v0 && edges(i, 1) == v2)
                REQUIRE(del2(i, j) == -1);
            else if (edges(i, 0) == v0 && edges(i, 1) == v1)
                REQUIRE(del2(i, j) == 1); 
            else
                REQUIRE(del2(i, j) == 0);  
        }
    }

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send each [v0, v1] to [v1] - [v0] 
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 9 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 16
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            // The (i, j)-th entry is the coefficient of the i-th 0-simplex
            // in the image of the j-th 1-simplex
            if (i == edges(j, 0))
                REQUIRE(del1(i, j) == -1); 
            else if (i == edges(j, 1))
                REQUIRE(del1(i, j) == 1);
            else 
                REQUIRE(del1(i, j) == 0);  
        }
    }

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 15
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 6
    edges.resize(15, 2);
    edges << 0, 1,    // Order edges lexicographically 
             0, 2,
             0, 7,
             1, 2,
             1, 3,
             1, 7,
             1, 8,
             2, 4,
             2, 5,
             3, 4,
             3, 6,
             3, 8,
             4, 5,
             4, 6,
             7, 8;
    triangles.resize(6, 3);
    triangles << 0, 1, 2,    // Order triangles lexicographically
                 0, 1, 7,
                 1, 3, 8,
                 1, 7, 8,
                 2, 4, 5,
                 3, 4, 6;
    for (int i = 0; i < 15; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            // The (i, j)-th entry is the coefficient of the i-th 1-simplex
            // in the image of the j-th 2-simplex
            int v0 = triangles(j, 0); 
            int v1 = triangles(j, 1); 
            int v2 = triangles(j, 2); 
            if (edges(i, 0) == v1 && edges(i, 1) == v2)
                REQUIRE(del2(i, j) == 1); 
            else if (edges(i, 0) == v0 && edges(i, 1) == v2)
                REQUIRE(del2(i, j) == -1);
            else if (edges(i, 0) == v0 && edges(i, 1) == v1)
                REQUIRE(del2(i, j) == 1); 
            else
                REQUIRE(del2(i, j) == 0);  
        }
    }

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send each [v0, v1] to [v1] - [v0] 
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 9 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 15
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 15; ++j)
        {
            // The (i, j)-th entry is the coefficient of the i-th 0-simplex
            // in the image of the j-th 1-simplex
            if (i == edges(j, 0))
                REQUIRE(del1(i, j) == -1); 
            else if (i == edges(j, 1))
                REQUIRE(del1(i, j) == 1);
            else 
                REQUIRE(del1(i, j) == 0);  
        }
    }

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- // 
    cplex = complex_annulus(); 
    
    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 24
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 12
    for (int i = 0; i < del2.rows(); ++i)
    {
        for (int j = 0; j < del2.cols(); ++j)
        {
            if ((j == 0 && (i == 0 || i == 5)) ||
                (j == 1 && (i == 1 || i == 17)) || 
                (j == 2 && (i == 2 || i == 19)) ||
                (j == 3 && (i == 4 || i == 8)) ||
                (j == 4 && (i == 5 || i == 18)) ||
                (j == 5 && (i == 7 || i == 11)) ||
                (j == 6 && (i == 8 || i == 20)) ||
                (j == 7 && (i == 10 || i == 14)) ||
                (j == 8 && (i == 11 || i == 21)) ||
                (j == 9 && (i == 13 || i == 16)) || 
                (j == 10 && (i == 14 || i == 22)) ||
                (j == 11 && (i == 16 || i == 23))
            )
            {
                REQUIRE(del2(i, j) == 1); 
            }
            else if (
                (j == 0 && i == 2) || (j == 1 && i == 3) || (j == 2 && i == 3) ||
                (j == 3 && i == 6) || (j == 4 && i == 6) || (j == 5 && i == 9) ||
                (j == 6 && i == 9) || (j == 7 && i == 12) || (j == 8 && i == 12) ||
                (j == 9 && i == 15) || (j == 10 && i == 15) || (j == 11 && i == 17)
            )
            {
                REQUIRE(del2(i, j) == -1); 
            }
            else 
            {
                REQUIRE(del2(i, j) == 0); 
            }
        }
    }

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send each [v0, v1] to [v1] - [v0] 
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 12 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 24
    for (int i = 0; i < del1.rows(); ++i)
    {
        for (int j = 0; j < del1.cols(); ++j)
        {
            if ((j == 0 && i == 0) || (j == 1 && i == 0) || (j == 2 && i == 0) ||
                (j == 3 && i == 0) || (j == 4 && i == 1) || (j == 5 && i == 1) ||
                (j == 6 && i == 1) || (j == 7 && i == 2) || (j == 8 && i == 2) ||
                (j == 9 && i == 2) || (j == 10 && i == 3) || (j == 11 && i == 3) || 
                (j == 12 && i == 3) || (j == 13 && i == 4) || (j == 14 && i == 4) ||
                (j == 15 && i == 4) || (j == 16 && i == 5) || (j == 17 && i == 5) ||
                (j == 18 && i == 6) || (j == 19 && i == 6) || (j == 20 && i == 7) || 
                (j == 21 && i == 8) || (j == 22 && i == 9) || (j == 23 && i == 10)
            )
            {
                REQUIRE(del1(i, j) == -1); 
            }
            else if (
                (j == 0 && i == 1) || (j == 1 && i == 5) || (j == 2 && i == 6) ||
                (j == 3 && i == 11) || (j == 4 && i == 2) || (j == 5 && i == 6) ||
                (j == 6 && i == 7) || (j == 7 && i == 3) || (j == 8 && i == 7) ||
                (j == 9 && i == 8) || (j == 10 && i == 4) || (j == 11 && i == 8) || 
                (j == 12 && i == 9) || (j == 13 && i == 5) || (j == 14 && i == 9) ||
                (j == 15 && i == 10) || (j == 16 && i == 10) || (j == 17 && i == 11) ||
                (j == 18 && i == 7) || (j == 19 && i == 11) || (j == 20 && i == 8) || 
                (j == 21 && i == 9) || (j == 22 && i == 10) || (j == 23 && i == 11)
            )
            {
                REQUIRE(del1(i, j) == 1); 
            }
            else 
            {
                REQUIRE(del1(i, j) == 0); 
            }
        }
    }

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();

    // Get the boundary homomorphism from C_3 to C_2
    //
    // This map should send [0, 1, 2, 3] to [1, 2, 3] - [0, 2, 3] + [0, 1, 3] - [0, 1, 2]
    del3 = cplex.getBoundaryHomomorphism<0>(3); 
    REQUIRE(del3.rows() == cplex.getNumSimplices(2));    // 4
    REQUIRE(del3.cols() == cplex.getNumSimplices(3));    // 1
    REQUIRE(del3(0) == -1);    // [0, 1, 2]
    REQUIRE(del3(1) == 1);     // [0, 1, 3]
    REQUIRE(del3(2) == -1);    // [0, 2, 3]
    REQUIRE(del3(3) == 1);     // [1, 2, 3]

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 6
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 4
    REQUIRE(del2(0, 0) == 1);    // [0, 1, 2] -> [1, 2] - [0, 2] + [0, 1]
    REQUIRE(del2(1, 0) == -1); 
    REQUIRE(del2(2, 0) == 0); 
    REQUIRE(del2(3, 0) == 1); 
    REQUIRE(del2(4, 0) == 0); 
    REQUIRE(del2(5, 0) == 0);
    REQUIRE(del2(0, 1) == 1);    // [0, 1, 3] -> [1, 3] - [0, 3] + [0, 1]
    REQUIRE(del2(1, 1) == 0); 
    REQUIRE(del2(2, 1) == -1); 
    REQUIRE(del2(3, 1) == 0); 
    REQUIRE(del2(4, 1) == 1); 
    REQUIRE(del2(5, 1) == 0);
    REQUIRE(del2(0, 2) == 0);    // [0, 2, 3] -> [2, 3] - [0, 3] + [0, 2]
    REQUIRE(del2(1, 2) == 1); 
    REQUIRE(del2(2, 2) == -1); 
    REQUIRE(del2(3, 2) == 0); 
    REQUIRE(del2(4, 2) == 0); 
    REQUIRE(del2(5, 2) == 1);
    REQUIRE(del2(0, 3) == 0);    // [1, 2, 3] -> [2, 3] - [1, 3] + [1, 2]
    REQUIRE(del2(1, 3) == 0); 
    REQUIRE(del2(2, 3) == 0); 
    REQUIRE(del2(3, 3) == 1); 
    REQUIRE(del2(4, 3) == -1); 
    REQUIRE(del2(5, 3) == 1); 

    // Get the boundary homomorphism from C_1 to C_0
    //
    // Here, the map should send each [v0, v1] to [v1] - [v0] 
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 4 
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 6
    REQUIRE(del1(0, 0) == -1);    // [0, 1] -> [1] - [0]
    REQUIRE(del1(1, 0) == 1); 
    REQUIRE(del1(2, 0) == 0); 
    REQUIRE(del1(3, 0) == 0); 
    REQUIRE(del1(0, 1) == -1);    // [0, 2] -> [2] - [0] 
    REQUIRE(del1(1, 1) == 0);
    REQUIRE(del1(2, 1) == 1);
    REQUIRE(del1(3, 1) == 0); 
    REQUIRE(del1(0, 2) == -1);    // [0, 3] -> [3] - [0]
    REQUIRE(del1(1, 2) == 0); 
    REQUIRE(del1(2, 2) == 0); 
    REQUIRE(del1(3, 2) == 1);
    REQUIRE(del1(0, 3) == 0);     // [1, 2] -> [2] - [1]
    REQUIRE(del1(1, 3) == -1); 
    REQUIRE(del1(2, 3) == 1); 
    REQUIRE(del1(3, 3) == 0); 
    REQUIRE(del1(0, 4) == 0);     // [1, 3] -> [3] - [1]
    REQUIRE(del1(1, 4) == -1);
    REQUIRE(del1(2, 4) == 0);
    REQUIRE(del1(3, 4) == 1); 
    REQUIRE(del1(0, 5) == 0);     // [2, 3] -> [3] - [2]
    REQUIRE(del1(1, 5) == 0); 
    REQUIRE(del1(2, 5) == -1); 
    REQUIRE(del1(3, 5) == 1); 

    // Check that the composition of each pair of homomorphisms is zero
    REQUIRE(((del2 * del3).array() == 0).all()); 
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del3_p2 = cplex.getBoundaryHomomorphism<2>(3); 
    del2_p2 = cplex.getBoundaryHomomorphism<2>(2); 
    del1_p2 = cplex.getBoundaryHomomorphism<2>(1);
    REQUIRE(del3_p2.rows() == del3.rows()); 
    REQUIRE(del3_p2.cols() == del3.cols()); 
    for (int i = 0; i < del3_p2.rows(); ++i)
    {
        for (int j = 0; j < del3_p2.cols(); ++j)
        {
            if (del3(i, j) == -1 || del3(i, j) == 1)
                REQUIRE(del3_p2(i, j) == 1); 
            else    // del3(i, j) == 0
                REQUIRE(del3_p2(i, j) == 0); 
        }
    }
    REQUIRE(del2_p2.rows() == del2.rows()); 
    REQUIRE(del2_p2.cols() == del2.cols()); 
    for (int i = 0; i < del2_p2.rows(); ++i)
    {
        for (int j = 0; j < del2_p2.cols(); ++j)
        {
            if (del2(i, j) == -1 || del2(i, j) == 1)
                REQUIRE(del2_p2(i, j) == 1); 
            else    // del2(i, j) == 0
                REQUIRE(del2_p2(i, j) == 0); 
        }
    }
    REQUIRE(del1_p2.rows() == del1.rows()); 
    REQUIRE(del1_p2.cols() == del1.cols()); 
    for (int i = 0; i < del1_p2.rows(); ++i)
    {
        for (int j = 0; j < del1_p2.cols(); ++j)
        {
            if (del1(i, j) == -1 || del1(i, j) == 1)
                REQUIRE(del1_p2(i, j) == 1); 
            else    // del1(i, j) == 0
                REQUIRE(del1_p2(i, j) == 0); 
        }
    }
    REQUIRE(((del2_p2 * del3_p2).array() == 0).all()); 
    REQUIRE(((del1_p2 * del2_p2).array() == 0).all());
}

TEST_CASE(
    "Tests for combinatorial Laplacian calculations",
    "[SimplicialComplex3D::getCombinatorialLaplacian()]"
)
{
    Matrix<Fp<0>, Dynamic, Dynamic> L0, L1, L2, L3;

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // L0 should be the 3x3 zero matrix 
    L0 = cplex.getCombinatorialLaplacian(0);
    REQUIRE(L0.rows() == 3);
    REQUIRE(L0.cols() == 3); 
    REQUIRE((L0.array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 3); 
    REQUIRE(L0.cols() == 3);
    Matrix<Fp<0>, Dynamic, Dynamic> kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 3); 
    REQUIRE(L1.cols() == 3); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 1); 
    REQUIRE(L2.cols() == 1); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 

    // This complex has one 1-cycle
    REQUIRE(L0.rows() == 3); 
    REQUIRE(L0.cols() == 3);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 3); 
    REQUIRE(L1.cols() == 3); 
    Matrix<Fp<0>, Dynamic, Dynamic> kerL1 = ::kernel<Fp<0> >(L1); 
    REQUIRE(kerL1.cols() == 1); 

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 7); 
    REQUIRE(L0.cols() == 7);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 7); 
    REQUIRE(L1.cols() == 7); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 1); 
    REQUIRE(L2.cols() == 1); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 16); 
    REQUIRE(L1.cols() == 16); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 8); 
    REQUIRE(L2.cols() == 8); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is still connected but *not* simply connected, and has
    // one hole (1-D cycle)
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 15); 
    REQUIRE(L1.cols() == 15); 
    kerL1 = ::kernel<Fp<0> >(L1); 
    REQUIRE(kerL1.cols() == 1); 
    REQUIRE(L2.rows() == 6); 
    REQUIRE(L2.cols() == 6); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is still connected but *not* simply connected, and has
    // one hole (1-D cycle)
    REQUIRE(L0.rows() == 12); 
    REQUIRE(L0.cols() == 12);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 24); 
    REQUIRE(L1.cols() == 24); 
    kerL1 = ::kernel<Fp<0> >(L1); 
    REQUIRE(kerL1.cols() == 1); 
    REQUIRE(L2.rows() == 12); 
    REQUIRE(L2.cols() == 12); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1);
    L2 = cplex.getCombinatorialLaplacian(2);
    L3 = cplex.getCombinatorialLaplacian(3);

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 4); 
    REQUIRE(L0.cols() == 4);
    kerL0 = ::kernel<Fp<0> >(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 6); 
    REQUIRE(L1.cols() == 6); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 4); 
    REQUIRE(L2.cols() == 4); 
    REQUIRE(L2.determinant() != 0);
    REQUIRE(L3.rows() == 1); 
    REQUIRE(L3.cols() == 1); 
    REQUIRE(L3.determinant() != 0);
}

TEST_CASE(
    "Tests for homology calculations with rational coefficients",
    "[SimplicialComplex3D::getZeroCharHomology(),"
    " SimplicialComplex3D::getPrimeCharHomology(),"
    " SimplicialComplex3D::getZeroCharBettiNumbers()]"
)
{
    Matrix<Fp<0>, Dynamic, Dynamic> H0_p0, H1_p0, H2_p0, H3_p0;
    Matrix<Fp<0>, Dynamic, Dynamic> del1, del2;
    Matrix<Fp<0>, Dynamic, Dynamic> aug1, aug2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // Get a basis for the zeroth homology group over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    REQUIRE(H0_p0.rows() == 3);  

    // Check the Betti numbers 
    Array<int, Dynamic, 1> betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 3); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();

    // Get bases for the homology groups over the rationals
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    Matrix<Fp<0>, Dynamic, 1> v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Check that the basis vector for H1 is a scalar multiple of the vector
    // (1, -1, 1), corresponding to the cycle [v0,v1], [v2,v0], [v1,v2]
    //
    // This is the only possibility, as there are no 2-simplices and 
    // therefore the image of \del_2 is trivial
    REQUIRE(H1_p0(1) == -H1_p0(0)); 
    REQUIRE(H1_p0(2) == H1_p0(0)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Again check that the basis vector for H1 is a scalar multiple of the
    // vector (1, -1, 1)
    REQUIRE(H1_p0(1) == -H1_p0(0)); 
    REQUIRE(H1_p0(2) == H1_p0(0)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1); 
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- //
    cplex = complex_2d_mesh();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1)); 

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, 0, 0, -1, 1, 0, 0, -1, 0, 1, 0, ..., 0), corresponding to the
    // cycle [v1,v3], [v3,v4], [v4,v2], [v2,v1]
    Matrix<Fp<0>, Dynamic, 1> v2 = Matrix<Fp<0>, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(3) = -1; 
    v2(4) = 1; 
    v2(7) = -1; 
    v2(9) = 1;
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p0;  
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<0> >(aug2, v2)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1));

    // Again check that the basis vector for H1 is homologous to the 
    // above 1-cycle 
    aug2.col(del2.cols()) = H1_p0;
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<0> >(aug2, v2)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1));

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, ..., 0, 1, -1, 1, 1, 1, 1), corresponding to the cycle [v6,v7], 
    // [v7,v8], [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    v2 = Matrix<Fp<0>, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(18) = 1; 
    v2(19) = -1; 
    v2(20) = 1; 
    v2(21) = 1;
    v2(22) = 1;
    v2(23) = 1;
    del2 = cplex.getBoundaryHomomorphism<0>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p0;  
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<0> >(aug2, v2)); 

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1));

    // Again check that the basis vector for H1 is homologous to the 
    // above 1-cycle 
    aug2.col(del2.cols()) = H1_p0;
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<0> >(aug2, v2)); 
    
    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getZeroCharHomology(0);
    H1_p0 = cplex.getZeroCharHomology(1); 
    H2_p0 = cplex.getZeroCharHomology(2);
    H3_p0 = cplex.getZeroCharHomology(3); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);
    REQUIRE(H3_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<0>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<0>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1));

    // Get bases for the homology groups over the rationals, but with
    // getPrimeCharHomology()
    H0_p0 = cplex.getPrimeCharHomology<0>(0); 
    H1_p0 = cplex.getPrimeCharHomology<0>(1); 
    H2_p0 = cplex.getPrimeCharHomology<0>(2);
    H3_p0 = cplex.getPrimeCharHomology<0>(3);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);
    REQUIRE(H3_p0.cols() == 0);

    // Again check that the basis vector for H0 is homologous to the 
    // all-ones vector
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Fp<0> >(aug1, v1));

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0); 
}

TEST_CASE(
    "Tests for homology calculations with Z/2Z coefficients",
    "[SimplicialComplex3D::getPrimeCharHomology(),"
    " SimplicialComplex3D::getPrimeCharBettiNumbers()]"
)
{
    Matrix<Fp<2>, Dynamic, Dynamic> H0_p2, H1_p2, H2_p2, H3_p2;
    Matrix<Fp<2>, Dynamic, Dynamic> del1, del2;
    Matrix<Fp<2>, Dynamic, Dynamic> aug1, aug2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // Get a basis for the zeroth homology group over the rationals 
    H0_p2 = cplex.getPrimeCharHomology<2>(0);
    REQUIRE(H0_p2.rows() == 3);

    // Check the Betti numbers 
    Array<int, Dynamic, 1> betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 3); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0); 

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();
    
    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0); 
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    Matrix<Fp<2>, Dynamic, 1> v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);
    
    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0); 
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1)); 

    // Check that the basis vector for H1 is a scalar multiple of the vector
    // (1, 1, 1), corresponding to the cycle [v0,v1], [v2,v0], [v1,v2]
    //
    // Since we are in Z/2Z, this means that the basis vector must be equal
    // to (1, 1, 1) 
    //
    // This is the only possibility, as there are no 2-simplices and 
    // therefore the image of \del_2 is trivial
    REQUIRE(H1_p2(0) == 1); 
    REQUIRE(H1_p2(1) == 1); 
    REQUIRE(H1_p2(2) == 1); 

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0); 
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh();
    
    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0); 
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1)); 

    // Check the Betti numbers 
    betti = cplex.getZeroCharBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0); 
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1);
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1)); 

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, ..., 0), corresponding to the
    // cycle [v1,v3], [v3,v4], [v4,v2], [v2,v1]
    Matrix<Fp<2>, Dynamic, 1> v2 = Matrix<Fp<2>, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(3) = 1; 
    v2(4) = 1; 
    v2(7) = 1; 
    v2(9) = 1;
    del2 = cplex.getBoundaryHomomorphism<2>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p2;  
    REQUIRE(((del1 * H1_p2).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<2> >(aug2, v2)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0);
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1));

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, ..., 0, 1, 1, 1, 1, 1, 1), corresponding to the cycle [v6,v7], 
    // [v7,v8], [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    v2 = Matrix<Fp<2>, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(18) = 1; 
    v2(19) = 1; 
    v2(20) = 1; 
    v2(21) = 1;
    v2(22) = 1;
    v2(23) = 1;
    del2 = cplex.getBoundaryHomomorphism<2>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p2;  
    REQUIRE(((del1 * H1_p2).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Fp<2> >(aug2, v2));

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getPrimeCharHomology<2>(0);
    H1_p2 = cplex.getPrimeCharHomology<2>(1); 
    H2_p2 = cplex.getPrimeCharHomology<2>(2);
    H3_p2 = cplex.getPrimeCharHomology<2>(3);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);
    REQUIRE(H3_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Fp<2>, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getBoundaryHomomorphism<2>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Fp<2> >(aug1, v1));

    // Check the Betti numbers mod 2 
    betti = cplex.getPrimeCharBettiNumbers<2>(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);
}

TEST_CASE("Tests for minimal cycle calculations", "[getMinimalCycles()]")
{
    Matrix<double, Dynamic, Dynamic> opt_cycles; 

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_cycle();
    opt_cycles = cplex.getPrimeCharMinimalCycles<2>(1);
    std::cout << opt_cycles << "\n--\n";
    
    // ------------------------------------------------------------- // 
    // Test for annulus 
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 
    opt_cycles = cplex.getPrimeCharMinimalCycles<2>(1);
    std::cout << opt_cycles << "\n--\n";
}
