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
 * Return true if the given row is a row in the given array. 
 *
 * @param array Input array.
 * @param row Input row. 
 * @returns True if the given row is a row in the given array, false otherwise.
 */
template <typename T, int NCols>
bool rowInArray(const Ref<const Array<T, Dynamic, NCols> >& array, 
                const Ref<const Array<T, 1, NCols> >& row)
{
    for (int i = 0; i < array.rows(); ++i)
    {
        if ((array.row(i) == row).all())
            return true; 
    }
    return false; 
}

/**
 * Generate a collection of 0-simplices. 
 */
SimplicialComplex3D<T> complex1()
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
SimplicialComplex3D<T> complex2()
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
 * Generate a single 2-simplex with 3 additional 1-simplices.  
 */
SimplicialComplex3D<T> complex3()
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
SimplicialComplex3D<T> complex4()
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
SimplicialComplex3D<T> complex5()
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
 * Generate a single 3-simplex. 
 */
SimplicialComplex3D<T> complex6()
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
    // Test for complex 1
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex1(); 

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
    // Test for complex 2
    // ------------------------------------------------------------- // 
    cplex = complex2(); 

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
    // Test for complex 3
    // ------------------------------------------------------------- // 
    cplex = complex3(); 

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
    // Test for complex 4
    // ------------------------------------------------------------- // 
    cplex = complex4(); 

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
    // Test for complex 5
    // ------------------------------------------------------------- // 
    cplex = complex5(); 

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
    // Test for complex 6
    // ------------------------------------------------------------- // 
    cplex = complex6(); 

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
    // Test for complex 1
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex1();
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
    // Test for complex 2
    // ------------------------------------------------------------- // 
    cplex = complex2();
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
    // Test for complex 3
    // ------------------------------------------------------------- // 
    cplex = complex3();
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
    // Test for complex 4
    // ------------------------------------------------------------- // 
    cplex = complex4(); 
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
    // Test for complex 5
    // ------------------------------------------------------------- // 
    cplex = complex5();
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
    // Test for complex 6
    // ------------------------------------------------------------- // 
    cplex = complex6();
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
    // ------------------------------------------------------------- // 
    // Test for complex 1
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex1(); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism(0));
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism(1)); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism(2)); 
    REQUIRE_THROWS(cplex.getBoundaryHomomorphism(3));

    // ------------------------------------------------------------- // 
    // Test for complex 2
    // ------------------------------------------------------------- // 
    cplex = complex2();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [0, 1, 2] to [1, 2] - [0, 2] + [0, 1]
    Matrix<T, Dynamic, Dynamic> del2 = cplex.getBoundaryHomomorphism(2);
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
    Matrix<T, Dynamic, Dynamic> del1 = cplex.getBoundaryHomomorphism(1); 
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
    Matrix<T, Dynamic, 1> x(3), y(3);
    x << 4, 2, -3;
    y = del1 * x;
    REQUIRE(y(0) == -6); 
    REQUIRE(y(1) == 7); 
    REQUIRE(y(2) == -1);

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for complex 3
    // ------------------------------------------------------------- // 
    cplex = complex3();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [0, 1, 2] to [1, 2] - [0, 2] + [0, 1]
    del2 = cplex.getBoundaryHomomorphism(2);
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
    del1 = cplex.getBoundaryHomomorphism(1); 
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

    // ------------------------------------------------------------- // 
    // Test for complex 4
    // ------------------------------------------------------------- // 
    cplex = complex4();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism(2);
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
    del1 = cplex.getBoundaryHomomorphism(1); 
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

    // ------------------------------------------------------------- // 
    // Test for complex 5
    // ------------------------------------------------------------- // 
    cplex = complex5();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism(2);
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
    del1 = cplex.getBoundaryHomomorphism(1); 
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

    // ------------------------------------------------------------- // 
    // Test for complex 6
    // ------------------------------------------------------------- // 
    cplex = complex6();

    // Get the boundary homomorphism from C_3 to C_2
    //
    // This map should send [0, 1, 2, 3] to [1, 2, 3] - [0, 2, 3] + [0, 1, 3] - [0, 1, 2]
    Matrix<T, Dynamic, Dynamic> del3 = cplex.getBoundaryHomomorphism(3); 
    REQUIRE(del3.rows() == cplex.getNumSimplices(2));    // 4
    REQUIRE(del3.cols() == cplex.getNumSimplices(3));    // 1
    REQUIRE(del3(0) == -1);    // [0, 1, 2]
    REQUIRE(del3(1) == 1);     // [0, 1, 3]
    REQUIRE(del3(2) == -1);    // [0, 2, 3]
    REQUIRE(del3(3) == 1);     // [1, 2, 3]

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getBoundaryHomomorphism(2);
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
    del1 = cplex.getBoundaryHomomorphism(1); 
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
}

TEST_CASE(
    "Tests for combinatorial Laplacian calculations",
    "[SimplicialComplex3D::getCombinatorialLaplacian()]"
)
{
    Matrix<T, Dynamic, Dynamic> L0, L1, L2, L3;

    // ------------------------------------------------------------- // 
    // Test for complex 1
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex1();

    // L0 should be the 3x3 zero matrix 
    L0 = cplex.getCombinatorialLaplacian(0);
    REQUIRE(L0.rows() == 3);
    REQUIRE(L0.cols() == 3); 
    REQUIRE((L0.array() == 0).all());

    // Check the Betti numbers 
    Array<int, Dynamic, 1> betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 3); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  
    
    // ------------------------------------------------------------- // 
    // Test for complex 2
    // ------------------------------------------------------------- // 
    cplex = complex2();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 3); 
    REQUIRE(L0.cols() == 3);
    JacobiSVD<Matrix<T, Dynamic, Dynamic> > svd0(L0);
    Matrix<T, Dynamic, 1> singvals = svd0.singularValues(); 
    int n_zero_singvals = 0; 
    for (int i = 0; i < 3; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L1.rows() == 3); 
    REQUIRE(L1.cols() == 3); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 1); 
    REQUIRE(L2.cols() == 1); 
    REQUIRE(L2.determinant() != 0);

    // Check the Betti numbers 
    betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  

    // ------------------------------------------------------------- // 
    // Test for complex 3
    // ------------------------------------------------------------- // 
    cplex = complex3();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 7); 
    REQUIRE(L0.cols() == 7);
    svd0.compute(L0); 
    singvals = svd0.singularValues();
    n_zero_singvals = 0; 
    for (int i = 0; i < 7; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L1.rows() == 7); 
    REQUIRE(L1.cols() == 7); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 1); 
    REQUIRE(L2.cols() == 1); 
    REQUIRE(L2.determinant() != 0);

    // Check the Betti numbers 
    betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  

    // ------------------------------------------------------------- // 
    // Test for complex 4
    // ------------------------------------------------------------- // 
    cplex = complex4();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    svd0.compute(L0); 
    singvals = svd0.singularValues();
    n_zero_singvals = 0; 
    for (int i = 0; i < 9; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L1.rows() == 16); 
    REQUIRE(L1.cols() == 16); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 8); 
    REQUIRE(L2.cols() == 8); 
    REQUIRE(L2.determinant() != 0);

    // Check the Betti numbers 
    betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  

    // ------------------------------------------------------------- // 
    // Test for complex 5
    // ------------------------------------------------------------- // 
    cplex = complex5();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1); 
    L2 = cplex.getCombinatorialLaplacian(2); 

    // This complex is still connected but *not* simply connected, and has
    // one hole (1-D cycle)
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    svd0.compute(L0); 
    singvals = svd0.singularValues();
    n_zero_singvals = 0; 
    for (int i = 0; i < 9; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L1.rows() == 15); 
    REQUIRE(L1.cols() == 15); 
    JacobiSVD<Matrix<T, Dynamic, Dynamic> > svd1(L1); 
    singvals = svd1.singularValues();
    n_zero_singvals = 0; 
    for (int i = 0; i < 15; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L2.rows() == 6); 
    REQUIRE(L2.cols() == 6); 
    REQUIRE(L2.determinant() != 0);

    // Check the Betti numbers 
    betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  

    // ------------------------------------------------------------- // 
    // Test for complex 6
    // ------------------------------------------------------------- // 
    cplex = complex6();
    L0 = cplex.getCombinatorialLaplacian(0);
    L1 = cplex.getCombinatorialLaplacian(1);
    L2 = cplex.getCombinatorialLaplacian(2);
    L3 = cplex.getCombinatorialLaplacian(3);

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 4); 
    REQUIRE(L0.cols() == 4);
    svd0.compute(L0); 
    singvals = svd0.singularValues();
    n_zero_singvals = 0; 
    for (int i = 0; i < 4; ++i)
    {
        if (abs(singvals(i)) < 1e-8)
           n_zero_singvals++;  
    }
    REQUIRE(n_zero_singvals == 1); 
    REQUIRE(L1.rows() == 6); 
    REQUIRE(L1.cols() == 6); 
    REQUIRE(L1.determinant() != 0);
    REQUIRE(L2.rows() == 4); 
    REQUIRE(L2.cols() == 4); 
    REQUIRE(L2.determinant() != 0);
    REQUIRE(L3.rows() == 1); 
    REQUIRE(L3.cols() == 1); 
    REQUIRE(L3.determinant() != 0);

    // Check the Betti numbers 
    betti = cplex.getBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);  
}

/*
void testRepresentativeCycles(const Ref<const Array<T, Dynamic, 3> >& points, 
                              const Ref<const Array<int, Dynamic, 2> >& edges, 
                              const Ref<const Array<int, Dynamic, 3> >& triangles, 
                              const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
{
    SimplicialComplex3D<T> sc(points, edges, triangles, tetrahedra);
    sc.getRepresentativeCycles(); 
}
*/
