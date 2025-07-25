/**
 * Test module for the `SimplicialComplex3D` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/18/2025
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
typedef boost::multiprecision::mpq_rational Rational; 

using std::sin; 
using boost::multiprecision::sin;
using std::cos; 
using boost::multiprecision::cos; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::abs; 
using boost::multiprecision::abs;

/**
 * Return true if the two arrays are the same up to a permutation of the
 * columns.
 *
 * @param A First array. 
 * @param B Second array. 
 * @returns True if the two arrays are the same up to a permutation of the 
 *          columns, false otherwise.  
 */
bool matchArraysUpToPermutedCols(const Ref<const Matrix<Z2, Dynamic, Dynamic> >& A,
                                 const Ref<const Matrix<Z2, Dynamic, Dynamic> >& B)
{
    // Check that the arrays have the same shape 
    if (A.rows() != B.rows() || A.cols() != B.cols())
        return false; 

    // Check that each column in A is present in B
    Array<int, Dynamic, 1> found_cols = Array<int, Dynamic, 1>::Zero(B.cols()); 
    for (int i = 0; i < A.cols(); ++i)
    {
        int found_col = -1; 
        
        // Run over all columns of B that have not already been accounted for
        for (int j = 0; j < B.cols(); ++j)
        {
            if (!found_cols(j) && (A.array().col(i) == B.array().col(j)).all())
            {
                found_col = j; 
                break;
            }
        }

        // If the column in A does not exist in B (ignoring all columns of 
        // B that have already contributed a column in A), return false 
        if (found_col == -1)
            return false;

        // Keep track of the contributing column in B
        found_cols(found_col) = 1;  
    }

    return true; 
}

/**
 * Return true if there is a one-to-one correspondence between the columns 
 * of the two arrays in homology, i.e., for each column in A, there is 
 * exactly one column in B for which the corresponding cycles are homologous.
 *
 * @param cplex Input complex.
 * @param A First array. Each column corresponds to a cycle in the complex. 
 * @param B Second array. Each column corresponds to a cycle in the complex.
 * @param dim Dimension of the chains in the two arrays. 
 * @returns True if each cycle in A is homologous to exactly one cycle in B,
 *          false otherwise. 
 */
template <typename T>
bool matchCyclesUpToPermutedCols(const SimplicialComplex3D<T>& cplex, 
                                 const Ref<const Matrix<Z2, Dynamic, Dynamic> >& A,
                                 const Ref<const Matrix<Z2, Dynamic, Dynamic> >& B,
                                 const int dim)
{
    // Check that the two arrays have the correct number of rows 
    int n = cplex.getNumSimplices(dim); 
    if (A.rows() != n || B.rows() != n)
        throw std::runtime_error("Array shapes do not match input dimension"); 

    // Check that the arrays have the same number of columns 
    if (A.cols() != B.cols())
        return false;

    // Check that each column in A is homologous to *exactly* one column in B 
    Array<int, Dynamic, 1> found_cols = Array<int, Dynamic, 1>::Zero(B.cols()); 
    for (int i = 0; i < A.cols(); ++i)
    {
        // Index of column in B that is homologous to the i-th column in A
        int found_col = -1;
        
        // Run over all columns of B
        for (int j = 0; j < B.cols(); ++j)
        {
            // If there is another column in A that is already homologous to
            // the j-th column in B, the i-th column in A should not be 
            // homologous  
            if (found_cols(j))
            {
                if (cplex.areHomologousCycles(A.col(i), B.col(j), dim))
                    return false; 
            }
            // If no matching column in A has been found yet for the j-th 
            // column in B, check if the i-th column in A is homologous 
            else 
            {
                if (cplex.areHomologousCycles(A.col(i), B.col(j), dim))
                {
                    found_col = j;  
                    break;
                }
            }
        }

        // If there is no matching column of B for the i-th column of A, 
        // then return false 
        if (found_col == -1)
            return false;

        // Keep track of the homologous column in B
        found_cols(found_col) = 1; 

        // Check that the i-th column of A is not homologous to any other 
        // column in B
        for (int j = 0; j < B.cols(); ++j)
        {
            if (j != found_col && cplex.areHomologousCycles(A.col(i), B.col(j), dim))
                return false; 
        } 
    }

    return true; 
}

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
 * Generate a multicyclic graph. 
 */
SimplicialComplex3D<T> complex_multicycle()
{
    Array<T, Dynamic, 3> points(6, 3); 
    Array<int, Dynamic, 2> edges(9, 2); 
    Array<int, Dynamic, 3> triangles(0, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 0, 0, 0,
              1, 0, 0,
              2, 0, 0,
              0, 1, 0,
              1, 1, 0,
              2, 1, 0; 
    edges << 0, 1,
             0, 3,
             1, 2,
             1, 3,
             1, 4,
             1, 5,
             2, 5,
             3, 4,
             4, 5;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a graph of three disconnected cycles.
 */
SimplicialComplex3D<T> complex_disconnected_cycles()
{
    Array<T, Dynamic, 3> points(9, 3); 
    Array<int, Dynamic, 2> edges(9, 2); 
    Array<int, Dynamic, 3> triangles(0, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    T sin60 = sin(boost::math::constants::third_pi<T>()); 
    points <<   0,     0, 0,
                1,     0, 0,
              0.5, sin60, 0,
                2,     0, 0,
                3,     0, 0,
              2.5, sin60, 0,
                4,     0, 0,
                5,     0, 0,
              4.5, sin60, 0; 
    edges << 0, 1,
             0, 2,
             1, 2,
             3, 4,
             3, 5,
             4, 5,
             6, 7,
             6, 8,
             7, 8;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a single 2-simplex with 3 additional 1-simplices.  
 */
SimplicialComplex3D<T> complex_triangles_with_appendages()
{
    Array<T, Dynamic, 3> points(7, 3); 
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
 * Generate the example 2-D complex from Busaryev et al. 2010. 
 */
SimplicialComplex3D<T> complex_busaryev_example()
{
    Array<T, Dynamic, 3> points(5, 3); 
    Array<int, Dynamic, 2> edges(8, 2); 
    Array<int, Dynamic, 3> triangles(2, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4); 
    points << 0, 0, 0,
              1, 1, 0,
              2, 0, 0,
              0, 2, 0,
              2, 2, 0;
    edges << 0, 1,
             0, 2,
             0, 3,
             1, 2,
             1, 3,
             1, 4,
             2, 4,
             3, 4;
    triangles << 0, 1, 2,
                 1, 3, 4;
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
    Array<T, Dynamic, 3> points = Array<T, Dynamic, 3>::Zero(12, 3); 
    Array<int, Dynamic, 2> edges(24, 2); 
    Array<int, Dynamic, 3> triangles(12, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    const T r1 = 1.0; 
    const T r2 = 2.0; 
    for (int i = 0; i < 6; ++i)
    { 
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6.0) * boost::math::constants::two_pi<T>()
        ); 
        points(i, 0) = r2 * cos(theta); 
        points(i, 1) = r2 * sin(theta);
        points(6 + i, 0) = r1 * cos(theta); 
        points(6 + i, 1) = r1 * sin(theta);  
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
 * Generate a two-dimensional triangulation of a two-holed annulus. 
 */
SimplicialComplex3D<T> complex_annulus_two_holes()
{
    Array<T, Dynamic, 3> points = Array<T, Dynamic, 3>::Zero(22, 3); 
    Array<int, Dynamic, 2> edges(47, 2); 
    Array<int, Dynamic, 3> triangles(24, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    const T r1 = 1.0; 
    const T r2 = 2.0;
    const T c = 2 * r2 * cos(boost::math::constants::sixth_pi<T>());
    for (int i = 0; i < 6; ++i)     // Points for left annulus
    { 
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6.0) * boost::math::constants::two_pi<T>()
        ); 
        points(i, 0) = r2 * cos(theta); 
        points(i, 1) = r2 * sin(theta);
        points(6 + i, 0) = r1 * cos(theta); 
        points(6 + i, 1) = r1 * sin(theta);
    }
    points(12, 0) = c;              // Outer points for right annulus
    points(12, 1) = points(0, 1);
    points(13, 0) = c + r2 * cos(boost::math::constants::sixth_pi<T>());
    points(13, 1) = points(1, 1);
    points(14, 0) = c + r2 * cos(boost::math::constants::sixth_pi<T>());
    points(14, 1) = points(2, 1); 
    points(15, 0) = c; 
    points(15, 1) = points(3, 1);
    for (int i = 0; i < 6; ++i)     // Inner points for right annulus
    {
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6.0) * boost::math::constants::two_pi<T>()
        ); 
        points(16 + i, 0) = c + r1 * cos(theta); 
        points(16 + i, 1) = r1 * sin(theta);
    }
    edges <<  0,  1,
              0,  5,
              0,  6,
              0, 11,
              1,  2,   // 4
              1,  6, 
              1,  7, 
              1, 12,
              1, 20,
              1, 21,   // 9
              2,  3, 
              2,  7,
              2,  8,
              2, 15,
              2, 19,   // 14
              2, 20,
              3,  4, 
              3,  8,
              3,  9,
              4,  5,   // 19
              4,  9,
              4, 10,
              5, 10,
              5, 11,
              6,  7,   // 24
              6, 11,
              7,  8,
              8,  9,
              9, 10,
             10, 11,   // 29
             12, 13,
             12, 16,
             12, 21,
             13, 14,
             13, 16,   // 34
             13, 17,
             14, 15,
             14, 17,
             14, 18,
             15, 18,   // 39
             15, 19,
             16, 17, 
             16, 21,
             17, 18,
             18, 19,   // 44
             19, 20,
             20, 21;
    triangles <<  0,  1,  6,
                  0,  5, 11,
                  0,  6, 11,
                  1,  2,  7,
                  1,  2, 20,
                  1,  6,  7,
                  1, 12, 21,
                  1, 20, 21,
                  2,  3,  8,
                  2,  7,  8,
                  2, 15, 19,
                  2, 19, 20,
                  3,  4,  9,
                  3,  8,  9,
                  4,  5, 10,
                  4,  9, 10,
                  5, 10, 11,
                 12, 13, 16,
                 12, 16, 21,
                 13, 14, 17,
                 13, 16, 17,
                 14, 15, 18,
                 14, 17, 18,
                 15, 18, 19;
    return SimplicialComplex3D<T>(points, edges, triangles, tetrahedra); 
}

/**
 * Generate a two-dimensional triangulation of a disjoint union of two annuli.
 */
SimplicialComplex3D<T> complex_disjoint_annuli()
{
    Array<T, Dynamic, 3> points = Array<T, Dynamic, 3>::Zero(24, 3); 
    Array<int, Dynamic, 2> edges(48, 2); 
    Array<int, Dynamic, 3> triangles(24, 3); 
    Array<int, Dynamic, 4> tetrahedra(0, 4);
    const T r1 = 1.0; 
    const T r2 = 2.0;
    const T c = 5.0;
    for (int i = 0; i < 6; ++i)     // Points for left annulus
    { 
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6.0) * boost::math::constants::two_pi<T>()
        ); 
        points(i, 0) = r2 * cos(theta); 
        points(i, 1) = r2 * sin(theta);
        points(6 + i, 0) = r1 * cos(theta); 
        points(6 + i, 1) = r1 * sin(theta);
    }
    for (int i = 0; i < 6; ++i)     // Points for right annulus
    {
        T theta = (
            boost::math::constants::half_pi<T>() -
            (i / 6.0) * boost::math::constants::two_pi<T>()
        );
        points(12 + i, 0) = c + r2 * cos(theta); 
        points(12 + i, 1) = r2 * sin(theta);  
        points(18 + i, 0) = c + r1 * cos(theta); 
        points(18 + i, 1) = r1 * sin(theta);
    }
    edges <<  0,  1,
              0,  5,
              0,  6,
              0, 11,
              1,  2,   // 4
              1,  6, 
              1,  7, 
              2,  3, 
              2,  7,
              2,  8,   // 9
              3,  4, 
              3,  8,
              3,  9,
              4,  5,
              4,  9,   // 14
              4, 10,
              5, 10,
              5, 11,
              6,  7,
              6, 11,   // 19
              7,  8,
              8,  9,
              9, 10,
             10, 11,
             12, 13,   // 24
             12, 17,
             12, 18,
             12, 23,
             13, 14,
             13, 18,   // 29
             13, 19,
             14, 15,
             14, 19,
             14, 20,
             15, 16,   // 34
             15, 20,
             15, 21, 
             16, 17,
             16, 21,
             16, 22,   // 39
             17, 22,
             17, 23,
             18, 19,
             18, 23,
             19, 20,   // 44
             20, 21,
             21, 22,
             22, 23;
    triangles <<  0,  1,  6,
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
                  5, 10, 11,
                 12, 13, 18,
                 12, 17, 23,
                 12, 18, 23,
                 13, 14, 19,
                 13, 18, 19,
                 14, 15, 20,
                 14, 19, 20,
                 15, 16, 21,
                 15, 20, 21,
                 16, 17, 22,
                 16, 21, 22,
                 17, 22, 23;
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
    // Test for annulus
    // ------------------------------------------------------------- // 
    cplex = complex_annulus(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 12);  
    REQUIRE(cplex.getNumSimplices() == 12 + 24 + 12); 
    REQUIRE(cplex.getNumSimplices(0) == 12);
    REQUIRE(cplex.getNumSimplices(1) == 24);
    REQUIRE(cplex.getNumSimplices(2) == 12);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 12);
    for (int i = 0; i < 12; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 11
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 24);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 5)
    REQUIRE(simplices1(1, 1) == 5); 
    REQUIRE(simplices1(2, 0) == 0);     // (0, 6)
    REQUIRE(simplices1(2, 1) == 6);
    REQUIRE(simplices1(3, 0) == 0);     // (0, 11)
    REQUIRE(simplices1(3, 1) == 11);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(4, 1) == 2);
    REQUIRE(simplices1(5, 0) == 1);     // (1, 6)
    REQUIRE(simplices1(5, 1) == 6);
    REQUIRE(simplices1(6, 0) == 1);     // (1, 7)
    REQUIRE(simplices1(6, 1) == 7);
    REQUIRE(simplices1(7, 0) == 2);     // (2, 3)
    REQUIRE(simplices1(7, 1) == 3);
    REQUIRE(simplices1(8, 0) == 2);     // (2, 7)
    REQUIRE(simplices1(8, 1) == 7);
    REQUIRE(simplices1(9, 0) == 2);     // (2, 8)
    REQUIRE(simplices1(9, 1) == 8);
    REQUIRE(simplices1(10, 0) == 3);    // (3, 4)
    REQUIRE(simplices1(10, 1) == 4);
    REQUIRE(simplices1(11, 0) == 3);    // (3, 8)
    REQUIRE(simplices1(11, 1) == 8);
    REQUIRE(simplices1(12, 0) == 3);    // (3, 9)
    REQUIRE(simplices1(12, 1) == 9);
    REQUIRE(simplices1(13, 0) == 4);    // (4, 5)
    REQUIRE(simplices1(13, 1) == 5);
    REQUIRE(simplices1(14, 0) == 4);    // (4, 9)
    REQUIRE(simplices1(14, 1) == 9);
    REQUIRE(simplices1(15, 0) == 4);    // (4, 10)
    REQUIRE(simplices1(15, 1) == 10);
    REQUIRE(simplices1(16, 0) == 5);    // (5, 10)
    REQUIRE(simplices1(16, 1) == 10);
    REQUIRE(simplices1(17, 0) == 5);    // (5, 11)
    REQUIRE(simplices1(17, 1) == 11);
    REQUIRE(simplices1(18, 0) == 6);    // (6, 7)
    REQUIRE(simplices1(18, 1) == 7);
    REQUIRE(simplices1(19, 0) == 6);    // (6, 11)
    REQUIRE(simplices1(19, 1) == 11);
    REQUIRE(simplices1(20, 0) == 7);    // (7, 8)
    REQUIRE(simplices1(20, 1) == 8);
    REQUIRE(simplices1(21, 0) == 8);    // (8, 9)
    REQUIRE(simplices1(21, 1) == 9);
    REQUIRE(simplices1(22, 0) == 9);    // (9, 10)
    REQUIRE(simplices1(22, 1) == 10);
    REQUIRE(simplices1(23, 0) == 10);   // (10, 11)
    REQUIRE(simplices1(23, 1) == 11);

    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 12);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 6)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 6);
    REQUIRE(simplices2(1, 0) == 0);     // (0, 5, 11)
    REQUIRE(simplices2(1, 1) == 5);
    REQUIRE(simplices2(1, 2) == 11);
    REQUIRE(simplices2(2, 0) == 0);     // (0, 6, 11)
    REQUIRE(simplices2(2, 1) == 6);
    REQUIRE(simplices2(2, 2) == 11);
    REQUIRE(simplices2(3, 0) == 1);     // (1, 2, 7)
    REQUIRE(simplices2(3, 1) == 2);
    REQUIRE(simplices2(3, 2) == 7);
    REQUIRE(simplices2(4, 0) == 1);     // (1, 6, 7)
    REQUIRE(simplices2(4, 1) == 6);
    REQUIRE(simplices2(4, 2) == 7);
    REQUIRE(simplices2(5, 0) == 2);     // (2, 3, 8)
    REQUIRE(simplices2(5, 1) == 3);
    REQUIRE(simplices2(5, 2) == 8);
    REQUIRE(simplices2(6, 0) == 2);     // (2, 7, 8)
    REQUIRE(simplices2(6, 1) == 7);
    REQUIRE(simplices2(6, 2) == 8);
    REQUIRE(simplices2(7, 0) == 3);     // (3, 4, 9)
    REQUIRE(simplices2(7, 1) == 4);
    REQUIRE(simplices2(7, 2) == 9);
    REQUIRE(simplices2(8, 0) == 3);     // (3, 8, 9)
    REQUIRE(simplices2(8, 1) == 8);
    REQUIRE(simplices2(8, 2) == 9);
    REQUIRE(simplices2(9, 0) == 4);     // (4, 5, 10)
    REQUIRE(simplices2(9, 1) == 5);
    REQUIRE(simplices2(9, 2) == 10);
    REQUIRE(simplices2(10, 0) == 4);    // (4, 9, 10)
    REQUIRE(simplices2(10, 1) == 9);
    REQUIRE(simplices2(10, 2) == 10);
    REQUIRE(simplices2(11, 0) == 5);    // (5, 10, 11)
    REQUIRE(simplices2(11, 1) == 10);
    REQUIRE(simplices2(11, 2) == 11);

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus
    // ------------------------------------------------------------- // 
    cplex = complex_annulus_two_holes(); 

    // Check the dimension of the complex 
    REQUIRE(cplex.dimension() == 2);

    // Check the number of simplices in the complex
    REQUIRE(cplex.getNumPoints() == 22);  
    REQUIRE(cplex.getNumSimplices() == 22 + 47 + 24); 
    REQUIRE(cplex.getNumSimplices(0) == 22);
    REQUIRE(cplex.getNumSimplices(1) == 47);
    REQUIRE(cplex.getNumSimplices(2) == 24);
    REQUIRE_THROWS(cplex.getNumSimplices(3)); 

    // Get the simplices in the complex, for each dimension ... 
    simplices0 = cplex.getSimplices<0>();
    REQUIRE(simplices0.size() == 22);
    for (int i = 0; i < 22; ++i)
        REQUIRE(simplices0(i) == i);    // 0, ..., 21
    simplices1 = cplex.getSimplices<1>();
    REQUIRE(simplices1.rows() == 47);
    REQUIRE(simplices1(0, 0) == 0);     // (0, 1)
    REQUIRE(simplices1(0, 1) == 1); 
    REQUIRE(simplices1(1, 0) == 0);     // (0, 5)
    REQUIRE(simplices1(1, 1) == 5); 
    REQUIRE(simplices1(2, 0) == 0);     // (0, 6)
    REQUIRE(simplices1(2, 1) == 6);
    REQUIRE(simplices1(3, 0) == 0);     // (0, 11)
    REQUIRE(simplices1(3, 1) == 11);
    REQUIRE(simplices1(4, 0) == 1);     // (1, 2)
    REQUIRE(simplices1(4, 1) == 2);
    REQUIRE(simplices1(5, 0) == 1);     // (1, 6)
    REQUIRE(simplices1(5, 1) == 6);
    REQUIRE(simplices1(6, 0) == 1);     // (1, 7)
    REQUIRE(simplices1(6, 1) == 7);
    REQUIRE(simplices1(7, 0) == 1);     // (1, 12)
    REQUIRE(simplices1(7, 1) == 12);
    REQUIRE(simplices1(8, 0) == 1);     // (1, 20)
    REQUIRE(simplices1(8, 1) == 20);
    REQUIRE(simplices1(9, 0) == 1);     // (1, 21)
    REQUIRE(simplices1(9, 1) == 21);
    REQUIRE(simplices1(10, 0) == 2);    // (2, 3)
    REQUIRE(simplices1(10, 1) == 3);
    REQUIRE(simplices1(11, 0) == 2);    // (2, 7)
    REQUIRE(simplices1(11, 1) == 7);
    REQUIRE(simplices1(12, 0) == 2);    // (2, 8)
    REQUIRE(simplices1(12, 1) == 8);
    REQUIRE(simplices1(13, 0) == 2);    // (2, 15)
    REQUIRE(simplices1(13, 1) == 15);
    REQUIRE(simplices1(14, 0) == 2);    // (2, 19)
    REQUIRE(simplices1(14, 1) == 19);
    REQUIRE(simplices1(15, 0) == 2);    // (2, 20)
    REQUIRE(simplices1(15, 1) == 20);
    REQUIRE(simplices1(16, 0) == 3);    // (3, 4)
    REQUIRE(simplices1(16, 1) == 4);
    REQUIRE(simplices1(17, 0) == 3);    // (3, 8)
    REQUIRE(simplices1(17, 1) == 8);
    REQUIRE(simplices1(18, 0) == 3);    // (3, 9)
    REQUIRE(simplices1(18, 1) == 9);
    REQUIRE(simplices1(19, 0) == 4);    // (4, 5)
    REQUIRE(simplices1(19, 1) == 5);
    REQUIRE(simplices1(20, 0) == 4);    // (4, 9)
    REQUIRE(simplices1(20, 1) == 9);
    REQUIRE(simplices1(21, 0) == 4);    // (4, 10)
    REQUIRE(simplices1(21, 1) == 10);
    REQUIRE(simplices1(22, 0) == 5);    // (5, 10)
    REQUIRE(simplices1(22, 1) == 10);
    REQUIRE(simplices1(23, 0) == 5);    // (5, 11)
    REQUIRE(simplices1(23, 1) == 11);
    REQUIRE(simplices1(24, 0) == 6);    // (6, 7)
    REQUIRE(simplices1(24, 1) == 7);
    REQUIRE(simplices1(25, 0) == 6);    // (6, 11)
    REQUIRE(simplices1(25, 1) == 11);
    REQUIRE(simplices1(26, 0) == 7);    // (7, 8)
    REQUIRE(simplices1(26, 1) == 8);
    REQUIRE(simplices1(27, 0) == 8);    // (8, 9)
    REQUIRE(simplices1(27, 1) == 9);
    REQUIRE(simplices1(28, 0) == 9);    // (9, 10)
    REQUIRE(simplices1(28, 1) == 10);
    REQUIRE(simplices1(29, 0) == 10);   // (10, 11)
    REQUIRE(simplices1(29, 1) == 11);
    REQUIRE(simplices1(30, 0) == 12);   // (12, 13)
    REQUIRE(simplices1(30, 1) == 13);
    REQUIRE(simplices1(31, 0) == 12);   // (12, 16)
    REQUIRE(simplices1(31, 1) == 16);
    REQUIRE(simplices1(32, 0) == 12);   // (12, 21)
    REQUIRE(simplices1(32, 1) == 21);
    REQUIRE(simplices1(33, 0) == 13);   // (13, 14)
    REQUIRE(simplices1(33, 1) == 14);
    REQUIRE(simplices1(34, 0) == 13);   // (13, 16)
    REQUIRE(simplices1(34, 1) == 16);
    REQUIRE(simplices1(35, 0) == 13);   // (13, 17)
    REQUIRE(simplices1(35, 1) == 17);
    REQUIRE(simplices1(36, 0) == 14);   // (14, 15)
    REQUIRE(simplices1(36, 1) == 15);
    REQUIRE(simplices1(37, 0) == 14);   // (14, 17)
    REQUIRE(simplices1(37, 1) == 17);
    REQUIRE(simplices1(38, 0) == 14);   // (14, 18)
    REQUIRE(simplices1(38, 1) == 18);
    REQUIRE(simplices1(39, 0) == 15);   // (15, 18)
    REQUIRE(simplices1(39, 1) == 18);
    REQUIRE(simplices1(40, 0) == 15);   // (15, 19)
    REQUIRE(simplices1(40, 1) == 19);
    REQUIRE(simplices1(41, 0) == 16);   // (16, 17)
    REQUIRE(simplices1(41, 1) == 17);
    REQUIRE(simplices1(42, 0) == 16);   // (16, 21)
    REQUIRE(simplices1(42, 1) == 21);
    REQUIRE(simplices1(43, 0) == 17);   // (17, 18)
    REQUIRE(simplices1(43, 1) == 18);
    REQUIRE(simplices1(44, 0) == 18);   // (18, 19)
    REQUIRE(simplices1(44, 1) == 19);
    REQUIRE(simplices1(45, 0) == 19);   // (19, 20)
    REQUIRE(simplices1(45, 1) == 20);
    REQUIRE(simplices1(46, 0) == 20);   // (20, 21)
    REQUIRE(simplices1(46, 1) == 21);

    simplices2 = cplex.getSimplices<2>();
    REQUIRE(simplices2.rows() == 24);
    REQUIRE(simplices2(0, 0) == 0);     // (0, 1, 6)
    REQUIRE(simplices2(0, 1) == 1);
    REQUIRE(simplices2(0, 2) == 6);
    REQUIRE(simplices2(1, 0) == 0);     // (0, 5, 11)
    REQUIRE(simplices2(1, 1) == 5);
    REQUIRE(simplices2(1, 2) == 11);
    REQUIRE(simplices2(2, 0) == 0);     // (0, 6, 11)
    REQUIRE(simplices2(2, 1) == 6);
    REQUIRE(simplices2(2, 2) == 11);
    REQUIRE(simplices2(3, 0) == 1);     // (1, 2, 7)
    REQUIRE(simplices2(3, 1) == 2);
    REQUIRE(simplices2(3, 2) == 7);
    REQUIRE(simplices2(4, 0) == 1);     // (1, 2, 20)
    REQUIRE(simplices2(4, 1) == 2);
    REQUIRE(simplices2(4, 2) == 20);
    REQUIRE(simplices2(5, 0) == 1);     // (1, 6, 7)
    REQUIRE(simplices2(5, 1) == 6);
    REQUIRE(simplices2(5, 2) == 7);
    REQUIRE(simplices2(6, 0) == 1);     // (1, 12, 21)
    REQUIRE(simplices2(6, 1) == 12);
    REQUIRE(simplices2(6, 2) == 21);
    REQUIRE(simplices2(7, 0) == 1);     // (1, 20, 21)
    REQUIRE(simplices2(7, 1) == 20);
    REQUIRE(simplices2(7, 2) == 21);
    REQUIRE(simplices2(8, 0) == 2);     // (2, 3, 8)
    REQUIRE(simplices2(8, 1) == 3);
    REQUIRE(simplices2(8, 2) == 8);
    REQUIRE(simplices2(9, 0) == 2);     // (2, 7, 8)
    REQUIRE(simplices2(9, 1) == 7);
    REQUIRE(simplices2(9, 2) == 8);
    REQUIRE(simplices2(10, 0) == 2);    // (2, 15, 19)
    REQUIRE(simplices2(10, 1) == 15);
    REQUIRE(simplices2(10, 2) == 19);
    REQUIRE(simplices2(11, 0) == 2);    // (2, 19, 20)
    REQUIRE(simplices2(11, 1) == 19);
    REQUIRE(simplices2(11, 2) == 20);
    REQUIRE(simplices2(12, 0) == 3);    // (3, 4, 9)
    REQUIRE(simplices2(12, 1) == 4);
    REQUIRE(simplices2(12, 2) == 9);
    REQUIRE(simplices2(13, 0) == 3);    // (3, 8, 9)
    REQUIRE(simplices2(13, 1) == 8);
    REQUIRE(simplices2(13, 2) == 9);
    REQUIRE(simplices2(14, 0) == 4);    // (4, 5, 10)
    REQUIRE(simplices2(14, 1) == 5);
    REQUIRE(simplices2(14, 2) == 10);
    REQUIRE(simplices2(15, 0) == 4);    // (4, 9, 10)
    REQUIRE(simplices2(15, 1) == 9);
    REQUIRE(simplices2(15, 2) == 10);
    REQUIRE(simplices2(16, 0) == 5);    // (5, 10, 11)
    REQUIRE(simplices2(16, 1) == 10);
    REQUIRE(simplices2(16, 2) == 11);
    REQUIRE(simplices2(17, 0) == 12);   // (12, 13, 16)
    REQUIRE(simplices2(17, 1) == 13);
    REQUIRE(simplices2(17, 2) == 16);
    REQUIRE(simplices2(18, 0) == 12);   // (12, 16, 21)
    REQUIRE(simplices2(18, 1) == 16);
    REQUIRE(simplices2(18, 2) == 21);
    REQUIRE(simplices2(19, 0) == 13);   // (13, 14, 17)
    REQUIRE(simplices2(19, 1) == 14);
    REQUIRE(simplices2(19, 2) == 17);
    REQUIRE(simplices2(20, 0) == 13);   // (13, 16, 17)
    REQUIRE(simplices2(20, 1) == 16);
    REQUIRE(simplices2(20, 2) == 17);
    REQUIRE(simplices2(21, 0) == 14);   // (14, 15, 18)
    REQUIRE(simplices2(21, 1) == 15);
    REQUIRE(simplices2(21, 2) == 18);
    REQUIRE(simplices2(22, 0) == 14);   // (14, 17, 18)
    REQUIRE(simplices2(22, 1) == 17);
    REQUIRE(simplices2(22, 2) == 18);
    REQUIRE(simplices2(23, 0) == 15);   // (15, 18, 19)
    REQUIRE(simplices2(23, 1) == 18);
    REQUIRE(simplices2(23, 2) == 19);
    
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
    "[SimplicialComplex3D::getRealBoundaryHomomorphism(),"
    " SimplicialComplex3D::getZ2BoundaryHomomorphism()]"
)
{
    Matrix<Rational, Dynamic, Dynamic> del1, del2, del3; 
    Matrix<Z2, Dynamic, Dynamic> del1_p2, del2_p2, del3_p2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points(); 
    REQUIRE_THROWS(cplex.getRealBoundaryHomomorphism<Rational>(0));
    REQUIRE_THROWS(cplex.getRealBoundaryHomomorphism<Rational>(1)); 
    REQUIRE_THROWS(cplex.getRealBoundaryHomomorphism<Rational>(2)); 
    REQUIRE_THROWS(cplex.getRealBoundaryHomomorphism<Rational>(3));

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [0, 1, 2] to [1, 2] - [0, 2] + [0, 1]
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    Matrix<Rational, Dynamic, 1> x(3), y(3);
    x << 4, 2, -3;
    y = del1 * x;
    REQUIRE(y(0) == -6); 
    REQUIRE(y(1) == 7); 
    REQUIRE(y(2) == -1);

    // Check that the composition of the two homomorphisms is zero
    REQUIRE(((del1 * del2).array() == 0).all());

    // Compare against the boundary homomorphisms mod 2
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    // Test for two-holed annulus
    // ------------------------------------------------------------- // 
    cplex = complex_annulus_two_holes(); 
    
    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
    REQUIRE(del2.rows() == cplex.getNumSimplices(1));    // = 47
    REQUIRE(del2.cols() == cplex.getNumSimplices(2));    // = 24
    for (int i = 0; i < del2.rows(); ++i)
    {
        for (int j = 0; j < del2.cols(); ++j)
        {
            if ((j == 0 && (i == 0 || i == 5)) ||        // [0, 1, 6] 
                (j == 1 && (i == 1 || i == 23)) ||       // [0, 5, 11]
                (j == 2 && (i == 2 || i == 25)) ||       // [0, 6, 11]
                (j == 3 && (i == 4 || i == 11)) ||       // [1, 2, 7]
                (j == 4 && (i == 4 || i == 15)) ||       // [1, 2, 20]
                (j == 5 && (i == 5 || i == 24)) ||       // [1, 6, 7]
                (j == 6 && (i == 7 || i == 32)) ||       // [1, 12, 21]
                (j == 7 && (i == 8 || i == 46)) ||       // [1, 20, 21] 
                (j == 8 && (i == 10 || i == 17)) ||      // [2, 3, 8]
                (j == 9 && (i == 11 || i == 26)) ||      // [2, 7, 8] 
                (j == 10 && (i == 13 || i == 40)) ||     // [2, 15, 19]
                (j == 11 && (i == 14 || i == 45)) ||     // [2, 19, 20]
                (j == 12 && (i == 16 || i == 20)) ||     // [3, 4, 9]
                (j == 13 && (i == 17 || i == 27)) ||     // [3, 8, 9]
                (j == 14 && (i == 19 || i == 22)) ||     // [4, 5, 10]
                (j == 15 && (i == 20 || i == 28)) ||     // [4, 9, 10]
                (j == 16 && (i == 22 || i == 29)) ||     // [5, 10, 11]
                (j == 17 && (i == 30 || i == 34)) ||     // [12, 13, 16]
                (j == 18 && (i == 31 || i == 42)) ||     // [12, 16, 21]
                (j == 19 && (i == 33 || i == 37)) ||     // [13, 14, 17]
                (j == 20 && (i == 34 || i == 41)) ||     // [13, 16, 17]
                (j == 21 && (i == 36 || i == 39)) ||     // [14, 15, 18]
                (j == 22 && (i == 37 || i == 43)) ||     // [14, 17, 18]
                (j == 23 && (i == 39 || i == 44))        // [15, 18, 19]
            )
            {
                REQUIRE(del2(i, j) == 1); 
            }
            else if (
                (j == 0 && i == 2) ||      // [0, 1, 6] -> [0, 6]
                (j == 1 && i == 3) ||      // [0, 5, 11] -> [0, 11]
                (j == 2 && i == 3) ||      // [0, 6, 11] -> [0, 11]
                (j == 3 && i == 6) ||      // [1, 2, 7] -> [1, 7]
                (j == 4 && i == 8) ||      // [1, 2, 20] -> [1, 20]
                (j == 5 && i == 6) ||      // [1, 6, 7] -> [1, 7]
                (j == 6 && i == 9) ||      // [1, 12, 21] -> [1, 21]
                (j == 7 && i == 9) ||      // [1, 20, 21] -> [1, 21]
                (j == 8 && i == 12) ||     // [2, 3, 8] -> [2, 8]
                (j == 9 && i == 12) ||     // [2, 7, 8] -> [2, 8] 
                (j == 10 && i == 14) ||    // [2, 15, 19] -> [2, 19]
                (j == 11 && i == 15) ||    // [2, 19, 20] -> [2, 20]
                (j == 12 && i == 18) ||    // [3, 4, 9] -> [3, 9]
                (j == 13 && i == 18) ||    // [3, 8, 9] -> [3, 9]
                (j == 14 && i == 21) ||    // [4, 5, 10] -> [4, 10]
                (j == 15 && i == 21) ||    // [4, 9, 10] -> [4, 10]
                (j == 16 && i == 23) ||    // [5, 10, 11] -> [5, 11]
                (j == 17 && i == 31) ||    // [12, 13, 16] -> [12, 16]
                (j == 18 && i == 32) ||    // [12, 16, 21] -> [12, 21]
                (j == 19 && i == 35) ||    // [13, 14, 17] -> [13, 17]
                (j == 20 && i == 35) ||    // [13, 16, 17] -> [13, 17]
                (j == 21 && i == 38) ||    // [14, 15, 18] -> [14, 18]
                (j == 22 && i == 38) ||    // [14, 17, 18] -> [14, 18]
                (j == 23 && i == 40)       // [15, 18, 19] -> [15, 19]
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
    REQUIRE(del1.rows() == cplex.getNumSimplices(0));    // = 22
    REQUIRE(del1.cols() == cplex.getNumSimplices(1));    // = 47
    for (int i = 0; i < del1.rows(); ++i)
    {
        for (int j = 0; j < del1.cols(); ++j)
        {
            if ((j == 0 && i == 0) ||
                (j == 1 && i == 0) ||
                (j == 2 && i == 0) ||
                (j == 3 && i == 0) ||
                (j == 4 && i == 1) ||
                (j == 5 && i == 1) ||
                (j == 6 && i == 1) ||
                (j == 7 && i == 1) ||
                (j == 8 && i == 1) ||
                (j == 9 && i == 1) ||
                (j == 10 && i == 2) ||
                (j == 11 && i == 2) || 
                (j == 12 && i == 2) ||
                (j == 13 && i == 2) ||
                (j == 14 && i == 2) ||
                (j == 15 && i == 2) ||
                (j == 16 && i == 3) ||
                (j == 17 && i == 3) ||
                (j == 18 && i == 3) ||
                (j == 19 && i == 4) ||
                (j == 20 && i == 4) || 
                (j == 21 && i == 4) ||
                (j == 22 && i == 5) ||
                (j == 23 && i == 5) ||
                (j == 24 && i == 6) ||
                (j == 25 && i == 6) ||
                (j == 26 && i == 7) ||
                (j == 27 && i == 8) ||
                (j == 28 && i == 9) ||
                (j == 29 && i == 10) ||
                (j == 30 && i == 12) ||
                (j == 31 && i == 12) ||
                (j == 32 && i == 12) ||
                (j == 33 && i == 13) ||
                (j == 34 && i == 13) ||
                (j == 35 && i == 13) ||
                (j == 36 && i == 14) ||
                (j == 37 && i == 14) ||
                (j == 38 && i == 14) ||
                (j == 39 && i == 15) ||
                (j == 40 && i == 15) ||
                (j == 41 && i == 16) ||
                (j == 42 && i == 16) ||
                (j == 43 && i == 17) ||
                (j == 44 && i == 18) ||
                (j == 45 && i == 19) ||
                (j == 46 && i == 20)
            )
            {
                REQUIRE(del1(i, j) == -1); 
            }
            else if (
                (j == 0 && i == 1) ||
                (j == 1 && i == 5) ||
                (j == 2 && i == 6) ||
                (j == 3 && i == 11) ||
                (j == 4 && i == 2) ||
                (j == 5 && i == 6) ||
                (j == 6 && i == 7) ||
                (j == 7 && i == 12) ||
                (j == 8 && i == 20) ||
                (j == 9 && i == 21) ||
                (j == 10 && i == 3) ||
                (j == 11 && i == 7) || 
                (j == 12 && i == 8) ||
                (j == 13 && i == 15) ||
                (j == 14 && i == 19) ||
                (j == 15 && i == 20) ||
                (j == 16 && i == 4) ||
                (j == 17 && i == 8) ||
                (j == 18 && i == 9) ||
                (j == 19 && i == 5) ||
                (j == 20 && i == 9) || 
                (j == 21 && i == 10) ||
                (j == 22 && i == 10) ||
                (j == 23 && i == 11) ||
                (j == 24 && i == 7) ||
                (j == 25 && i == 11) ||
                (j == 26 && i == 8) ||
                (j == 27 && i == 9) ||
                (j == 28 && i == 10) ||
                (j == 29 && i == 11) ||
                (j == 30 && i == 13) ||
                (j == 31 && i == 16) ||
                (j == 32 && i == 21) ||
                (j == 33 && i == 14) ||
                (j == 34 && i == 16) ||
                (j == 35 && i == 17) ||
                (j == 36 && i == 15) ||
                (j == 37 && i == 17) ||
                (j == 38 && i == 18) ||
                (j == 39 && i == 18) ||
                (j == 40 && i == 19) ||
                (j == 41 && i == 17) ||
                (j == 42 && i == 21) ||
                (j == 43 && i == 18) ||
                (j == 44 && i == 19) ||
                (j == 45 && i == 20) ||
                (j == 46 && i == 21)
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
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    del3 = cplex.getRealBoundaryHomomorphism<Rational>(3); 
    REQUIRE(del3.rows() == cplex.getNumSimplices(2));    // 4
    REQUIRE(del3.cols() == cplex.getNumSimplices(3));    // 1
    REQUIRE(del3(0) == -1);    // [0, 1, 2]
    REQUIRE(del3(1) == 1);     // [0, 1, 3]
    REQUIRE(del3(2) == -1);    // [0, 2, 3]
    REQUIRE(del3(3) == 1);     // [1, 2, 3]

    // Get the boundary homomorphism from C_2 to C_1
    //
    // Here, the map should send [v0, v1, v2] to [v1, v2] - [v0, v2] + [v0, v1]
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
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
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
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
    del3_p2 = cplex.getZ2BoundaryHomomorphism(3); 
    del2_p2 = cplex.getZ2BoundaryHomomorphism(2); 
    del1_p2 = cplex.getZ2BoundaryHomomorphism(1);
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
    Matrix<Rational, Dynamic, Dynamic> L0, L1, L2, L3;

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // L0 should be the 3x3 zero matrix 
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    REQUIRE(L0.rows() == 3);
    REQUIRE(L0.cols() == 3); 
    REQUIRE((L0.array() == 0).all());

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 3); 
    REQUIRE(L0.cols() == 3);
    Matrix<Rational, Dynamic, Dynamic> kerL0 = ::kernel<Rational>(L0); 
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
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 

    // This complex has one 1-cycle
    REQUIRE(L0.rows() == 3); 
    REQUIRE(L0.cols() == 3);
    kerL0 = ::kernel<Rational>(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 3); 
    REQUIRE(L1.cols() == 3); 
    Matrix<Rational, Dynamic, Dynamic> kerL1 = ::kernel<Rational>(L1); 
    REQUIRE(kerL1.cols() == 1); 

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 7); 
    REQUIRE(L0.cols() == 7);
    kerL0 = ::kernel<Rational>(L0); 
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
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    kerL0 = ::kernel<Rational>(L0); 
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
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is still connected but *not* simply connected, and has
    // one hole (1-D cycle)
    REQUIRE(L0.rows() == 9); 
    REQUIRE(L0.cols() == 9);
    kerL0 = ::kernel<Rational>(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 15); 
    REQUIRE(L1.cols() == 15); 
    kerL1 = ::kernel<Rational>(L1); 
    REQUIRE(kerL1.cols() == 1); 
    REQUIRE(L2.rows() == 6); 
    REQUIRE(L2.cols() == 6); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is still connected but *not* simply connected, and has
    // one hole (1-D cycle)
    REQUIRE(L0.rows() == 12); 
    REQUIRE(L0.cols() == 12);
    kerL0 = ::kernel<Rational>(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 24); 
    REQUIRE(L1.cols() == 24); 
    kerL1 = ::kernel<Rational>(L1); 
    REQUIRE(kerL1.cols() == 1); 
    REQUIRE(L2.rows() == 12); 
    REQUIRE(L2.cols() == 12); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus_two_holes(); 
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1); 
    L2 = cplex.getCombinatorialLaplacian<Rational>(2); 

    // This complex is still connected but *not* simply connected, and has
    // two holes (1-D cycle)
    REQUIRE(L0.rows() == 22); 
    REQUIRE(L0.cols() == 22);
    kerL0 = ::kernel<Rational>(L0); 
    REQUIRE(kerL0.cols() == 1); 
    REQUIRE(L1.rows() == 47); 
    REQUIRE(L1.cols() == 47); 
    kerL1 = ::kernel<Rational>(L1); 
    REQUIRE(kerL1.cols() == 2); 
    REQUIRE(L2.rows() == 24); 
    REQUIRE(L2.cols() == 24); 
    REQUIRE(L2.determinant() != 0);

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();
    L0 = cplex.getCombinatorialLaplacian<Rational>(0);
    L1 = cplex.getCombinatorialLaplacian<Rational>(1);
    L2 = cplex.getCombinatorialLaplacian<Rational>(2);
    L3 = cplex.getCombinatorialLaplacian<Rational>(3);

    // This complex is contractible and therefore has trivial homology
    REQUIRE(L0.rows() == 4); 
    REQUIRE(L0.cols() == 4);
    kerL0 = ::kernel<Rational>(L0); 
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
    "[SimplicialComplex3D::getRealHomology(),"
    " SimplicialComplex3D::getRealBettiNumbers()]"
)
{
    Matrix<Rational, Dynamic, Dynamic> H0_p0, H1_p0, H2_p0, H3_p0;
    Matrix<Rational, Dynamic, Dynamic> del1, del2;
    Matrix<Rational, Dynamic, Dynamic> aug1, aug2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // Get a basis for the zeroth homology group over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    REQUIRE(H0_p0.rows() == 3);  

    // Check the Betti numbers 
    Array<int, Dynamic, 1> betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 3); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- // 
    cplex = complex_triangle();

    // Get bases for the homology groups over the rationals
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    Matrix<Rational, Dynamic, 1> v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check that the basis vector for H1 is a scalar multiple of the vector
    // (1, -1, 1), corresponding to the cycle [v0,v1], [v2,v0], [v1,v2]
    //
    // This is the only possibility, as there are no 2-simplices and 
    // therefore the image of \del_2 is trivial
    REQUIRE(H1_p0(1) == -H1_p0(0)); 
    REQUIRE(H1_p0(2) == H1_p0(0)); 

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1); 
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0; 
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- //
    cplex = complex_2d_mesh();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Rational>(aug1, v1)); 

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, 0, 0, -1, 1, 0, 0, -1, 0, 1, 0, ..., 0), corresponding to the
    // cycle [v1,v3], [v3,v4], [v4,v2], [v2,v1]
    Matrix<Rational, Dynamic, 1> v2 = Matrix<Rational, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(3) = -1; 
    v2(4) = 1; 
    v2(7) = -1; 
    v2(9) = 1;
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p0;  
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Rational>(aug2, v2)); 

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 1); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, ..., 0, 1, -1, 1, 1, 1, 1), corresponding to the cycle [v6,v7], 
    // [v7,v8], [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    v2 = Matrix<Rational, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(18) = 1; 
    v2(19) = -1; 
    v2(20) = 1; 
    v2(21) = 1;
    v2(22) = 1;
    v2(23) = 1;
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p0;  
    REQUIRE(((del1 * H1_p0).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Rational>(aug2, v2)); 

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus_two_holes(); 

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2);
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 2); 
    REQUIRE(H2_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check that the two basis vectors for H1 are not homologous to each other
    del2 = cplex.getRealBoundaryHomomorphism<Rational>(2);
    REQUIRE(((del1 * H1_p0.col(0)).array() == 0).all());
    REQUIRE(((del1 * H1_p0.col(1)).array() == 0).all());
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2;
    aug2.col(del2.cols()) = H1_p0.col(0);
    REQUIRE(containsInconsistency<Rational>(aug2, H1_p0.col(1))); 

    // Consider 11 different cycles ...  
    Matrix<Rational, Dynamic, Dynamic> cycles(cplex.getNumSimplices(1), 11);

    // Left hole, small: [v6,v7], [v7,v8], [v8,v9], [v9,v10], [v10,v11],
    //                   [v11,v6]
    cycles(24, 0) = 1;
    cycles(25, 0) = -1; 
    cycles(26, 0) = 1; 
    cycles(27, 0) = 1;
    cycles(28, 0) = 1;
    cycles(29, 0) = 1;
    // Left hole, large: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 1) = 1;
    cycles(1, 1) = -1;
    cycles(4, 1) = 1; 
    cycles(10, 1) = 1; 
    cycles(16, 1) = 1; 
    cycles(19, 1) = 1;
    // Right hole, small: [v16,v17], [v17,v18], [v18,v19], [v19,v20], [v20,v21],
    //                    [v21,v16]
    cycles(41, 2) = 1;
    cycles(42, 2) = -1;
    cycles(43, 2) = 1; 
    cycles(44, 2) = 1; 
    cycles(45, 2) = 1; 
    cycles(46, 2) = 1;
    // Right hole, large: [v12,v13], [v13,v14], [v14,v15], [v15,v2], [v2,v1],
    //                    [v1,v12]
    cycles(30, 3) = 1;
    cycles(33, 3) = 1; 
    cycles(36, 3) = 1; 
    cycles(13, 3) = -1; 
    cycles(4, 3) = -1; 
    cycles(7, 3) = 1; 
    // Both holes, small: [v6,v7], [v7,v1], [v1,v21], [v21,v16], [v16,v17],
    //                    [v17,v18], [v18,v19], [v19,v20], [v20,v2], [v2,v8],
    //                    [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    cycles(24, 4) = 1; 
    cycles(6, 4) = -1; 
    cycles(9, 4) = 1; 
    cycles(42, 4) = -1;
    cycles(41, 4) = 1;
    cycles(43, 4) = 1;
    cycles(44, 4) = 1;
    cycles(45, 4) = 1;
    cycles(15, 4) = -1; 
    cycles(12, 4) = 1;
    cycles(27, 4) = 1;
    cycles(28, 4) = 1;
    cycles(29, 4) = 1;
    cycles(25, 4) = -1;
    // Both holes, large: [v0,v1], [v1,v12], [v12,v13], [v13,v14], [v14,v15],
    //                    [v15,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 5) = 1; 
    cycles(7, 5) = 1; 
    cycles(30, 5) = 1; 
    cycles(33, 5) = 1;
    cycles(36, 5) = 1;
    cycles(13, 5) = -1;
    cycles(10, 5) = 1;
    cycles(16, 5) = 1;
    cycles(19, 5) = 1;
    cycles(1, 5) = -1;
    // Left hole, twice 1: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v0], [v0,v6], [v6,v1], [v1,v7],
    //                     [v7,v2], [v2,v8], [v8,v3], [v3,v9], [v9,v4],
    //                     [v4,v10], [v10,v5], [v5,v0]
    cycles(0, 6) = 1; 
    cycles(4, 6) = 1; 
    cycles(10, 6) = 1; 
    cycles(16, 6) = 1; 
    cycles(19, 6) = 1;
    cycles(23, 6) = 1; 
    cycles(3, 6) = -1; 
    cycles(2, 6) = 1;
    cycles(5, 6) = -1;
    cycles(6, 6) = 1;
    cycles(11, 6) = -1;
    cycles(12, 6) = 1; 
    cycles(17, 6) = -1;
    cycles(18, 6) = 1;
    cycles(20, 6) = -1;
    cycles(21, 6) = 1;
    cycles(22, 6) = -1; 
    cycles(1, 6) = -1;
    // Left hole, twice 2: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v6], [v6,v7], [v7,v8], [v8,v9],
    //                     [v9,v10], [v10,v11], [v11,v0]
    cycles(0, 7) = 1; 
    cycles(4, 7) = 1; 
    cycles(10, 7) = 1; 
    cycles(16, 7) = 1; 
    cycles(19, 7) = 1;
    cycles(23, 7) = 1;
    cycles(25, 7) = -1;
    cycles(24, 7) = 1;
    cycles(26, 7) = 1;
    cycles(27, 7) = 1;
    cycles(28, 7) = 1;
    cycles(29, 7) = 1;
    cycles(3, 7) = -1;
    // Right hole, twice 1: [v12,v13], [v13,v14], [v14,v15], [v15,v2],
    //                      [v2,v1], [v1,v21], [v21,v12], [v12,v16],
    //                      [v16,v13], [v13,v17], [v17,v14], [v14,v18],
    //                      [v18,v15], [v15,v19], [v19,v2], [v2,v20],
    //                      [v20,v1], [v1,v12]
    cycles(30, 8) = 1;
    cycles(33, 8) = 1; 
    cycles(36, 8) = 1;
    cycles(13, 8) = -1; 
    cycles(4, 8) = -1;
    cycles(9, 8) = 1;
    cycles(32, 8) = -1;
    cycles(31, 8) = 1;
    cycles(34, 8) = -1; 
    cycles(35, 8) = 1;
    cycles(37, 8) = -1;
    cycles(38, 8) = 1;
    cycles(39, 8) = -1;
    cycles(40, 8) = 1;
    cycles(14, 8) = -1;
    cycles(15, 8) = 1;
    cycles(8, 8) = -1; 
    cycles(7, 8) = 1;
    // Right hole, twice 2: [v12,v13], [v13,v14], [v14,v15], [v15,v2],
    //                      [v2,v1], [v1,v21], [v21,v16], [v16,v17],
    //                      [v17,v18], [v18,v19], [v19,v20], [v20,v21],
    //                      [v21,v12]
    cycles(30, 9) = 1;
    cycles(33, 9) = 1; 
    cycles(36, 9) = 1;
    cycles(13, 9) = -1; 
    cycles(4, 9) = -1;
    cycles(9, 9) = 1;
    cycles(42, 9) = -1;
    cycles(41, 9) = 1;
    cycles(43, 9) = 1; 
    cycles(44, 9) = 1;
    cycles(45, 9) = 1;
    cycles(46, 9) = 1;
    cycles(32, 9) = -1;
    // Figure eight: [v0,v1], [v1,v20], [v20,v19], [v19,v18], [v18,v17],
    //               [v17,v16], [v16,v21], [v21,v20], [v20,v2], [v2,v3],
    //               [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 10) = 1;
    cycles(8, 10) = 1;
    cycles(45, 10) = -1;
    cycles(44, 10) = -1;
    cycles(43, 10) = -1; 
    cycles(41, 10) = -1;
    cycles(42, 10) = 1;
    cycles(46, 10) = -1;
    cycles(15, 10) = -1;
    cycles(10, 10) = 1;
    cycles(16, 10) = 1;
    cycles(19, 10) = 1;
    cycles(1, 10) = -1;

    // Check that all 11 cycles are indeed cycles 
    for (int i = 0; i < 11; ++i)
        REQUIRE(((del1 * cycles.col(i)).array() == 0).all());
    
    // Check the homology of each cycle
    for (int i = 0; i < 11; ++i)
    {
        for (int j = i + 1; j < 11; ++j)
        {
            if (i == 0 && j == 1)          // Left holes 
                REQUIRE(!containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j)));
            else if (i == 2 && j == 3)     // Right holes
                REQUIRE(!containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j)));
            else if (i == 4 && j == 5)     // Both holes
                REQUIRE(!containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j)));
            else if (i == 6 && j == 7)     // Left hole, twice around
                REQUIRE(!containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j)));
            else if (i == 8 && j == 9)     // Right hole, twice around
                REQUIRE(!containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j)));
            else    // Figure eight should not be homologous to any other cycle
                REQUIRE(containsInconsistency<Rational>(del2, cycles.col(i) - cycles.col(j))); 
        }
    }

    // Check that each cycle belongs to a homology class that lies within 
    // the span of the two basis vectors
    aug2.resize(del2.rows(), del2.cols() + 2);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p0.col(0);
    aug2.col(del2.cols() + 1) = H1_p0.col(1);
    for (int i = 0; i < 11; ++i)
        REQUIRE(!containsInconsistency<Rational>(aug2, cycles.col(i)));

    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- // 
    cplex = complex_tetrahedron();

    // Get bases for the homology groups over the rationals 
    H0_p0 = cplex.getRealHomology<Rational>(0);
    H1_p0 = cplex.getRealHomology<Rational>(1); 
    H2_p0 = cplex.getRealHomology<Rational>(2);
    H3_p0 = cplex.getRealHomology<Rational>(3); 
    REQUIRE(H0_p0.cols() == 1);
    REQUIRE(H1_p0.cols() == 0); 
    REQUIRE(H2_p0.cols() == 0);
    REQUIRE(H3_p0.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Rational, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getRealBoundaryHomomorphism<Rational>(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p0;
    REQUIRE(!containsInconsistency<Rational>(aug1, v1));

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0); 
}

TEST_CASE(
    "Tests for homology calculations with Z/2Z coefficients",
    "[SimplicialComplex3D::getZ2Homology(),"
    " SimplicialComplex3D::getZ2BettiNumbers()]"
)
{
    Matrix<Z2, Dynamic, Dynamic> H0_p2, H1_p2, H2_p2, H3_p2;
    Matrix<Z2, Dynamic, Dynamic> del1, del2;
    Matrix<Z2, Dynamic, Dynamic> aug1, aug2; 

    // ------------------------------------------------------------- // 
    // Test for discrete set of points 
    // ------------------------------------------------------------- // 
    SimplicialComplex3D<T> cplex = complex_points();

    // Get a basis for the zeroth homology group over the rationals
    H0_p2 = cplex.getZ2Homology(0); 
    REQUIRE(H0_p2.rows() == 3);

    // Check the Betti numbers 
    Array<int, Dynamic, 1> betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 3); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0); 

    // ------------------------------------------------------------- // 
    // Test for triangle
    // ------------------------------------------------------------- //
    cplex = complex_triangle();
    
    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0); 
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    Matrix<Z2, Dynamic, 1> v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Z2>(aug1, v1)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);
    
    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0); 
    H1_p2 = cplex.getZ2Homology(1); 
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Z2>(aug1, v1)); 

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
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for triangle with additional 1-simplices 
    // ------------------------------------------------------------- //
    cplex = complex_triangles_with_appendages();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0); 
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Z2>(aug1, v1)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for simply connected 2-D mesh
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh();
    
    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0); 
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2; 
    REQUIRE(!containsInconsistency<Z2>(aug1, v1)); 

    // Check the Betti numbers 
    betti = cplex.getRealBettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0); 
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1);
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Z2>(aug1, v1)); 

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, ..., 0), corresponding to the
    // cycle [v1,v3], [v3,v4], [v4,v2], [v2,v1]
    Matrix<Z2, Dynamic, 1> v2 = Matrix<Z2, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(3) = 1; 
    v2(4) = 1; 
    v2(7) = 1; 
    v2(9) = 1;
    del2 = cplex.getZ2BoundaryHomomorphism(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p2;  
    REQUIRE(((del1 * H1_p2).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Z2>(aug2, v2)); 

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0);
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 1); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Z2>(aug1, v1));

    // Check that the basis vector for H1 is homologous to the cycle 
    // (0, ..., 0, 1, 1, 1, 1, 1, 1), corresponding to the cycle [v6,v7], 
    // [v7,v8], [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    v2 = Matrix<Z2, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(18) = 1; 
    v2(19) = 1; 
    v2(20) = 1; 
    v2(21) = 1;
    v2(22) = 1;
    v2(23) = 1;
    del2 = cplex.getZ2BoundaryHomomorphism(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2; 
    aug2.col(del2.cols()) = H1_p2;  
    REQUIRE(((del1 * H1_p2).array() == 0).all());
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(!containsInconsistency<Z2>(aug2, v2));

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 1); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus_two_holes();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0);
    H1_p2 = cplex.getZ2Homology(1);
    H2_p2 = cplex.getZ2Homology(2);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 2); 
    REQUIRE(H2_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Z2>(aug1, v1));

    // - Check that one basis vector for H1 is homologous to the cycle [v6,v7], 
    //   [v7,v8], [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    // - Check that one basis vector for H1 is homologous to the cycle [v16,v17],
    //   [v17,v18], [v18,v19], [v19,v20], [v20,v21], [v21,v16]
    v2 = Matrix<Z2, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v2(24) = 1; 
    v2(25) = 1; 
    v2(26) = 1; 
    v2(27) = 1;
    v2(28) = 1;
    v2(29) = 1;
    Matrix<Z2, Dynamic, 1> v3 = Matrix<Z2, Dynamic, 1>::Zero(cplex.getNumSimplices(1));
    v3(41) = 1; 
    v3(42) = 1;
    v3(43) = 1; 
    v3(44) = 1; 
    v3(45) = 1; 
    v3(46) = 1;
    REQUIRE(((del1 * v2).array() == 0).all());
    REQUIRE(((del1 * v3).array() == 0).all());
    REQUIRE(((del1 * H1_p2.col(0)).array() == 0).all());
    REQUIRE(((del1 * H1_p2.col(1)).array() == 0).all());
    del2 = cplex.getZ2BoundaryHomomorphism(2);
    aug2.resize(del2.rows(), del2.cols() + 1);
    aug2(Eigen::all, Eigen::seq(0, del2.cols() - 1)) = del2;
    aug2.col(del2.cols()) = H1_p2.col(0);     // Check first basis vector 
    REQUIRE((!containsInconsistency<Z2>(aug2, v2) ^ !containsInconsistency<Z2>(aug2, v3)));
    if (!containsInconsistency<Z2>(aug2, v2))
    {
        // If the first basis vector is homologous to v2 ... 
        aug2.col(del2.cols()) = H1_p2.col(1); 
        REQUIRE(!containsInconsistency<Z2>(aug2, v3));
    }
    else
    {
        // If the first basis vector is homologous to v3 ... 
        aug2.col(del2.cols()) = H1_p2.col(1); 
        REQUIRE(!containsInconsistency<Z2>(aug2, v2));
    }

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 2); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);
    
    // ------------------------------------------------------------- // 
    // Test for tetrahedron
    // ------------------------------------------------------------- //
    cplex = complex_tetrahedron();

    // Get bases for the homology groups over Z/2Z
    H0_p2 = cplex.getZ2Homology(0);
    H1_p2 = cplex.getZ2Homology(1); 
    H2_p2 = cplex.getZ2Homology(2);
    H3_p2 = cplex.getZ2Homology(3);
    REQUIRE(H0_p2.cols() == 1);
    REQUIRE(H1_p2.cols() == 0); 
    REQUIRE(H2_p2.cols() == 0);
    REQUIRE(H3_p2.cols() == 0);

    // Check that the basis vector for H0 is homologous to the all-ones
    // vector
    //
    // Since \del_0 is the zero map, this means that the difference lies
    // in the image of \del_1
    v1 = Matrix<Z2, Dynamic, 1>::Ones(cplex.getNumSimplices(0));
    del1 = cplex.getZ2BoundaryHomomorphism(1);
    aug1.resize(del1.rows(), del1.cols() + 1); 
    aug1(Eigen::all, Eigen::seq(0, del1.cols() - 1)) = del1; 
    aug1.col(del1.cols()) = H0_p2;
    REQUIRE(!containsInconsistency<Z2>(aug1, v1));

    // Check the Betti numbers mod 2 
    betti = cplex.getZ2BettiNumbers(); 
    REQUIRE(betti(0) == 1); 
    REQUIRE(betti(1) == 0); 
    REQUIRE(betti(2) == 0);
    REQUIRE(betti(3) == 0);
}

TEST_CASE("Tests for sentinel cycle calculation", "[getSentinelCycles()]")
{
    SimplicialComplex3D<T> cplex;
    Matrix<int, Dynamic, 2> edges, edges_reordered;  
    Matrix<Z2, Dynamic, Dynamic> sentinel_cycles;
    Matrix<int, Dynamic, 1> vertices_in_component; 
    Matrix<int, Dynamic, 1> edges_in_component; 
    std::unordered_map<std::pair<int, int>, int, boost::hash<std::pair<int, int> > > edge_map;
    std::vector<std::vector<int> > tree_paths; 
    Graph tree;

    // ------------------------------------------------------------- // 
    // Test for Busaryev et al.'s example 
    // ------------------------------------------------------------- //
    cplex = complex_busaryev_example();
    int nv = cplex.getNumSimplices(0); 
    int ne = cplex.getNumSimplices(1);
    edges = cplex.getSimplices<1>(); 
    vertices_in_component = Matrix<int, Dynamic, 1>::Ones(nv); 
    edges_in_component = Matrix<int, Dynamic, 1>::Ones(ne); 
    edge_map = cplex.getEdgeOrdering();
    auto result = cplex.getSentinelCycles(
        0, vertices_in_component, edges_in_component, edge_map
    );
    edges_reordered = result.first; 
    sentinel_cycles = result.second;
    REQUIRE(edges_reordered.rows() == ne); 
    REQUIRE(edges_reordered.cols() == 2); 
    REQUIRE(sentinel_cycles.rows() == ne); 
    REQUIRE(sentinel_cycles.cols() == ne - nv + 1);  

    // To check the sentinel cycles, we need the spanning tree rooted at 0
    auto result2 = cplex.getMinimumWeightPathTree(0);
    tree_paths = result2.first;
    tree = result2.second;

    // Check that the sentinel edges are not in the tree 
    int ne_tree = boost::num_edges(tree); 
    REQUIRE(ne_tree == nv - 1);
    for (int i = 0; i < ne - nv + 1; ++i)
        REQUIRE(!boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);
    for (int i = ne - nv + 1; i < ne; ++i)
        REQUIRE(boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);

    // Check that each sentinel cycle contains the corresponding sentinel 
    // edge, plus the shortest path between the edge's endpoints in the tree
    for (int j = 0; j < ne - nv + 1; ++j)
    {
        // Get the j-th sentinel edge
        std::pair<int, int> sentinel_edge = std::make_pair(
            edges_reordered(j, 0), edges_reordered(j, 1)
        );
        
        // Keep track of the non-sentinel edges in the j-th sentinel cycle 
        std::vector<std::pair<int, int> > path;

        // Run over the edges in the 1-skeleton ...  
        for (int i = 0; i < ne; ++i)
        {
            // Is the i-th edge in the complex in the j-th sentinel cycle?
            if (sentinel_cycles(i, j) == 1)
            {
                // If so, the edge should be the j-th sentinel edge or 
                // an edge in the tree
                std::pair<int, int> edge = std::make_pair(edges(i, 0), edges(i, 1));  
                REQUIRE((
                    edge == sentinel_edge || boost::edge(edges(i, 0), edges(i, 1), tree).second
                ));
                if (edge != sentinel_edge)
                    path.push_back(edge);  
            } 
        }
       
        // Does the non-sentinel path form a simple path between the
        // endpoints of the j-th sentinel edge?
        int u = sentinel_edge.first; 
        int v = sentinel_edge.second;
        int curr = u;
        std::unordered_set<int> visited;
        int n_traversed = 0; 
        while (curr != v)
        {
            // Mark the current vertex as visited 
            visited.insert(curr);

            // Find the first edge incident to the current vertex for which
            // the other endpoint has not been visited
            bool found_next = false;  
            for (int i = 0; i < path.size(); ++i)
            {
                int a = path[i].first; 
                int b = path[i].second; 
                if (a == curr && visited.find(b) == visited.end())
                {
                    curr = b;
                    found_next = true; 
                    break;  
                }
                else if (b == curr && visited.find(a) == visited.end())
                {
                    curr = a; 
                    found_next = true; 
                    break; 
                }
            }
            n_traversed++; 
            REQUIRE(found_next); 
        }

        // Check that we have ended up at v 
        REQUIRE(curr == v);

        // Check that each edge was traversed and each vertex was visited 
        REQUIRE(n_traversed == path.size());  
        REQUIRE(visited.size() == path.size());   // v was not added to the set
    }

    // ------------------------------------------------------------- // 
    // Test for annulus 
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 
    nv = cplex.getNumSimplices(0); 
    ne = cplex.getNumSimplices(1);
    edges = cplex.getSimplices<1>(); 
    vertices_in_component = Matrix<int, Dynamic, 1>::Ones(nv); 
    edges_in_component = Matrix<int, Dynamic, 1>::Ones(ne); 
    edge_map = cplex.getEdgeOrdering();
    result = cplex.getSentinelCycles(
        0, vertices_in_component, edges_in_component, edge_map
    );
    edges_reordered = result.first; 
    sentinel_cycles = result.second;
    REQUIRE(edges_reordered.rows() == ne); 
    REQUIRE(edges_reordered.cols() == 2); 
    REQUIRE(sentinel_cycles.rows() == ne); 
    REQUIRE(sentinel_cycles.cols() == ne - nv + 1);  

    // To check the sentinel cycles, we need the spanning tree rooted at 0
    result2 = cplex.getMinimumWeightPathTree(0);
    tree_paths = result2.first;
    tree = result2.second;

    // Check that the sentinel edges are not in the tree 
    ne_tree = boost::num_edges(tree); 
    REQUIRE(ne_tree == nv - 1);
    for (int i = 0; i < ne - nv + 1; ++i)
        REQUIRE(!boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);
    for (int i = ne - nv + 1; i < ne; ++i)
        REQUIRE(boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);

    // Check that each sentinel cycle contains the corresponding sentinel 
    // edge, plus the shortest path between the edge's endpoints in the tree
    for (int j = 0; j < ne - nv + 1; ++j)
    {
        // Get the j-th sentinel edge
        std::pair<int, int> sentinel_edge = std::make_pair(
            edges_reordered(j, 0), edges_reordered(j, 1)
        );
        
        // Keep track of the non-sentinel edges in the j-th sentinel cycle 
        std::vector<std::pair<int, int> > path;

        // Run over the edges in the 1-skeleton ...  
        for (int i = 0; i < ne; ++i)
        {
            // Is the i-th edge in the complex in the j-th sentinel cycle?
            if (sentinel_cycles(i, j) == 1)
            {
                // If so, the edge should be the j-th sentinel edge or 
                // an edge in the tree
                std::pair<int, int> edge = std::make_pair(edges(i, 0), edges(i, 1));  
                REQUIRE((
                    edge == sentinel_edge || boost::edge(edges(i, 0), edges(i, 1), tree).second
                ));
                if (edge != sentinel_edge)
                    path.push_back(edge);  
            } 
        }
       
        // Does the non-sentinel path form a simple path between the
        // endpoints of the j-th sentinel edge?
        int u = sentinel_edge.first; 
        int v = sentinel_edge.second;
        int curr = u;
        std::unordered_set<int> visited;
        int n_traversed = 0; 
        while (curr != v)
        {
            // Mark the current vertex as visited 
            visited.insert(curr);

            // Find the first edge incident to the current vertex for which
            // the other endpoint has not been visited
            bool found_next = false;  
            for (int i = 0; i < path.size(); ++i)
            {
                int a = path[i].first; 
                int b = path[i].second; 
                if (a == curr && visited.find(b) == visited.end())
                {
                    curr = b;
                    found_next = true; 
                    break;  
                }
                else if (b == curr && visited.find(a) == visited.end())
                {
                    curr = a; 
                    found_next = true; 
                    break; 
                }
            }
            n_traversed++; 
            REQUIRE(found_next); 
        }

        // Check that we have ended up at v 
        REQUIRE(curr == v);

        // Check that each edge was traversed and each vertex was visited 
        REQUIRE(n_traversed == path.size());  
        REQUIRE(visited.size() == path.size());   // v was not added to the set
    }

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus 
    // ------------------------------------------------------------- //
    cplex = complex_annulus_two_holes(); 
    nv = cplex.getNumSimplices(0); 
    ne = cplex.getNumSimplices(1);
    edges = cplex.getSimplices<1>(); 
    vertices_in_component = Matrix<int, Dynamic, 1>::Ones(nv); 
    edges_in_component = Matrix<int, Dynamic, 1>::Ones(ne); 
    edge_map = cplex.getEdgeOrdering();
    result = cplex.getSentinelCycles(
        0, vertices_in_component, edges_in_component, edge_map
    );
    edges_reordered = result.first; 
    sentinel_cycles = result.second;
    REQUIRE(edges_reordered.rows() == ne); 
    REQUIRE(edges_reordered.cols() == 2); 
    REQUIRE(sentinel_cycles.rows() == ne); 
    REQUIRE(sentinel_cycles.cols() == ne - nv + 1);  

    // To check the sentinel cycles, we need the spanning tree rooted at 0
    result2 = cplex.getMinimumWeightPathTree(0);
    tree_paths = result2.first;
    tree = result2.second;

    // Check that the sentinel edges are not in the tree 
    ne_tree = boost::num_edges(tree); 
    REQUIRE(ne_tree == nv - 1);
    for (int i = 0; i < ne - nv + 1; ++i)
        REQUIRE(!boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);
    for (int i = ne - nv + 1; i < ne; ++i)
        REQUIRE(boost::edge(edges_reordered(i, 0), edges_reordered(i, 1), tree).second);

    // Check that each sentinel cycle contains the corresponding sentinel 
    // edge, plus the shortest path between the edge's endpoints in the tree
    for (int j = 0; j < ne - nv + 1; ++j)
    {
        // Get the j-th sentinel edge
        std::pair<int, int> sentinel_edge = std::make_pair(
            edges_reordered(j, 0), edges_reordered(j, 1)
        );
        
        // Keep track of the non-sentinel edges in the j-th sentinel cycle 
        std::vector<std::pair<int, int> > path;

        // Run over the edges in the 1-skeleton ...  
        for (int i = 0; i < ne; ++i)
        {
            // Is the i-th edge in the complex in the j-th sentinel cycle?
            if (sentinel_cycles(i, j) == 1)
            {
                // If so, the edge should be the j-th sentinel edge or 
                // an edge in the tree
                std::pair<int, int> edge = std::make_pair(edges(i, 0), edges(i, 1));  
                REQUIRE((
                    edge == sentinel_edge || boost::edge(edges(i, 0), edges(i, 1), tree).second
                ));
                if (edge != sentinel_edge)
                    path.push_back(edge);  
            } 
        }
       
        // Does the non-sentinel path form a simple path between the
        // endpoints of the j-th sentinel edge?
        int u = sentinel_edge.first; 
        int v = sentinel_edge.second;
        int curr = u;
        std::unordered_set<int> visited;
        int n_traversed = 0; 
        while (curr != v)
        {
            // Mark the current vertex as visited 
            visited.insert(curr);

            // Find the first edge incident to the current vertex for which
            // the other endpoint has not been visited
            bool found_next = false;  
            for (int i = 0; i < path.size(); ++i)
            {
                int a = path[i].first; 
                int b = path[i].second; 
                if (a == curr && visited.find(b) == visited.end())
                {
                    curr = b;
                    found_next = true; 
                    break;  
                }
                else if (b == curr && visited.find(a) == visited.end())
                {
                    curr = a; 
                    found_next = true; 
                    break; 
                }
            }
            n_traversed++; 
            REQUIRE(found_next); 
        }

        // Check that we have ended up at v 
        REQUIRE(curr == v);

        // Check that each edge was traversed and each vertex was visited 
        REQUIRE(n_traversed == path.size());  
        REQUIRE(visited.size() == path.size());   // v was not added to the set
    } 
}

TEST_CASE("Tests for minimal homology basis calculation", "[getMinimalFirstHomology()]")
{
    SimplicialComplex3D<T> cplex; 
    Matrix<Z2, Dynamic, Dynamic> min_basis;
    Matrix<Z2, Dynamic, Dynamic> cycles;

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 1);

    // Check that the minimal basis cycle is precisely the one cycle in 
    // the complex
    REQUIRE(min_basis(0) == 1); 
    REQUIRE(min_basis(1) == 1); 
    REQUIRE(min_basis(2) == 1);

    // ------------------------------------------------------------- // 
    // Test for multicyclic graph 
    // ------------------------------------------------------------- // 
    cplex = complex_multicycle();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 4);

    // Check that the minimal basis is precisely the minimal cycle basis
    cycles.resize(9, 4); 
    cycles << 1, 0, 0, 0,
              1, 0, 0, 0,
              0, 0, 0, 1,
              1, 1, 0, 0,
              0, 1, 1, 0,
              0, 0, 1, 1,
              0, 0, 0, 1,
              0, 1, 0, 0,
              0, 0, 1, 0;
    REQUIRE(matchCyclesUpToPermutedCols<T>(cplex, min_basis, cycles, 1));
    REQUIRE(matchArraysUpToPermutedCols(min_basis, cycles)); 

    // ------------------------------------------------------------- // 
    // Test for disjoint cycles 
    // ------------------------------------------------------------- // 
    cplex = complex_disconnected_cycles();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 3);

    // Check that the minimal basis is precisely the minimal cycle basis
    cycles.resize(9, 3); 
    cycles << 1, 0, 0,
              1, 0, 0,
              1, 0, 0,
              0, 1, 0,
              0, 1, 0,
              0, 1, 0,
              0, 0, 1,
              0, 0, 1,
              0, 0, 1;
    REQUIRE(matchCyclesUpToPermutedCols<T>(cplex, min_basis, cycles, 1));
    REQUIRE(matchArraysUpToPermutedCols(min_basis, cycles)); 

    // ------------------------------------------------------------- // 
    // Test for Busaryev et al.'s example 
    // ------------------------------------------------------------- //
    cplex = complex_busaryev_example(); 
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 2);
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(0), min_basis.col(1), 1));

    // Define the two cycles 
    cycles = Matrix<Z2, Dynamic, Dynamic>::Zero(8, 2);
    cycles(0, 0) = 1; 
    cycles(2, 0) = 1; 
    cycles(4, 0) = 1; 
    cycles(3, 1) = 1; 
    cycles(5, 1) = 1; 
    cycles(6, 1) = 1; 
    REQUIRE(matchCyclesUpToPermutedCols<T>(cplex, min_basis, cycles, 1)); 

    // ------------------------------------------------------------- // 
    // Test for annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 1);

    // Define 3 cycles: 1) a tight cycle around the hole, 2) a loose cycle
    // around the hole (around the circumference of the annulus), and 3) a
    // null-homologous cycle
    cycles = Matrix<Z2, Dynamic, Dynamic>::Zero(24, 3);
    cycles(18, 0) = 1; 
    cycles(19, 0) = 1; 
    cycles(20, 0) = 1; 
    cycles(21, 0) = 1; 
    cycles(22, 0) = 1; 
    cycles(23, 0) = 1;
    cycles(0, 1) = 1; 
    cycles(1, 1) = 1; 
    cycles(4, 1) = 1; 
    cycles(7, 1) = 1; 
    cycles(10, 1) = 1; 
    cycles(13, 1) = 1; 
    cycles(0, 2) = 1; 
    cycles(2, 2) = 1; 
    cycles(4, 2) = 1; 
    cycles(9, 2) = 1; 
    cycles(18, 2) = 1; 
    cycles(20, 2) = 1; 
    REQUIRE(cplex.areHomologousCycles(cycles.col(0), cycles.col(1), 1)); 
    REQUIRE(cplex.areHomologousCycles(min_basis.col(0), cycles.col(0), 1));
    REQUIRE(cplex.areHomologousCycles(min_basis.col(0), cycles.col(1), 1)); 
    REQUIRE(!cplex.areHomologousCycles(cycles.col(0), cycles.col(2), 1));
    REQUIRE(!cplex.areHomologousCycles(cycles.col(1), cycles.col(2), 1)); 
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(0), cycles.col(2), 1)); 

    // ------------------------------------------------------------- // 
    // Test for two-holed annulus
    // ------------------------------------------------------------- //
    cplex = complex_annulus_two_holes();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 2);
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(0), min_basis.col(1), 1));

    // Consider 11 different cycles ... 
    //
    // Left hole, small: [v6,v7], [v7,v8], [v8,v9], [v9,v10], [v10,v11],
    //                   [v11,v6]
    cycles = Matrix<Z2, Dynamic, Dynamic>::Zero(47, 11);
    cycles(24, 0) = 1;
    cycles(25, 0) = 1; 
    cycles(26, 0) = 1; 
    cycles(27, 0) = 1;
    cycles(28, 0) = 1;
    cycles(29, 0) = 1;
    // Left hole, large: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 1) = 1; 
    cycles(1, 1) = 1; 
    cycles(4, 1) = 1; 
    cycles(10, 1) = 1; 
    cycles(16, 1) = 1; 
    cycles(19, 1) = 1;
    // Right hole, small: [v16,v17], [v17,v18], [v18,v19], [v19,v20], [v20,v21],
    //                    [v21,v16]
    cycles(41, 2) = 1;
    cycles(42, 2) = -1;
    cycles(43, 2) = 1; 
    cycles(44, 2) = 1; 
    cycles(45, 2) = 1; 
    cycles(46, 2) = 1;
    // Right hole, large: [v12,v13], [v13,v14], [v14,v15], [v15,v2], [v2,v1],
    //                    [v1,v12]
    cycles(30, 3) = 1;
    cycles(33, 3) = 1; 
    cycles(36, 3) = 1; 
    cycles(13, 3) = -1; 
    cycles(4, 3) = -1; 
    cycles(7, 3) = 1; 
    // Both holes, small: [v6,v7], [v7,v1], [v1,v21], [v21,v16], [v16,v17],
    //                    [v17,v18], [v18,v19], [v19,v20], [v20,v2], [v2,v8],
    //                    [v8,v9], [v9,v10], [v10,v11], [v11,v6]
    cycles(24, 4) = 1; 
    cycles(6, 4) = -1; 
    cycles(9, 4) = 1; 
    cycles(42, 4) = -1;
    cycles(41, 4) = 1;
    cycles(43, 4) = 1;
    cycles(44, 4) = 1;
    cycles(45, 4) = 1;
    cycles(15, 4) = -1; 
    cycles(12, 4) = 1;
    cycles(27, 4) = 1;
    cycles(28, 4) = 1;
    cycles(29, 4) = 1;
    cycles(25, 4) = -1;
    // Both holes, large: [v0,v1], [v1,v12], [v12,v13], [v13,v14], [v14,v15],
    //                    [v15,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 5) = 1; 
    cycles(7, 5) = 1; 
    cycles(30, 5) = 1; 
    cycles(33, 5) = 1;
    cycles(36, 5) = 1;
    cycles(13, 5) = -1;
    cycles(10, 5) = 1;
    cycles(16, 5) = 1;
    cycles(19, 5) = 1;
    cycles(1, 5) = -1;
    // Left hole, twice 1: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v0], [v0,v6], [v6,v1], [v1,v7],
    //                     [v7,v2], [v2,v8], [v8,v3], [v3,v9], [v9,v4],
    //                     [v4,v10], [v10,v5], [v5,v0]
    cycles(0, 6) = 1; 
    cycles(4, 6) = 1; 
    cycles(10, 6) = 1; 
    cycles(16, 6) = 1; 
    cycles(19, 6) = 1;
    cycles(23, 6) = 1; 
    cycles(3, 6) = -1; 
    cycles(2, 6) = 1;
    cycles(5, 6) = -1;
    cycles(6, 6) = 1;
    cycles(11, 6) = -1;
    cycles(12, 6) = 1; 
    cycles(17, 6) = -1;
    cycles(18, 6) = 1;
    cycles(20, 6) = -1;
    cycles(21, 6) = 1;
    cycles(22, 6) = -1; 
    cycles(1, 6) = -1;
    // Left hole, twice 2: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v6], [v6,v7], [v7,v8], [v8,v9],
    //                     [v9,v10], [v10,v11], [v11,v0]
    cycles(0, 7) = 1; 
    cycles(4, 7) = 1; 
    cycles(10, 7) = 1; 
    cycles(16, 7) = 1; 
    cycles(19, 7) = 1;
    cycles(23, 7) = 1;
    cycles(25, 7) = -1;
    cycles(24, 7) = 1;
    cycles(26, 7) = 1;
    cycles(27, 7) = 1;
    cycles(28, 7) = 1;
    cycles(29, 7) = 1;
    cycles(3, 7) = -1;
    // Right hole, twice 1: [v12,v13], [v13,v14], [v14,v15], [v15,v2],
    //                      [v2,v1], [v1,v21], [v21,v12], [v12,v16],
    //                      [v16,v13], [v13,v17], [v17,v14], [v14,v18],
    //                      [v18,v15], [v15,v19], [v19,v2], [v2,v20],
    //                      [v20,v1], [v1,v12]
    cycles(30, 8) = 1;
    cycles(33, 8) = 1; 
    cycles(36, 8) = 1;
    cycles(13, 8) = -1; 
    cycles(4, 8) = -1;
    cycles(9, 8) = 1;
    cycles(32, 8) = -1;
    cycles(31, 8) = 1;
    cycles(34, 8) = -1; 
    cycles(35, 8) = 1;
    cycles(37, 8) = -1;
    cycles(38, 8) = 1;
    cycles(39, 8) = -1;
    cycles(40, 8) = 1;
    cycles(14, 8) = -1;
    cycles(15, 8) = 1;
    cycles(8, 8) = -1; 
    cycles(7, 8) = 1;
    // Right hole, twice 2: [v12,v13], [v13,v14], [v14,v15], [v15,v2],
    //                      [v2,v1], [v1,v21], [v21,v16], [v16,v17],
    //                      [v17,v18], [v18,v19], [v19,v20], [v20,v21],
    //                      [v21,v12]
    cycles(30, 9) = 1;
    cycles(33, 9) = 1; 
    cycles(36, 9) = 1;
    cycles(13, 9) = -1; 
    cycles(4, 9) = -1;
    cycles(9, 9) = 1;
    cycles(42, 9) = -1;
    cycles(41, 9) = 1;
    cycles(43, 9) = 1; 
    cycles(44, 9) = 1;
    cycles(45, 9) = 1;
    cycles(46, 9) = 1;
    cycles(32, 9) = -1;
    // Figure eight: [v0,v1], [v1,v20], [v20,v19], [v19,v18], [v18,v17],
    //               [v17,v16], [v16,v21], [v21,v20], [v20,v2], [v2,v3],
    //               [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 10) = 1;
    cycles(8, 10) = 1;
    cycles(45, 10) = -1;
    cycles(44, 10) = -1;
    cycles(43, 10) = -1; 
    cycles(41, 10) = -1;
    cycles(42, 10) = 1;
    cycles(46, 10) = -1;
    cycles(15, 10) = -1;
    cycles(10, 10) = 1;
    cycles(16, 10) = 1;
    cycles(19, 10) = 1;
    cycles(1, 10) = -1;

    // Check that the two basis cycles are homologous to only the cycles 
    // that wrap around the left or right holes once 
    REQUIRE((
        cplex.areHomologousCycles(min_basis.col(0), cycles.col(0), 1) ^ 
        cplex.areHomologousCycles(min_basis.col(1), cycles.col(0), 1)
    ));

    // In the first case, basis cycle 0 is homologous to cycles around the
    // left hole
    int left, right;  
    if (cplex.areHomologousCycles(min_basis.col(0), cycles.col(0), 1))
    {
        left = 0; 
        right = 1;
    }
    else 
    {
        left = 1; 
        right = 0;
    }
    REQUIRE(cplex.areHomologousCycles(min_basis.col(left), cycles.col(1), 1));
    for (int i = 2; i < 11; ++i)
        REQUIRE(!cplex.areHomologousCycles(min_basis.col(left), cycles.col(i), 1));
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(right), cycles.col(1), 1));  
    REQUIRE(cplex.areHomologousCycles(min_basis.col(right), cycles.col(2), 1)); 
    REQUIRE(cplex.areHomologousCycles(min_basis.col(right), cycles.col(3), 1)); 
    for (int i = 4; i < 11; ++i)
        REQUIRE(!cplex.areHomologousCycles(min_basis.col(right), cycles.col(i), 1));

    // ------------------------------------------------------------- // 
    // Test for disjoint union or two annuli 
    // ------------------------------------------------------------- //
    cplex = complex_disjoint_annuli();
    min_basis = cplex.getMinimalFirstHomology();
    REQUIRE(min_basis.cols() == 2);
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(0), min_basis.col(1), 1));

    // Consider 8 different cycles ... 
    //
    // Left hole, small: [v6,v7], [v7,v8], [v8,v9], [v9,v10], [v10,v11],
    //                   [v11,v6]
    cycles = Matrix<Z2, Dynamic, Dynamic>::Zero(48, 8);
    cycles(18, 0) = 1;
    cycles(19, 0) = 1; 
    cycles(20, 0) = 1; 
    cycles(21, 0) = 1;
    cycles(22, 0) = 1;
    cycles(23, 0) = 1;
    // Left hole, large: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0]
    cycles(0, 1) = 1; 
    cycles(1, 1) = 1; 
    cycles(4, 1) = 1; 
    cycles(7, 1) = 1; 
    cycles(10, 1) = 1; 
    cycles(13, 1) = 1;
    // Right hole, small: [v16,v17], [v17,v18], [v18,v19], [v19,v20], [v20,v21],
    //                    [v21,v16]
    cycles(42, 2) = 1;
    cycles(43, 2) = 1;
    cycles(44, 2) = 1; 
    cycles(45, 2) = 1; 
    cycles(46, 2) = 1; 
    cycles(47, 2) = 1;
    // Right hole, large: [v12,v13], [v13,v14], [v14,v15], [v15,v16], [v16,v17],
    //                    [v17,v12]
    cycles(24, 3) = 1;
    cycles(25, 3) = 1; 
    cycles(28, 3) = 1; 
    cycles(31, 3) = 1; 
    cycles(34, 3) = 1; 
    cycles(37, 3) = 1; 
    // Left hole, twice 1: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v0], [v0,v6], [v6,v1], [v1,v7],
    //                     [v7,v2], [v2,v8], [v8,v3], [v3,v9], [v9,v4],
    //                     [v4,v10], [v10,v5], [v5,v0]
    cycles(0, 4) = 1; 
    cycles(4, 4) = 1; 
    cycles(7, 4) = 1; 
    cycles(10, 4) = 1; 
    cycles(13, 4) = 1;
    cycles(17, 4) = 1; 
    cycles(3, 4) = 1; 
    cycles(2, 4) = 1;
    cycles(5, 4) = 1;
    cycles(6, 4) = 1;
    cycles(8, 4) = 1;
    cycles(9, 4) = 1; 
    cycles(11, 4) = 1;
    cycles(12, 4) = 1;
    cycles(14, 4) = 1;
    cycles(15, 4) = 1;
    cycles(16, 4) = 1; 
    cycles(1, 4) = 1;
    // Left hole, twice 2: [v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5],
    //                     [v5,v11], [v11,v6], [v6,v7], [v7,v8], [v8,v9],
    //                     [v9,v10], [v10,v11], [v11,v0]
    cycles(0, 5) = 1; 
    cycles(4, 5) = 1; 
    cycles(7, 5) = 1; 
    cycles(10, 5) = 1; 
    cycles(13, 5) = 1;
    cycles(17, 5) = 1; 
    cycles(19, 5) = 1;
    cycles(18, 5) = 1;
    cycles(20, 5) = 1;
    cycles(21, 5) = 1;
    cycles(22, 5) = 1;
    cycles(23, 5) = 1;
    cycles(3, 5) = 1;
    // Right hole, twice 1: [v12,v13], [v13,v14], [v14,v15], [v15,v16],
    //                      [v16,v17], [v17,v23], [v23,v12], [v12,v18],
    //                      [v18,v13], [v13,v19], [v19,v14], [v14,v20],
    //                      [v20,v15], [v15,v21], [v21,v16], [v16,v22],
    //                      [v22,v17], [v17,v12]
    cycles(24, 6) = 1;
    cycles(28, 6) = 1; 
    cycles(31, 6) = 1;
    cycles(34, 6) = 1; 
    cycles(37, 6) = 1;
    cycles(41, 6) = 1;
    cycles(27, 6) = 1;
    cycles(26, 6) = 1;
    cycles(29, 6) = 1;
    cycles(30, 6) = 1; 
    cycles(32, 6) = 1;
    cycles(33, 6) = 1;
    cycles(35, 6) = 1;
    cycles(36, 6) = 1;
    cycles(38, 6) = 1;
    cycles(39, 6) = 1;
    cycles(40, 6) = 1;
    cycles(25, 6) = 1;
    // Right hole, twice 2: [v12,v13], [v13,v14], [v14,v15], [v15,v16],
    //                      [v16,v17], [v17,v23], [v23,v18], [v18,v19],
    //                      [v19,v20], [v20,v21], [v21,v22], [v22,v17],
    //                      [v17,v12]
    cycles(24, 7) = 1;
    cycles(28, 7) = 1; 
    cycles(31, 7) = 1;
    cycles(34, 7) = 1; 
    cycles(37, 7) = 1;
    cycles(41, 7) = 1;
    cycles(43, 7) = 1;
    cycles(42, 7) = 1;
    cycles(44, 7) = 1; 
    cycles(45, 7) = 1;
    cycles(46, 7) = 1;
    cycles(40, 7) = 1;
    cycles(25, 7) = 1;

    // Check that the two basis cycles are homologous to only the cycles 
    // that wrap around the left or right holes once 
    REQUIRE((
        cplex.areHomologousCycles(min_basis.col(0), cycles.col(0), 1) ^ 
        cplex.areHomologousCycles(min_basis.col(1), cycles.col(0), 1)
    ));

    // In the first case, basis cycle 0 is homologous to cycles around the
    // left hole
    if (cplex.areHomologousCycles(min_basis.col(0), cycles.col(0), 1))
    {
        left = 0; 
        right = 1;
    }
    else 
    {
        left = 1; 
        right = 0;
    }
    REQUIRE(cplex.areHomologousCycles(min_basis.col(left), cycles.col(1), 1));
    for (int i = 2; i < 8; ++i)
        REQUIRE(!cplex.areHomologousCycles(min_basis.col(left), cycles.col(i), 1));
    REQUIRE(!cplex.areHomologousCycles(min_basis.col(right), cycles.col(1), 1));  
    REQUIRE(cplex.areHomologousCycles(min_basis.col(right), cycles.col(2), 1)); 
    REQUIRE(cplex.areHomologousCycles(min_basis.col(right), cycles.col(3), 1)); 
    for (int i = 4; i < 8; ++i)
        REQUIRE(!cplex.areHomologousCycles(min_basis.col(right), cycles.col(i), 1));
}

TEST_CASE("Tests for minimal cycle calculations", "[getMinimalCycles()]")
{
    SimplicialComplex3D<T> cplex;
    Matrix<Z2, Dynamic, Dynamic> min_basis; 
    Matrix<double, Dynamic, Dynamic> opt_cycles; 

    // ------------------------------------------------------------- // 
    // Test for simple cycle
    // ------------------------------------------------------------- // 
    cplex = complex_cycle();
    min_basis.resize(3, 1); 
    min_basis << 1, 1, 1;      // This is the only cycle in the complex  
    opt_cycles = cplex.minimizeCycles(
        min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_SIZE,
        Matrix<double, Dynamic, 1>::Zero(1)    // Weights do not matter 
    );

    // Check that the nonzero entries correspond to the minimal cycle, 
    // which contain all the edges
    REQUIRE(opt_cycles.rows() == cplex.getNumSimplices(1)); 
    REQUIRE(opt_cycles.cols() == 1);  
    REQUIRE(opt_cycles(0) == 1); 
    REQUIRE(opt_cycles(1) == 1); 
    REQUIRE(opt_cycles(2) == 1); 

    // ------------------------------------------------------------- // 
    // Test for 2-D mesh with hole 
    // ------------------------------------------------------------- // 
    cplex = complex_2d_mesh_with_hole();
    min_basis.resize(15, 1); 
    min_basis << 1,    // Define a long cycle around the hole 
                 1,
                 0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 1,
                 1,
                 0,
                 1,
                 1,
                 0,
                 0;
    REQUIRE(cplex.isCycle(min_basis.col(0), 1));  
    opt_cycles = cplex.minimizeCycles(
        min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_SIZE,
        Matrix<double, Dynamic, 1>::Zero(1)    // Weights do not matter
    );

    // Check that the nonzero entries correspond to the minimal cycle, 
    // which is the only cycle of length 4 and contains edges 3, 4, 7, 9
    // ([v1,v3], [v3,v4], [v4,v2], [v2,v1])
    REQUIRE(opt_cycles.rows() == cplex.getNumSimplices(1)); 
    REQUIRE(opt_cycles.cols() == 1);
    for (int i = 0; i < opt_cycles.rows(); ++i)
    {
        if (i == 3 || i == 4 || i == 7 || i == 9)
            REQUIRE(opt_cycles(i) != 0); 
        else 
            REQUIRE(opt_cycles(i) == 0); 
    } 
    
    // ------------------------------------------------------------- // 
    // Test for annulus 
    // ------------------------------------------------------------- //
    cplex = complex_annulus(); 
    min_basis.resize(24, 1); 
    min_basis << 0,    // Define a long cycle around the hole 
                 0,
                 1,    // [v0,v6]
                 1,    // [v0,v11]
                 0,
                 1,    // [v1,v6]
                 1,    // [v1,v7]
                 0,
                 1,    // [v2,v7]
                 1,    // [v2,v8]
                 0,
                 1,    // [v3,v8]
                 1,    // [v3,v9]
                 0,
                 1,    // [v4,v9]
                 1,    // [v4,v10]
                 1,    // [v5,v10]
                 1,    // [v5,v11]
                 0,
                 0,
                 0,
                 0,
                 0,
                 0;
    REQUIRE(cplex.isCycle(min_basis.col(0), 1));  
    opt_cycles = cplex.minimizeCycles(
        min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_SIZE,
        Matrix<double, Dynamic, 1>::Zero(1)    // Weights do not matter
    );
    
    // Check that the nonzero entries correspond to one of two possible
    // minimal cycles, which both contain six edges:
    // - edges 0, 1, 4, 7, 10, 13, corresponding to the largest possible
    //   cycle ([v0,v1], [v1,v2], [v2,v3], [v3,v4], [v4,v5], [v5,v0])
    // - edges 18, 19, 20, 21, 22, 23, corresponding to the smallest 
    //   possible cycle ([v6,v7], [v7,v8], [v8,v9], [v9,v10], [v10,v11],
    //   [v11,v6])
    REQUIRE(opt_cycles.rows() == cplex.getNumSimplices(1)); 
    REQUIRE(opt_cycles.cols() == 1);
    Matrix<double, Dynamic, 1> cycle1 = Matrix<double, Dynamic, 1>::Zero(24);
    cycle1(0) = 1; 
    cycle1(1) = 1; 
    cycle1(4) = 1; 
    cycle1(7) = 1; 
    cycle1(10) = 1; 
    cycle1(13) = 1;
    Matrix<double, Dynamic, 1> cycle2 = Matrix<double, Dynamic, 1>::Zero(24);
    cycle2(18) = 1; 
    cycle2(19) = 1; 
    cycle2(20) = 1; 
    cycle2(21) = 1; 
    cycle2(22) = 1; 
    cycle2(23) = 1;
    bool match_cycle1 = true; 
    bool match_cycle2 = true; 
    for (int i = 0; i < opt_cycles.rows(); ++i)
    {
        if (cycle1(i) == 0 && opt_cycles(i, 0) != 0)
        {
            match_cycle1 = false; 
            break; 
        }
        else if (cycle1(i) == 1 && opt_cycles(i, 0) == 0)
        {
            match_cycle1 = false; 
            break; 
        } 
    }
    for (int i = 0; i < opt_cycles.rows(); ++i)
    {
        if (cycle2(i) == 0 && opt_cycles(i, 0) != 0)
        {
            match_cycle2 = false; 
            break; 
        }
        else if (cycle2(i) == 1 && opt_cycles(i, 0) == 0)
        {
            match_cycle2 = false; 
            break; 
        } 
    }
    REQUIRE((match_cycle1 ^ match_cycle2)); 

    // Now minimize with respect to the cycle length ...
    opt_cycles = cplex.minimizeCycles(
        min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_VOLUME,
        Matrix<double, Dynamic, 1>::Zero(1)    // Weights do not matter
    );

    // Check that the nonzero entries correspond to the shortest possible
    // minimal cycle: edges 18, 19, 20, 21, 22, 23 ([v6,v7], [v7,v8], [v8,v9],
    // [v9,v10], [v10,v11], [v11,v6])
    match_cycle2 = true; 
    for (int i = 0; i < opt_cycles.rows(); ++i)
    {
        if (cycle2(i) == 0 && opt_cycles(i, 0) != 0)
        {
            match_cycle2 = false; 
            break; 
        }
        else if (cycle2(i) == 1 && opt_cycles(i, 0) == 0)
        {
            match_cycle2 = false; 
            break; 
        } 
    }
    REQUIRE(match_cycle2); 

    // Try minimizing with respect to the cycle length, again starting from
    // the longest possible cycle (which also has size 6)
    min_basis << 1,
                 1,
                 0,
                 0,
                 1,
                 0,
                 0,
                 1,
                 0,
                 0,
                 1,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0;
    REQUIRE(cplex.isCycle(min_basis.col(0), 1));  
    opt_cycles = cplex.minimizeCycles(
        min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_VOLUME,
        Matrix<double, Dynamic, 1>::Zero(1)    // Weights do not matter
    );

    // Check that the nonzero entries correspond to the shortest possible
    // minimal cycle: edges 18, 19, 20, 21, 22, 23 ([v6,v7], [v7,v8], [v8,v9],
    // [v9,v10], [v10,v11], [v11,v6])
    match_cycle2 = true; 
    for (int i = 0; i < opt_cycles.rows(); ++i)
    {
        if (cycle2(i) == 0 && opt_cycles(i, 0) != 0)
        {
            match_cycle2 = false; 
            break; 
        }
        else if (cycle2(i) == 1 && opt_cycles(i, 0) == 0)
        {
            match_cycle2 = false; 
            break; 
        } 
    }
    REQUIRE(match_cycle2); 
}

