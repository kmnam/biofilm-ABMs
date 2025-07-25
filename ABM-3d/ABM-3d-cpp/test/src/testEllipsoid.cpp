/**
 * Test module for the functions in `ellipsoid.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/24/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/ellipsoid.hpp"
#include "../../include/utils.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T; 

using std::abs; 
using boost::multiprecision::abs; 
using std::sin; 
using boost::multiprecision::sin;
using std::cos; 
using boost::multiprecision::cos; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::real; 
using boost::multiprecision::real; 
using std::imag; 
using boost::multiprecision::imag;
using std::acos; 
using boost::multiprecision::acos; 

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for rotationToXUnitVector(). 
 */
TEST_CASE("Tests for rotation matrix function", "[rotationToXUnitVector()]")
{
    const double tol = 1e-8; 
    Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::LinSpaced(20, 0, 1);
    for (int i = 0; i < 20; ++i)
    {
        for (int j = 0; j < 20; ++j)
        {
            T sqnorm = x(i) * x(i) + x(j) * x(j);  
            if (sqnorm < 1)
            {
                Matrix<T, 3, 1> n;
                n << x(i), x(j), sqrt(1 - x(i) * x(i) - x(j) * x(j));
                Matrix<T, 3, 3> rot = rotationToXUnitVector<T>(n); 
                Matrix<T, 3, 1> y = rot * n; 
                REQUIRE_THAT(static_cast<double>(y(0)), Catch::Matchers::WithinAbs(1.0, tol)); 
                REQUIRE_THAT(static_cast<double>(y(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
                REQUIRE_THAT(static_cast<double>(y(2)), Catch::Matchers::WithinAbs(0.0, tol));
                REQUIRE_THAT(static_cast<double>(rot.determinant()), Catch::Matchers::WithinAbs(1.0, tol));  
            }
        }
    }
}

/**
 * A series of tests for getEllipsoidQuadraticForm(). 
 */
TEST_CASE("Tests for ellipsoid quadratic form function", "[getEllipsoidQuadraticForm()]")
{
    const double tol = 1e-8; 

    // Case 1: Ellipsoid with long axis parallel to the x-axis  
    Matrix<T, 3, 1> r, n; 
    r << 0, 0, 0; 
    n << 1, 0, 0; 
    const T R = 0.8; 
    const T half_l = 0.5;
    auto result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> A = std::get<0>(result);
    Matrix<T, 3, 1> b = std::get<1>(result);
    T c = std::get<2>(result); 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 0) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(
                        static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol
                    )
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(1.0, tol)); 

    // Case 2: Parallel to x-axis but off-center
    //
    // Here, the matrix A should be the same as in Case 1, but the vector b
    // and scalar c should change due to the translation  
    r << 1, 2, 3;
    n << 1, 0, 0;
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result); 
    b = std::get<1>(result);
    c = std::get<2>(result);  
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 0) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(
                        static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol
                    )
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
    }
    // Compare against pre-computed values in Python
    REQUIRE_THAT(static_cast<double>(b(0)), Catch::Matchers::WithinAbs(-0.5917159763313609, tol)); 
    REQUIRE_THAT(static_cast<double>(b(1)), Catch::Matchers::WithinAbs(-3.125, tol)); 
    REQUIRE_THAT(static_cast<double>(b(2)), Catch::Matchers::WithinAbs(-4.6875, tol));
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(-19.904215976331358, tol));  

    // Case 3: Ellipsoid with long axis parallel to the y-axis
    //
    // Here, the matrix A should have its entries permuted to account for 
    // the rotation, and the vector b should be zero and scalar c should be
    // one
    r << 0, 0, 0;
    n << 0, 1, 0; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result); 
    b = std::get<1>(result);
    c = std::get<2>(result);  
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 1) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(1.0, tol)); 

    // Case 4: Parallel to y-axis but off-center
    //
    // Here, the matrix A should be the same as in Case 3, but the vector b
    // and scalar c should change due to the translation  
    r << 1, 2, 3;
    n << 0, 1, 0; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result); 
    b = std::get<1>(result);
    c = std::get<2>(result);  
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 1) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
    }
    // Compare against pre-computed values in Python
    REQUIRE_THAT(static_cast<double>(b(0)), Catch::Matchers::WithinAbs(-1.5625, tol));
    REQUIRE_THAT(static_cast<double>(b(1)), Catch::Matchers::WithinAbs(-1.1834319526627217, tol)); 
    REQUIRE_THAT(static_cast<double>(b(2)), Catch::Matchers::WithinAbs(-4.6875, tol));
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(-16.99186390532544, tol));  

    // Case 5: Ellipsoid with long axis parallel to the z-axis 
    //
    // Here, the matrix A should have its entries permuted to account for 
    // the rotation, and the vector b should be zero and scalar c should be
    // one
    r << 0, 0, 0;
    n << 0, 0, 1; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result); 
    b = std::get<1>(result);
    c = std::get<2>(result);  
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 2) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(1.0, tol));

    // Case 6: Parallel to z-axis but off-center
    //
    // Here, the matrix A should be the same as in Case 5, but the vector b
    // and scalar c should change due to the translation  
    r << 1, 2, 3;
    n << 0, 0, 1; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result); 
    b = std::get<1>(result);
    c = std::get<2>(result);  
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 2) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
    }
    // Compare against pre-computed values in Python
    REQUIRE_THAT(static_cast<double>(b(0)), Catch::Matchers::WithinAbs(-1.5625, tol));
    REQUIRE_THAT(static_cast<double>(b(1)), Catch::Matchers::WithinAbs(-3.125, tol)); 
    REQUIRE_THAT(static_cast<double>(b(2)), Catch::Matchers::WithinAbs(-1.7751479289940826, tol));
    REQUIRE_THAT(static_cast<double>(c), Catch::Matchers::WithinAbs(-12.137943786982246, tol));  

    // Case 7: Ellipsoid with misaligned long axes
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;  
    r << 1, 2, 3;
    const int n_cases = 100;
    Matrix<T, Dynamic, 3> orientations = Matrix<T, Dynamic, 3>::Zero(n_cases, 3);
    for (int k = 0; k < n_cases; ++k)
    {
        orientations(k, 0) = dist(rng); 
        orientations(k, 1) = dist(rng); 
        orientations(k, 2) = dist(rng); 
        orientations.row(k) /= orientations.row(k).norm(); 
    }

    // For each orientation ...  
    for (int k = 0; k < n_cases; ++k)
    {
        result = getEllipsoidQuadraticForm<T>(r, orientations.row(k), R, half_l); 
        A = std::get<0>(result); 
        b = std::get<1>(result); 
        c = std::get<2>(result); 

        // Test that A is symmetric 
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (i != j)
                    REQUIRE(abs(A(i, j) - A(j, i)) < tol); 
            }
        }

        // Test that the eigenvalues are all positive real
        EigenSolver<Matrix<double, 3, 3> > eigen(A.cast<double>());  
        Matrix<std::complex<double>, 3, 1> eigvals = eigen.eigenvalues(); 
        for (int i = 0; i < 3; ++i)
        {
            REQUIRE(real(eigvals(i)) > tol);
            REQUIRE(abs(imag(eigvals(i))) < tol);
        }

        // Check the semi-major and semi-minor axis lengths 
        Matrix<double, 3, 1> axis_lengths;
        axis_lengths << 1.0 / sqrt(real(eigvals(0))),
                        1.0 / sqrt(real(eigvals(1))), 
                        1.0 / sqrt(real(eigvals(2)));
        Matrix<double, 3, 1>::Index maxidx; 
        double max_length = axis_lengths.maxCoeff(&maxidx);
        REQUIRE_THAT(max_length, Catch::Matchers::WithinAbs(static_cast<double>(R + half_l), tol));  
        if (maxidx == 0) 
        {
            REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
            REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
        }
        else if (maxidx == 1)
        {
            REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
            REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
        }
        else 
        {
            REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
            REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
        }

        // Check the semi-major axis direction
        for (int i = 0; i < 3; ++i)
        { 
            REQUIRE_THAT(
                abs(eigen.eigenvectors()(i, maxidx)),
                Catch::Matchers::WithinAbs(static_cast<double>(orientations(k, i)), tol)
            );
        }
    }
}

/**
 * A series of tests for getPrincipalRadiiOfCurvature(). 
 */
TEST_CASE("Tests for principal radii of curvature", "[getPrincipalRadiiOfCurvature()]")
{
    const double tol = 1e-8; 
    Matrix<T, 3, 1> n, x; 
    const T R = 0.8;
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist; 

    // Case 1: Get principal radii of curvature on a sphere
    n << 1, 0, 0;      // Dummy orientation 
    T half_l = 0.0;

    // Sample points on the sphere and calculate radii 
    for (int i = 0; i < 100; ++i)
    {
        x << standardNormal<T>(rng, dist),
             standardNormal<T>(rng, dist),
             standardNormal<T>(rng, dist); 
        x *= (R / x.norm());
        std::pair<T, T> radii = getPrincipalRadiiOfCurvature<T>(n, R, half_l, x);
        REQUIRE_THAT(
            static_cast<double>(radii.first),
            Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(radii.second),
            Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
        ); 
    }

    // Case 2: Get principal radii of curvature on an ellipsoid
    //
    // For this, we implement a numerical estimation method for the principal
    // radii of curvature as eigenvalues of the shape operator 
    half_l = 0.5;
    T a = R + half_l; 
    T b = R; 
    std::function<Matrix<T, 3, 1>(const Ref<const Matrix<T, 3, 1> >&)> grad
        = [&a, &b](const Ref<const Matrix<T, 3, 1> >& x) -> Matrix<T, 3, 1>
        {
            // Gradient of (x[0] / a)^2 + (x[1] / b)^2 + (x[2] / b)^2 - 1
            Matrix<T, 3, 1> g; 
            g << 2 * x(0) / (a * a), 2 * x(1) / (b * b), 2 * x(2) / (b * b); 
            return g; 
        };
    std::function<Matrix<T, 3, 3>(const Ref<const Matrix<T, 3, 1> >&)> hessian
        = [&a, &b](const Ref<const Matrix<T, 3, 1> >& x) -> Matrix<T, 3, 3>
        {
            // Second partial derivatives of (x[0] / a)^2 + (x[1] / b)^2 + (x[2] / b)^2 - 1
            Matrix<T, 3, 3> H; 
            H << 2 / (a * a),           0,           0, 
                           0, 2 / (b * b),           0,
                           0,           0, 2 / (b * b);
            return H;  
        };

    // Simple function for calculating determinants of 4x4 matrices 
    std::function<T(const Ref<const Matrix<T, 4, 4> >&)> determinant
        = [](const Ref<const Matrix<T, 4, 4> >& A) -> T
        {
            // Initialize the determinant of the 4x4 matrix 
            T det = 0; 

            // Apply the Laplace expansion: For each column ...  
            for (int i = 0; i < 4; ++i)
            {
                // Get the submatrix obtained by removing row 0 and column i
                std::vector<int> cols;
                for (int j = 0; j < 4; ++j)
                {
                    if (i != j)
                        cols.push_back(j); 
                } 
                Matrix<T, 3, 3> sub_i = A(Eigen::seq(1, 3), cols);

                // Calculate the determinant of this 3x3 submatrix, again 
                // by applying the Laplace expansion  
                T det_i = 0;

                // For each column of the submatrix ...  
                for (int j = 0; j < 3; ++j)
                {
                    // Get the determinant of the submatrix obtained by 
                    // removing row 0 and column j
                    std::vector<int> cols2; 
                    for (int k = 0; k < 3; ++k)
                    {
                        if (j != k)
                            cols2.push_back(k); 
                    } 
                    T det_ij = (
                        sub_i(1, cols2[0]) * sub_i(2, cols2[1]) -
                        sub_i(1, cols2[1]) * sub_i(2, cols2[0])
                    );

                    // Update the 3x3 determinant
                    det_i += pow(-1, j) * sub_i(0, j) * det_ij;  
                }

                // Update the 4x4 determinant
                det += pow(-1, i) * A(0, i) * det_i;  
            }

            return det; 
        };

    // Sample points on the ellipsoid and calculate radii
    T dtheta = 1e-8;  
    for (int i = 0; i < 100; ++i)
    {
        // Sample a point from the ellipsoid using spherical coordinates
        T theta = boost::math::constants::pi<T>() * static_cast<T>(dist(rng));
        T phi = boost::math::constants::two_pi<T>() * static_cast<T>(dist(rng));
        Matrix<T, 3, 1> x;  
        x << (R + half_l) * sin(theta) * cos(phi),
             R * sin(theta) * sin(phi),
             R * cos(theta);

        // Calculate the principal radii of curvature 
        std::pair<T, T> radii = getPrincipalRadiiOfCurvature<T>(n, R, half_l, x);
        T curvature1 = 1.0 / radii.first; 
        T curvature2 = 1.0 / radii.second;  

        // Calculate the gradient, and check that it is normal to a tangent
        // vector at x
        //
        // This tangent vector is obtained along the curve with constant phi
        Matrix<T, 3, 1> grad_x = grad(x);
        Matrix<T, 3, 1> normal = grad_x / grad_x.norm(); 
        Matrix<T, 3, 1> x1, x2; 
        x1 << (R + half_l) * sin(theta + dtheta) * cos(phi), 
              R * sin(theta + dtheta) * sin(phi), 
              R * cos(theta + dtheta); 
        x2 << (R + half_l) * sin(theta - dtheta) * cos(phi),
              R * sin(theta - dtheta) * sin(phi),
              R * cos(theta - dtheta); 
        Matrix<T, 3, 1> tangent_x = (x1 - x2) / (2 * theta);
        tangent_x /= tangent_x.norm();  
        REQUIRE(static_cast<double>(abs(normal.dot(tangent_x))) < tol);

        // Calculate the mean curvature from the divergence of the normal,
        // which is the trace of the Hessian divided by the norm of the
        // gradient
        //
        // See Cor. 4.5 in Goldman, Computer Aided Geometric Design (2005)
        Matrix<T, 3, 3> hess_x = hessian(x); 
        T mean = hess_x.trace() / grad_x.norm();
        mean -= (
            8 * (x(0) * x(0) / pow(a, 6) + x(1) * x(1) / pow(b, 6) + x(2) * x(2) / pow(b, 6))
            * pow(grad_x.squaredNorm(), -1.5)
        );
        mean /= 2;

        // Check that this mean curvature is close to the mean of the 
        // principal curvatures
        REQUIRE_THAT(
            static_cast<double>(mean), 
            Catch::Matchers::WithinAbs(0.5 * static_cast<double>(curvature1 + curvature2), tol)
        );

        // Calculate the Gaussian curvature from the Hessian and normal
        //
        // See Cor. 4.2 in Goldman, Computer Aided Geometric Design (2005)
        Matrix<T, 4, 4> M = Matrix<T, 4, 4>::Zero();  
        M(Eigen::seq(0, 2), Eigen::seq(0, 2)) = hess_x; 
        M(Eigen::seq(0, 2), 3) = grad_x; 
        M(3, Eigen::seq(0, 2)) = grad_x.transpose(); 
        T gauss = -determinant(M) / pow(grad_x.squaredNorm(), 2);

        // Check that this mean curvature is close to the mean of the 
        // principal curvatures
        REQUIRE_THAT(
            static_cast<double>(gauss), 
            Catch::Matchers::WithinAbs(static_cast<double>(curvature1 * curvature2), tol)
        );
    }
}

/**
 * A series of tests for projectOntoEllipsoid().
 */
TEST_CASE("Tests for projection function", "[projectOntoEllipsoid()]")
{
    const double tol = 1e-8;
    const T inclusion_tol = 1e-8; 
    const T project_tol = 1e-20; 
    const int max_iter = 1000; 

    // -------------------------------------------------------------- //
    // Case 1: Ellipsoid with long axis parallel to the x-axis
    // -------------------------------------------------------------- //
    Matrix<T, 3, 1> r, n, a; 
    r << 0, 0, 0; 
    n << 1, 0, 0; 
    const T R = 0.8; 
    const T half_l = 0.5;
    auto result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> A = std::get<0>(result);
    std::function<bool(const Ref<const Matrix<T, 3, 1> >&)> in_ellipsoid_fuzzy
        = [&A, &r, &inclusion_tol](const Ref<const Matrix<T, 3, 1> >& q) -> bool
        {
            return ((q - r).dot(A * (q - r)) <= 1.0 + inclusion_tol);  
        };

    // Case 1a: Query point lies along x-axis
    //
    // In this case, the nearest point should be (R + half_l, 0, 0) 
    a << 2 * (R + half_l), 0, 0; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    Matrix<T, 3, 1> x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>(R + half_l), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));
    
    // Case 1b: Query point lies along x-axis, but on other side 
    //
    // In this case, the nearest point should be (-R + half_l, 0, 0) 
    a << -2 * (R + half_l), 0, 0;
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R - half_l), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 1c: Query point lies along y-axis
    //
    // In this case, the nearest point should be (0, R, 0) 
    a << 0, 2 * R, 0;
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 1d: Query point lies along y-axis, but on other side
    //
    // In this case, the nearest point should be (0, -R, 0) 
    a << 0, -2 * R, 0;
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 1e: Query point lies along z-axis
    //
    // In this case, the nearest point should be (0, 0, R) 
    a << 0, 0, 2 * R;
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(
        static_cast<double>(x(2)),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    );

    // Case 1f: Query point lies along z-axis, but on other side
    //
    // In this case, the nearest point should be (0, 0, -R) 
    a << 0, 0, -2 * R; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(
        static_cast<double>(x(2)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R), tol)
    );

    // Case 1g: Query point lies elsewhere 
    //
    // Here, we check optimality by randomly sampling values along the 
    // ellipsoid surface and checking their distance to the query point
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;  
    a << static_cast<T>(dist(rng)),
         static_cast<T>(dist(rng)), 
         static_cast<T>(dist(rng)); 
    a *= (5 * R / a.norm());    // Ensure that the point lies outside the ellipsoid
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    for (int i = 0; i < 1000; ++i)
    {
        // Sample a point from the ellipsoid using spherical coordinates
        T theta = boost::math::constants::pi<T>() * static_cast<T>(dist(rng));
        T phi = boost::math::constants::two_pi<T>() * static_cast<T>(dist(rng));
        Matrix<T, 3, 1> y; 
        y << (R + half_l) * sin(theta) * cos(phi),
             R * sin(theta) * sin(phi),
             R * cos(theta);
        REQUIRE(in_ellipsoid_fuzzy(y));  
        REQUIRE((x - a).norm() < (y - a).norm());  
    }

    // -------------------------------------------------------------- //
    // Case 2: Ellipsoid with orientation vector parallel to (cos(30), sin(30), 0)
    // -------------------------------------------------------------- //
    r << 0, 0, 0; 
    n << cos(boost::math::constants::sixth_pi<T>()),
         sin(boost::math::constants::sixth_pi<T>()),
         0; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = std::get<0>(result);
    in_ellipsoid_fuzzy
        = [&A, &r, &inclusion_tol](const Ref<const Matrix<T, 3, 1> >& q) -> bool
        {
            return ((q - r).dot(A * (q - r)) <= 1.0 + inclusion_tol);  
        };

    // Case 2a: Query point lies along orientation vector 
    //
    // In this case, the nearest point should be (b * cos(30), b * sin(30), 0)
    // with b = R + half_l
    a = 2 * (R + half_l) * n; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>((R + half_l) * n(0)), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>((R + half_l) * n(1)), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 2b: Query point lies along orientation vector, but on other side 
    //
    // In this case, the nearest point should be (-b * cos(30), -b * sin(30), 0)
    // with b = R + half_l
    a = -2 * (R + half_l) * n; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>(-(R + half_l) * n(0)), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>(-(R + half_l) * n(1)), tol)
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 2c: Query point lies perpendicular to orientation vector 
    //
    // In this case, the nearest point should be (-R * sin(30), R * cos(30), 0)
    a << -2 * R * sin(boost::math::constants::sixth_pi<T>()), 
          2 * R * cos(boost::math::constants::sixth_pi<T>()),
          0; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(
            static_cast<double>(-R * sin(boost::math::constants::sixth_pi<T>())), tol
        )
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(
            static_cast<double>(R * cos(boost::math::constants::sixth_pi<T>())), tol
        )
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 2d: Query point lies perpendicular to orientation vector, but 
    // on other side  
    //
    // In this case, the nearest point should be (R * sin(30), -R * cos(30), 0)
    a <<  2 * R * sin(boost::math::constants::sixth_pi<T>()), 
         -2 * R * cos(boost::math::constants::sixth_pi<T>()),
          0; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(
            static_cast<double>(R * sin(boost::math::constants::sixth_pi<T>())), tol
        )
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(
            static_cast<double>(-R * cos(boost::math::constants::sixth_pi<T>())), tol
        )
    ); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 2e: Query point lies perpendicular to orientation vector, but 
    // along different plane 
    //
    // In this case, the nearest point should be R * v, where v is the unit 
    // vector along which the query point lies 
    Matrix<T, 3, 1> u, v; 
    u <<  sin(boost::math::constants::sixth_pi<T>()),
         -cos(boost::math::constants::sixth_pi<T>()),
          0;
    v = n.cross(u);  
    a = 2 * R * v; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>(R * v(0)), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>(R * v(1)), tol) 
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(2)),
        Catch::Matchers::WithinAbs(static_cast<double>(R * v(2)), tol) 
    );

    // Case 2f: Query point lies along the same vector but on other side 
    //
    // In this case, the nearest point should be -R * v, where v is the same
    // unit vector as above 
    a *= -1; 
    REQUIRE(!in_ellipsoid_fuzzy(a)); 
    x = projectOntoEllipsoid<T>(a, A, r, project_tol, max_iter, true);
    REQUIRE(in_ellipsoid_fuzzy(x)); 
    REQUIRE_THAT(
        static_cast<double>(x(0)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R * v(0)), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(1)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R * v(1)), tol) 
    ); 
    REQUIRE_THAT(
        static_cast<double>(x(2)),
        Catch::Matchers::WithinAbs(static_cast<double>(-R * v(2)), tol) 
    );
}

/**
 * A series of tests for projectAndGetPrincipalRadiiOfCurvature(). 
 */
TEST_CASE(
    "Tests for projection + radii of curvature function",
    "[projectAndGetPrincipalRadiiOfCurvature()]"
)
{
    const double tol = 1e-8;
    const T inclusion_tol = 1e-8; 
    const T project_tol = 1e-20; 
    const int max_iter = 1000;  

    // Case 1: Ellipsoid with long axis parallel to the x-axis 
    Matrix<T, 3, 1> r, n, dnorm, x, y, z;
    std::pair<T, T> radii, radii_true; 
    T s;  
    r << 0, 0, 0; 
    n << 1, 0, 0;
    x << 1, 0, 0; 
    y << 0, 1, 0; 
    z << 0, 0, 1;  
    const T R = 0.8; 
    const T half_l = 0.5;

    // Case 1a: Query point lies along x-axis
    dnorm = x; 
    s = half_l;   
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    radii_true = getPrincipalRadiiOfCurvature<T>(n, R, half_l, (R + half_l) * x); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    ); 
    dnorm = -x; 
    s = -half_l; 
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    );
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        half_l, R, 0.0, half_l, project_tol, max_iter
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    );

    // Case 1b: Query point lies along y-axis 
    dnorm = y;
    s = 0;  
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    radii_true = getPrincipalRadiiOfCurvature<T>(n, R, half_l, R * y); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 
    dnorm = -y; 
    s = 0;  
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        half_l, R, acosSafe<T>(n.dot(dnorm)), s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 

    // Case 1c: Query point lies along z-axis 
    dnorm = z;
    s = 0;  
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 
    dnorm = -z; 
    s = 0;  
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    );
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        half_l, R, acosSafe<T>(n.dot(dnorm)), s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    );

    // Case 2: Ellipsoid with long axis parallel to the vector (cos(30), sin(30), 0)
    r << 0, 0, 0; 
    n << cos(boost::math::constants::sixth_pi<T>()),
         sin(boost::math::constants::sixth_pi<T>()),
         0;

    // Case 2a: Query point lies parallel to the orientation vector 
    dnorm = n; 
    s = half_l;   
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    radii_true = getPrincipalRadiiOfCurvature<T>(n, R, half_l, (R + half_l) * n);
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    );
    dnorm = -n; 
    s = -half_l;
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    );
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        half_l, R, 0.0, half_l, project_tol, max_iter
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    ); 
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.second), tol)
    );

    // Case 2b: Query point lies perpendicular to the orientation vector 
    dnorm << -sin(boost::math::constants::sixth_pi<T>()), 
              cos(boost::math::constants::sixth_pi<T>()),
              0;
    s = 0;
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    radii_true = getPrincipalRadiiOfCurvature<T>(n, R, half_l, R * dnorm);
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    );
    dnorm *= -1; 
    s = 0;
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        r, n, half_l, R, dnorm, s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    );
    radii = projectAndGetPrincipalRadiiOfCurvature<T>(
        half_l, R, acosSafe<T>(n.dot(dnorm)), s, project_tol, max_iter
    );
    REQUIRE_THAT(
        static_cast<double>(radii.first),
        Catch::Matchers::WithinAbs(static_cast<double>(radii_true.first), tol)
    );  
    REQUIRE_THAT(
        static_cast<double>(radii.second),
        Catch::Matchers::WithinAbs(static_cast<double>(R), tol)
    ); 
}

