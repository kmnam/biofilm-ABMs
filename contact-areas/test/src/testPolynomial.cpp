/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/14/2024
 */

#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include "../../include/polynomials.hpp"

using namespace Eigen;

/**
 * Solve for the roots of the quadratic polynomial, (x - 1) * (x - 2).
 */
void testSolveQuadratic()
{
    Matrix<double, Dynamic, 1> coefs(3); 
    coefs << 2, -3, 1;
    Polynomial<double> p(coefs);
    std::cout << p.solveCompanion() << std::endl;
    std::cout << p.solveCompanion(true) << std::endl;
    std::cout << p.solveDurandKerner(1e-8) << std::endl;
    std::cout << p.solveAberth(1e-8) << std::endl;
}

/**
 * Solve for the roots of the sextic polynomial given by: 
 *
 * (x + 8) * (x + 5) * (x + 3) * (x - 2) * (x - 3) * (x - 7).
 */
void testSolveSextic()
{
    Matrix<double, Dynamic, 1> coefs(7);
    coefs << -5040, 1602, 1127, -214, -72, 4, 1;
    Polynomial<double> p(coefs); 
    std::cout << p.solveCompanion() << std::endl;
    std::cout << p.solveCompanion(true) << std::endl;
    std::cout << p.solveDurandKerner(1e-8) << std::endl;
    std::cout << p.solveAberth(1e-8) << std::endl;
}

/**
 * Evaluate the circle polynomial, x^2 + y^2 - 1, at y == 0.5 and x == 0.2.
 */
void testEvalCircle()
{
    std::map<std::array<int, 2>, std::complex<double> > coefs;
    coefs.insert({{0, 0}, std::complex<double>(-1, 0)});
    coefs.insert({{2, 0}, std::complex<double>(1, 0)});
    coefs.insert({{0, 2}, std::complex<double>(1, 0)});
    MultivariatePolynomial<double, 2> p(coefs);
    std::cout << p.toString() << std::endl;
    std::cout << p.eval(1, 0.5).toString() << std::endl;
    std::array<std::complex<double>, 2> xy { 0.2, 0.5 };
    std::cout << p.eval(xy) << std::endl;
}

/**
 * Evaluate the sphere polynomial, x^2 + y^2 + z^2 - 1, at z == 0.2 and
 * eliminate z. 
 */
void testEvalElimSphere()
{
    std::map<std::array<int, 3>, std::complex<double> > coefs; 
    coefs.insert({{0, 0, 0}, std::complex<double>(-1, 0)});
    coefs.insert({{2, 0, 0}, std::complex<double>(1, 0)}); 
    coefs.insert({{0, 2, 0}, std::complex<double>(1, 0)});
    coefs.insert({{0, 0, 2}, std::complex<double>(1, 0)});
    MultivariatePolynomial<double, 3> p(coefs); 
    std::cout << p.toString() << std::endl; 
    std::cout << p.evalElim(2, 0.2).toString() << std::endl;
}

/**
 * Differentiate the circle polynomial, x^2 + y^2 - 1, with respect to x and y.
 */
void testDerivCircle()
{
    std::map<std::array<int, 2>, std::complex<double> > coefs;
    coefs.insert({{0, 0}, std::complex<double>(-1, 0)});
    coefs.insert({{2, 0}, std::complex<double>(1, 0)});
    coefs.insert({{0, 2}, std::complex<double>(1, 0)});
    MultivariatePolynomial<double, 2> p(coefs);
    std::cout << p.toString() << std::endl;
    std::cout << p.deriv(0).toString() << std::endl;
    std::cout << p.deriv(1).toString() << std::endl;
}

/**
 * Homogenize the circle polynomial, (x - 2)^2 + (y - 3)^2 - 1, which expands
 * to x^2 - 4*x + y^2 - 6*y + 12.
 */
void testHomogenizeCircle()
{
    std::map<std::array<int, 2>, std::complex<double> > coefs;
    coefs.insert({{0, 0}, std::complex<double>(12, 0)});
    coefs.insert({{1, 0}, std::complex<double>(-4, 0)});
    coefs.insert({{0, 1}, std::complex<double>(-6, 0)});
    coefs.insert({{2, 0}, std::complex<double>(1, 0)});
    coefs.insert({{0, 2}, std::complex<double>(1, 0)});
    MultivariatePolynomial<double, 2> p(coefs);
    std::cout << p.toString() << std::endl;
    std::cout << p.homogenize().toString() << std::endl;
}

int main()
{
    testSolveQuadratic();
    testSolveSextic();
    testEvalCircle();
    testEvalElimSphere();
    testDerivCircle();
    testHomogenizeCircle();
}
