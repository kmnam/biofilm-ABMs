/**
 * A lightweight implementation of univariate polynomials.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/4/2025
 */

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

using namespace Eigen;

using boost::multiprecision::abs; 
using boost::multiprecision::sin;
using boost::multiprecision::cos;
using boost::multiprecision::real; 

/**
 * A simple univariate polynomial class with complex-valued coefficients.
 *
 * This class assumes Boost.Multiprecision coefficient types. 
 */
template <int N>
class HighPrecisionPolynomial
{
    typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> >  RealType; 
    typedef boost::multiprecision::number<boost::multiprecision::mpc_complex_backend<N> > ComplexType;

    private:
        // Degree of the polynomial
        int degree;

        // Vector of coefficients in increasing order of degree
        Matrix<ComplexType, Dynamic, 1> coefs; 

    public:
        /**
         * Trivial constructor for the zero polynomial. 
         */
        HighPrecisionPolynomial()
        {
            this->degree = 0;
            this->coefs = Matrix<ComplexType, Dynamic, 1>::Zero(1);
        }

        /**
         * Constructor with input real-valued coefficients.
         *
         * @param coefs Input coefficients. 
         */
        HighPrecisionPolynomial(const Ref<const Matrix<RealType, Dynamic, 1> >& coefs)
        {
            // Identify the degree from the highest-degree nonzero coefficient
            this->degree = coefs.size() - 1;
            while (coefs(this->degree) == static_cast<RealType>(0) && this->degree > 0)
                this->degree--;
            
            this->coefs = coefs.head(this->degree + 1).template cast<ComplexType>();
        }

        /**
         * Constructor with input complex-valued coefficients.
         *
         * @param coefs Input coefficients.
         */
        HighPrecisionPolynomial(const Ref<const Matrix<ComplexType, Dynamic, 1> >& coefs)
        {
            // Identify the degree from the highest-degree nonzero coefficient
            this->degree = coefs.size() - 1;
            while (coefs(this->degree) == static_cast<RealType>(0) && this->degree > 0)
                this->degree--;
            
            this->coefs = coefs.head(this->degree + 1);
        }

        /**
         * Trivial destructor.
         */
        ~HighPrecisionPolynomial()
        {
        }

        /**
         * Return the coefficients of the polynomial.
         */
        Matrix<ComplexType, Dynamic, 1> getCoefs()
        {
            return this->coefs;
        }

        /**
         * Evaluate the polynomial at the given complex value with Horner's
         * method.
         *
         * @param x Input value for variable. 
         * @returns Polynomial value. 
         */
        ComplexType eval(const ComplexType x)
        {
            ComplexType y = this->coefs(this->degree);
            for (int i = this->degree - 1; i >= 0; i--)
                y = this->coefs(i) + x * y;

            return y;
        }

        /**
         * Get the derivative of the polynomial.
         *
         * @returns Derivative polynomial. 
         */
        HighPrecisionPolynomial<N> deriv()
        {
            // If the polynomial has degree zero, return the zero polynomial
            if (this->degree == 0)
            {
                return HighPrecisionPolynomial<N>();
            }
            // Otherwise, differentiate the polynomial term-by-term
            else
            {
                Matrix<ComplexType, Dynamic, 1> dcoefs(this->degree);
                for (int i = 1; i <= this->degree; ++i)
                    dcoefs(i - 1) = this->coefs(i) * static_cast<RealType>(i);
                return HighPrecisionPolynomial<N>(dcoefs);
            }
        }

        /**
         * Solve for the roots of the polynomial by computing the eigenvalues
         * of the corresponding companion matrix.
         *
         * @param real If true, the coefficients are assumed to be real.
         * @returns Vector of polynomial roots.  
         */
        Matrix<ComplexType, Dynamic, 1> solveCompanion(const bool real = false)
        {
            if (real)
            {
                // Define the companion matrix
                Matrix<RealType, Dynamic, Dynamic> companion
                    = Matrix<RealType, Dynamic, Dynamic>::Zero(this->degree, this->degree);
                companion(Eigen::seq(1, this->degree - 1), Eigen::seq(0, this->degree - 2))
                    = Matrix<RealType, Dynamic, Dynamic>::Identity(this->degree - 1, this->degree - 1);
                RealType lead = real(this->coefs(this->degree));
                for (int i = 0; i < this->degree; ++i)
                    companion(i, this->degree - 1) = real(-this->coefs(i)) / lead;

                // Get the eigenvalues of the companion matrix
                EigenSolver<Matrix<RealType, Dynamic, Dynamic> > es; 
                es.compute(companion, false);

                return es.eigenvalues();
            }
            else
            {
                // Define the companion matrix
                Matrix<ComplexType, Dynamic, Dynamic> companion
                    = Matrix<ComplexType, Dynamic, Dynamic>::Zero(this->degree, this->degree);
                companion(Eigen::seq(1, this->degree - 1), Eigen::seq(0, this->degree - 2))
                    = Matrix<ComplexType, Dynamic, Dynamic>::Identity(this->degree - 1, this->degree - 1);
                ComplexType lead = this->coefs(this->degree);
                companion(Eigen::all, this->degree - 1) = -this->coefs.head(this->degree) / lead;

                // Get the eigenvalues of the companion matrix
                ComplexEigenSolver<Matrix<ComplexType, Dynamic, Dynamic> > es; 
                es.compute(companion, false);

                return es.eigenvalues();
            }
        }

        /**
         * Solve for the roots of the polynomial with the Durand-Kerner method.
         *
         * @param tol Tolerance for judging whether the difference between 
         *            consecutive root approximations is zero.
         * @returns Vector of polynomial roots. 
         */
        Matrix<ComplexType, Dynamic, 1> solveDurandKerner(const RealType tol)
        {
            // Initialize the roots as the n-th roots of unity, where 
            // n is the degree of the polynomial 
            Matrix<ComplexType, Dynamic, 1> roots(this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                RealType a = cos(i * boost::math::constants::two_pi<RealType>() / this->degree);
                RealType b = sin(i * boost::math::constants::two_pi<RealType>() / this->degree);
                roots(i) = ComplexType(a, b);
            }

            // Store the differences between the roots in a matrix
            Matrix<ComplexType, Dynamic, Dynamic> diffs
                = Matrix<ComplexType, Dynamic, Dynamic>::Zero(this->degree, this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                for (int j = i + 1; j < this->degree; ++j)
                {
                    diffs(i, j) = roots(i) - roots(j); 
                    diffs(j, i) = -diffs(i, j);
                }
            }

            // Iteratively apply the Durand-Kerner method
            RealType max_update = std::numeric_limits<RealType>::infinity();
            while (max_update > tol)
            {
                Matrix<ComplexType, Dynamic, 1> updates(this->degree);

                // Compute the Durand-Kerner update for each root
                for (int i = 0; i < this->degree; ++i)
                {
                    // Evaluate the polynomial at the current value of the
                    // i-th root
                    ComplexType val_i = this->eval(roots(i)); 

                    // Get the product of the differences between the i-th
                    // root and every other root
                    ComplexType dprod = 1.0;
                    for (int j = 0; j < this->degree; ++j)
                    {
                        if (j != i)
                            dprod *= diffs(i, j);
                    }

                    // Compute the Durand-Kerner update
                    updates(i) = val_i / (this->coefs(this->degree) * dprod);
                }

                // Identify the update with the largest magnitude
                max_update = 0.0; 
                for (int i = 0; i < this->degree; ++i)
                {
                    if (max_update < abs(updates(i)))
                        max_update = abs(updates(i)); 
                }

                // Update the roots 
                roots -= updates;

                // Update the pairwise root differences
                for (int i = 0; i < this->degree; ++i)
                {
                    for (int j = i + 1; j < this->degree; ++j)
                    {
                        diffs(i, j) = roots(i) - roots(j); 
                        diffs(j, i) = -diffs(i, j);
                    }
                }
            }

            return roots;
        }

        /**
         * Solve for the roots of the polynomial with Aberth's method.
         *
         * @param tol Tolerance for judging whether the difference between 
         *            consecutive root approximations is zero.
         * @returns Vector of polynomial roots. 
         */
        Matrix<ComplexType, Dynamic, 1> solveAberth(const RealType tol)
        {
            // Get the derivative of the polynomial 
            HighPrecisionPolynomial<N> deriv = this->deriv();

            // Initialize the roots as the n-th roots of unity, where 
            // n is the degree of the polynomial 
            Matrix<ComplexType, Dynamic, 1> roots(this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                RealType a = cos(i * boost::math::constants::two_pi<RealType>() / this->degree);
                RealType b = sin(i * boost::math::constants::two_pi<RealType>() / this->degree);
                roots(i) = ComplexType(a, b);
            }

            // Store the differences between the roots in a matrix
            Matrix<ComplexType, Dynamic, Dynamic> diffs
                = Matrix<ComplexType, Dynamic, Dynamic>::Zero(this->degree, this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                for (int j = i + 1; j < this->degree; ++j)
                {
                    diffs(i, j) = roots(i) - roots(j); 
                    diffs(j, i) = -diffs(i, j);
                }
            }

            // Iteratively apply Aberth's method
            RealType max_update = std::numeric_limits<RealType>::infinity();
            while (max_update > tol)
            {
                Matrix<ComplexType, Dynamic, 1> updates(this->degree);

                // Compute the Aberth update for each root 
                for (int i = 0; i < this->degree; ++i)
                {
                    // Evaluate the polynomial and its derivative at the 
                    // current value of the i-th root
                    ComplexType val_i = this->eval(roots(i)); 
                    ComplexType dval_i = deriv.eval(roots(i));
                    ComplexType ratio_i = val_i / dval_i;

                    // Get the sum of one over the difference between the i-th
                    // root and every other root
                    ComplexType dsum = 0;
                    for (int j = 0; j < this->degree; ++j)
                    {
                        if (j != i)
                            dsum += 1.0 / diffs(i, j);
                    }

                    // Compute the Aberth update 
                    updates(i) = ratio_i / (1.0 - ratio_i * dsum);
                }

                // Identify the update with the largest magnitude 
                max_update = 0.0; 
                for (int i = 0; i < this->degree; ++i)
                {
                    if (max_update < abs(updates(i)))
                        max_update = abs(updates(i)); 
                }

                // Update the roots 
                roots -= updates;

                // Update the pairwise root differences
                for (int i = 0; i < this->degree; ++i)
                {
                    for (int j = i + 1; j < this->degree; ++j)
                    {
                        diffs(i, j) = roots(i) - roots(j); 
                        diffs(j, i) = -diffs(i, j);
                    }
                }
            }

            return roots;
        }
};

#endif
