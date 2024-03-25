/**
 * A lightweight implementation of univariate and multivariate polynomials.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/24/2024
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
#include "products.hpp"

using namespace Eigen;

/**
 * A simple univariate polynomial class with complex-valued coefficients.
 */
template <typename RealType>
class Polynomial
{
    typedef std::complex<RealType> ComplexType;

    private:
        // Degree of the polynomial
        int degree;

        // Vector of coefficients in increasing order of degree
        Matrix<ComplexType, Dynamic, 1> coefs; 

    public:
        /**
         * Trivial constructor for the zero polynomial. 
         */
        Polynomial()
        {
            this->degree = 0;
            this->coefs = Matrix<ComplexType, Dynamic, 1>::Zero(1);
        }

        /**
         * Constructor with input real-valued coefficients.
         *
         * @param coefs Input coefficients. 
         */
        Polynomial(const Ref<const Matrix<RealType, Dynamic, 1> >& coefs)
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
        Polynomial(const Ref<const Matrix<ComplexType, Dynamic, 1> >& coefs)
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
        ~Polynomial()
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
        Polynomial<RealType> deriv()
        {
            // If the polynomial has degree zero, return the zero polynomial
            if (this->degree == 0)
            {
                return Polynomial<RealType>();
            }
            // Otherwise, differentiate the polynomial term-by-term
            else
            {
                Matrix<ComplexType, Dynamic, 1> dcoefs(this->degree);
                for (int i = 1; i <= this->degree; ++i)
                    dcoefs(i - 1) = this->coefs(i) * static_cast<RealType>(i);
                return Polynomial<RealType>(dcoefs);
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
                RealType lead = std::real(this->coefs(this->degree));
                for (int i = 0; i < this->degree; ++i)
                    companion(i, this->degree - 1) = std::real(-this->coefs(i)) / lead;

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
                RealType a = std::cos(i * boost::math::constants::two_pi<RealType>() / this->degree);
                RealType b = std::sin(i * boost::math::constants::two_pi<RealType>() / this->degree);
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
                max_update = updates.cwiseAbs().maxCoeff();

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
            Polynomial<RealType> deriv = this->deriv();

            // Initialize the roots as the n-th roots of unity, where 
            // n is the degree of the polynomial 
            Matrix<ComplexType, Dynamic, 1> roots(this->degree);
            for (int i = 0; i < this->degree; ++i)
            {
                RealType a = std::cos(i * boost::math::constants::two_pi<RealType>() / this->degree);
                RealType b = std::sin(i * boost::math::constants::two_pi<RealType>() / this->degree);
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
                max_update = updates.cwiseAbs().maxCoeff();

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

/**
 * A simple multivariate polynomial class.
 */
template <typename RealType, int NVariables>
class MultivariatePolynomial
{
    typedef std::complex<RealType> ComplexType;
    typedef std::array<int, NVariables> MonomialType;
    typedef std::map<MonomialType, ComplexType> CoefMapType;

    private:
        // Degree of the polynomial in each variable
        MonomialType degrees;

        // Tensor of polynomial coefficients as a map
        CoefMapType coefs;

    public:
        /**
         * Trivial constructor for the zero polynomial in the given number
         * of variables. 
         */
        MultivariatePolynomial()
        {
            for (int i = 0; i < NVariables; ++i)
                this->degrees[i] = 0;
            this->coefs.insert({this->degrees, ComplexType(0, 0)});
        }

        /**
         * Constructor for a multivariate polynomial with complex-valued
         * coefficients.
         *
         * @param Input map of polynomial coefficients. 
         */
        MultivariatePolynomial(CoefMapType& coefs)
        {
            // Identify the degrees of the polynomial in each variable
            for (int i = 0; i < NVariables; ++i)
                this->degrees[i] = 0;
            for (const auto& p : coefs)
            {
                // For each monomial, if the coefficient is nonzero, 
                // update each degree as necessary
                if (p.second != ComplexType(0, 0))
                {
                    for (int i = 0; i < NVariables; ++i)
                    {
                        if (p.first[i] > this->degrees[i])
                            this->degrees[i] = p.first[i];
                    }
                }
            }

            // Given the degrees of the variables, fill in this->coefs
            std::vector<std::vector<int> > ranges; 
            for (int i = 0; i < NVariables; ++i)
            {
                std::vector<int> range; 
                for (int j = 0; j <= this->degrees[i]; ++j)
                    range.push_back(j);
                ranges.push_back(range);
            }
            std::vector<std::vector<int> > powers = getProduct(ranges);
            for (auto&& power : powers)
            {
                MonomialType p;
                for (int i = 0; i < NVariables; ++i)
                    p[i] = power[i];
                if (coefs.find(p) == coefs.end())
                    this->coefs.insert({p, ComplexType(0, 0)});
                else 
                    this->coefs.insert({p, coefs[p]});
            }
        }

        /**
         * Trivial destructor.
         */
        ~MultivariatePolynomial()
        {
        }

        /**
         * Return the degrees of the polynomial.
         *
         * @returns Array of polynomial degrees. 
         */
        MonomialType getDegrees()
        {
            return this->degrees;
        }

        /**
         * Return the coefficient of the given monomial.
         *
         * @param p Input monomial. 
         * @returns Coefficient corresponding to the input monomial.
         */
        ComplexType getCoef(MonomialType& p)
        {
            try
            {
                return this->coefs.at(p);
            }
            catch (const std::out_of_range& e)
            {
                throw;
            }
        }

        /**
         * Return the coefficients of the polynomial.
         *
         * @returns Map of polynomial coefficients. 
         */
        CoefMapType& getCoefs()
        {
            return this->coefs;
        }

        /**
         * Evaluate the polynomial with the given value for the indicated
         * variable and return the reduced polynomial.
         *
         * The indicated variable is eliminated from the polynomial. An
         * exception is raised if the indicated variable does not exist 
         * in the polynomial (i.e., if idx >= NVariables), or if there 
         * are fewer than two variables to begin with.
         *
         * @param idx Index of variable to evaluate. 
         * @param value Input value for indicated variable. 
         * @returns Evaluated polynomial with one fewer variable. 
         */
        MultivariatePolynomial<RealType, NVariables - 1> eval(const int idx,
                                                              const ComplexType value)
        {
            // The indicated variable has to be less than the number of variables
            if (idx >= NVariables)
                throw std::invalid_argument(
                    "Invalid variable specified for polynomial evaluation"
                );

            // The polynomial should have two or more variables
            if (NVariables < 2)
                throw std::runtime_error(
                    "Cannot evaluate and eliminate variable in polynomial with < 2 variables"
                );

            // Define a new map of coefficients (with the indicated variable
            // eliminated) 
            std::map<std::array<int, NVariables - 1>, ComplexType> coefs_reduced;
           
            // For each term in the polynomial, evaluate the term and update
            // the new map of coefficients
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second;
                ComplexType new_coef = coef * std::pow(value, p[idx]);
                std::array<int, NVariables - 1> q;
                for (int i = 0; i < NVariables; ++i)
                {
                    if (i < idx)
                        q[i] = p[i];
                    else if (i > idx)
                        q[i - 1] = p[i];
                }
                if (coefs_reduced.find(q) == coefs_reduced.end())
                    coefs_reduced[q] = new_coef;
                else 
                    coefs_reduced[q] += new_coef;
            }

            return MultivariatePolynomial<RealType, NVariables - 1>(coefs_reduced);
        }

        /**
         * Evaluate the polynomial at the given vector of real values for all
         * the variables. 
         *
         * The returned value is a complex scalar.
         *
         * The evaluation is done term-by-term, as opposed to using a 
         * scheme like Horner's method, with the assumption that the 
         * polynomial is relatively low-degree.
         *
         * @param values Vector of values for each variable. 
         * @returns Polynomial value.
         */
        ComplexType eval(const Ref<const Matrix<RealType, NVariables, 1> >& values)
        {
            // Evaluate the polynomial, one term at a time
            ComplexType total(0, 0);
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second;
                ComplexType vars(1, 0); 
                for (int i = 0; i < NVariables; ++i)
                {
                    for (int j = 0; j < p[i]; ++j)
                    {
                        vars *= values(i);
                    }
                }
                total += coef * vars;
            }
            return total;
        }

        /**
         * Evaluate the polynomial at the given vector of complex values for
         * all the variables. 
         *
         * The returned value is a complex scalar.
         *
         * The evaluation is done term-by-term, as opposed to using a 
         * scheme like Horner's method, with the assumption that the 
         * polynomial is relatively low-degree.
         *
         * @param values Vector of values for each variable. 
         * @returns Polynomial value.
         */
        ComplexType eval(const Ref<const Matrix<ComplexType, NVariables, 1> >& values)
        {
            // Evaluate the polynomial, one term at a time
            ComplexType total(0, 0);
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second;
                ComplexType vars(1, 0); 
                for (int i = 0; i < NVariables; ++i)
                {
                    for (int j = 0; j < p[i]; ++j)
                    {
                        vars *= values(i);
                    }
                }
                total += coef * vars;
            }
            return total;
        }
        
        /**
         * Return a string representation of the polynomial.
         *
         * The variables are written as x0, x1, x2, ..., and powers are 
         * denoted by x^y.
         *
         * @returns String representation of the polynomial. 
         */
        std::string toString()
        {
            std::stringstream ss_total;

            // Run through each term in the polynomial (there should be 
            // at least one, even if it is zero) 
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second;

                // If the coefficient is nonzero ...
                if (std::abs(coef) != 0)
                {
                    // Write the product of variables to a stringstream 
                    std::stringstream ss_term; 
                    for (int i = 0; i < NVariables; ++i)
                    {
                        if (p[i] == 1)
                            ss_term << "x" << i << "*";
                        else if (p[i] > 1)
                            ss_term << "x" << i << "^" << p[i] << "*";
                    }

                    // Combine the coefficient with the product of variables
                    std::string vars = ss_term.str();
                    if (!vars.empty())
                    {
                        vars.pop_back();
                        ss_total << coef << "*" << vars << " + ";
                    }
                    else 
                    {
                        ss_total << coef << " + ";
                    }
                }
            }

            // Remove the last plus sign (if the string is nonempty)
            std::string outstr = ss_total.str();
            if (outstr.empty())
            {
                return "0";
            }
            else
            { 
                outstr.pop_back();
                outstr.pop_back();
                outstr.pop_back();
                return outstr;
            }
        }

        /**
         * Return the product of the polynomial with the given complex scalar.
         *
         * @param mult Complex scalar. 
         * @returns The product polynomial.
         */
        MultivariatePolynomial<RealType, NVariables> operator*(const ComplexType mult)
        {
            // Instantiate a new polynomial with updated coefficients 
            CoefMapType coefs; 
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second; 
                coefs.insert({p, mult * coef}); 
            }

            return MultivariatePolynomial<RealType, NVariables>(coefs); 
        }

        /**
         * Multiply the polynomial by the given complex scalar.
         *
         * @param mult Complex scalar.
         * @returns Reference to updated polynomial.
         */
        MultivariatePolynomial<RealType, NVariables>& operator*=(const ComplexType mult)
        {
            // Update each coefficient in the polynomial 
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second; 
                this->coefs[p] = mult * coef; 
            }
            
            return *this;
        }

        /**
         * Return the partial derivative of the polynomial in the indicated
         * variable.
         *
         * @param idx Index of variable to differentiate the polynomial with
         *            respect to. 
         * @returns The partial derivative polynomial. 
         */
        MultivariatePolynomial<RealType, NVariables> deriv(const int idx)
        {
            // If the polynomial has degree zero in x, return the zero 
            // polynomial
            if (this->degrees[idx] == 0)
            {
                return MultivariatePolynomial<RealType, NVariables>(); 
            }
            // Otherwise, differentiate the polynomial term-by-term
            else
            {
                // Set up a new map of coefficients 
                CoefMapType dcoefs; 

                // For each term in the polynomial ... 
                for (auto&& term : this->coefs)
                {
                    MonomialType power = term.first; 
                    ComplexType coef = term.second;

                    // If the term is constant in the chosen variable, 
                    // skip over that term
                    if (power[idx] != 0)
                    {
                        // Otherwise, there is a unique monomial that is 
                        // generated by differentiating the term
                        MonomialType dpower;
                        for (int i = 0; i < NVariables; ++i)
                        {
                            if (i == idx)
                                dpower[i] = power[idx] - 1;
                            else
                                dpower[i] = power[i];
                        }
                        ComplexType dcoef = coef * static_cast<RealType>(power[idx]);
                        dcoefs.insert({dpower, dcoef});
                    }
                }
                
                return MultivariatePolynomial<RealType, NVariables>(dcoefs);
            }
        }

        /**
         * Return a homogenization of the polynomial, containing an additional 
         * variable.
         *
         * @returns The homogenized polynomial.
         */
        MultivariatePolynomial<RealType, NVariables + 1> homogenize()
        {
            // Get the total degree of the polynomial
            int total_degree = 0;
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first; 
                ComplexType coef = term.second; 
                if (std::abs(coef) != 0)
                {
                    int psum = 0;
                    for (int i = 0; i < NVariables; ++i)
                        psum += p[i]; 
                    if (total_degree < psum)
                        total_degree = psum;
                }
            }

            // Determine the coefficients of the new polynomial 
            std::map<std::array<int, NVariables + 1>, ComplexType> new_coefs; 
            for (auto&& term : this->coefs)
            {
                MonomialType p = term.first;
                ComplexType coef = term.second;
                int psum = 0;
                std::array<int, NVariables + 1> q; 
                for (int i = 0; i < NVariables; ++i)
                {
                    q[i] = p[i];
                    psum += p[i];
                }
                q[NVariables] = total_degree - psum;
                new_coefs.insert({q, coef});
            }

            return MultivariatePolynomial<RealType, NVariables + 1>(new_coefs);
        }
};

/**
 * Multiply the given polynomial by the given complex scalar (on the left-hand
 * side). 
 */
template <typename RealType, int NVariables>
MultivariatePolynomial<RealType, NVariables> operator*(const std::complex<RealType> a,
                                                       MultivariatePolynomial<RealType, NVariables> f)
{
    return f * a;
}

#endif
