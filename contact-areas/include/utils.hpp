/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/16/2024
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <map>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "products.hpp"

/**
 * Convert a (Eigen) vector of coefficients for a univariate polynomial into
 * a map.
 *
 * The map assumes the existence of N variables, with the indicated variable 
 * being that with the given coefficients.
 */
template <typename T, int N>
std::map<std::array<int, N>, T> vectorToMap(const Ref<const Matrix<T, Dynamic, 1> >& coefs,
                                            const int idx)
{
    std::map<std::array<int, N>, T> coefmap; 
    for (int i = 0; i < coefs.size(); ++i)
    {
        std::array<int, N> p;
        for (int j = 0; j < N; ++j)
        {
            if (j == idx)
                p[j] = i;
            else
                p[j] = 0;
        }
        coefmap.insert({p, coefs(i)});
    }
    return coefmap;
}

/**
 * Convert a (Eigen) matrix of coefficients for a bivariate polynomial into 
 * a map.
 *
 * The map assumes the existence of N variables, with the indicated variables
 * being those with the given coefficients.
 */
template <typename T, int N>
std::map<std::array<int, N>, T> matrixToMap(const Ref<const Matrix<T, Dynamic, Dynamic> >& coefs,
                                            const int idx1, const int idx2)
{
    std::map<std::array<int, N>, T> coefmap;
    for (int i = 0; i < coefs.rows(); ++i)
    {
        for (int j = 0; j < coefs.cols(); ++j)
        {
            std::array<int, N> p;
            for (int k = 0; k < N; ++k)
            {
                if (k == idx1)
                    p[k] = i;
                else if (k == idx2)
                    p[k] = j;
                else
                    p[k] = 0;
            }
            coefmap.insert({p, coefs(i, j)});
        }
    }
    return coefmap;
}

/**
 * Return the polynomial xi^d - 1, for some index i = 0, ..., N - 1, as a 
 * multivariate polynomial with N variables. 
 */
template <typename T, int N>
MultivariatePolynomial<T, N> polynomialOfUnity(const int idx, const int d)
{
    std::array<int, N> p, q;
    for (int i = 0; i < N; ++i)
    {
        p[i] = 0;
        q[i] = 0;
    }
    q[idx] = d;
    std::map<std::array<int, N>, std::complex<T> > coefmap;
    coefmap.insert({p, -1});
    coefmap.insert({q, 1});
    
    return MultivariatePolynomial<T, N>(coefmap);
}

/**
 * Return the polynomials xi^N - 1, for i = 0, ..., N - 1, as a polynomial
 * system.
 */
template <typename T, int N>
std::array<MultivariatePolynomial<T, N>, N> polynomialsOfUnity()
{
    std::array<MultivariatePolynomial<T, N>, N> polynomials;
    for (int i = 0; i < N; ++i)
        polynomials[i] = polynomialOfUnity<T, N>(i, N);

    return polynomials;
}

/**
 * Return the n-th roots of unity as a complex vector.
 */
template <typename T>
Matrix<std::complex<T>, Dynamic, 1> rootsOfUnity(const int n)
{
    Matrix<std::complex<T>, Dynamic, 1> roots(n); 
    for (int i = 0; i < n; ++i)
    {
        T a = std::cos(boost::math::constants::two_pi<T>() * i / n);
        T b = std::sin(boost::math::constants::two_pi<T>() * i / n);
        roots(i) = std::complex<T>(a, b); 
    }

    return roots;
}

/**
 * Return the roots of the polynomial system (x1^n - 1, ..., xk^n - 1)
 * as a complex matrix.
 *
 * Each row is a root (dimension n) of the system.
 */
template <typename T>
Matrix<std::complex<T>, Dynamic, Dynamic> rootsOfUnity(const int n, const int k)
{
    // First get the n-th roots of unity
    Matrix<std::complex<T>, Dynamic, 1> roots_single = rootsOfUnity<T>(n);

    // Then generate all possible k-tuples of the integers 0, ..., n - 1
    std::vector<std::vector<int> > ranges; 
    for (int i = 0; i < k; ++i)
    {
        std::vector<int> range;
        for (int j = 0; j < n; ++j)
            range.push_back(j);
        ranges.push_back(range);
    }
    std::vector<std::vector<int> > product = getProduct(ranges);

    // Get all k-fold combinations of the n-th roots of unity
    int nroots = 1; 
    for (int i = 0; i < k; ++i)
        nroots *= n;
    Matrix<std::complex<T>, Dynamic, Dynamic> roots(nroots, k);
    for (int i = 0; i < nroots; ++i)
    {
        std::vector<int> p = product[i];
        for (int j = 0; j < k; ++j)
            roots(i, j) = roots_single(p[j]);
    }

    return roots;
}

#endif
