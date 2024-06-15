/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/15/2024
 */

#ifndef KIHARA_GBK_POTENTIAL_FORCES_HPP
#define KIHARA_GBK_POTENTIAL_FORCES_HPP

#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>

using std::pow; 
using boost::multiprecision::pow;
using std::min;
using boost::multiprecision::min;

/**
 * TODO Write this docstring  
 */
template <typename T>
Matrix<T, 2, 4> attractiveForcesKihara2D(const Ref<const Matrix<T, 2, 1> >& r1, 
                                         const Ref<const Matrix<T, 2, 1> >& n1, 
                                         const T half_l1,
                                         const Ref<const Matrix<T, 2, 1> >& r2, 
                                         const Ref<const Matrix<T, 2, 1> >& n2, 
                                         const T half_l2,
                                         const Ref<const Matrix<T, 2, 1> >& d12, 
                                         const T s, const T t)
{
    Matrix<T, 2, 4> dEdq;

    // Normalize the distance vector 
    T dist = d12.norm(); 
    Matrix<T, 2, 1> d12n = d12 / dist; 

    // Get the derivative of the potential with respect to the cell-cell
    // distance
    T deriv = 6.0 / pow(dist, 7.0);

    // Use the above to get the partial derivative of the potential with
    // respect to each coordinate
    //
    // Partial derivatives w.r.t cell 1 center 
    dEdq(0, Eigen::seq(0, 1)) = -deriv * d12n; 

    // Partial derivatives w.r.t cell 1 orientation 
    dEdq(0, Eigen::seq(2, 3)) = -deriv * d12n * s; 

    // Partial derivatives w.r.t cell 2 center 
    dEdq(1, Eigen::seq(0, 1)) = -dEdq(0, Eigen::seq(0, 1)); 

    // Partial derivatives w.r.t cell 2 orientation 
    dEdq(1, Eigen::seq(2, 3)) = deriv * d12n * t;

    return dEdq; 
}

/**
 * TODO Write
 *
 * This function returns chi^2. 
 */
template <typename T>
T squaredAspectRatioParam(const T half_l1, const T half_l2, const T R)
{
    T total_l1 = half_l1 + R; 
    T total_l2 = half_l2 + R; 
    T width = R;
    T total_l1_sq = total_l1 * total_l1; 
    T total_l2_sq = total_l2 * total_l2; 
    T width_sq = width * width; 
    T chi2_term1 = total_l1_sq - width_sq; 
    T chi2_term2 = total_l2_sq - width_sq;
    T chi2_term3 = total_l2_sq + width_sq; 
    T chi2_term4 = total_l1_sq + width_sq;
    
    return (chi2_term1 * chi2_term2) / (chi2_term3 * chi2_term4);
}

/**
 * TODO Write
 */
template <typename T>
std::pair<T, Matrix<T, 2, 4> > anisotropyParamsGBK1(const Ref<const Matrix<T, 2, 1> >& r1, 
                                                    const Ref<const Matrix<T, 2, 1> >& n1,
                                                    const T half_l1,
                                                    const Ref<const Matrix<T, 2, 1> >& r2, 
                                                    const Ref<const Matrix<T, 2, 1> >& n2,
                                                    const T half_l2, const T R,
                                                    const T exp)
{
    // Compute the anisotropy parameter
    T chi2 = squaredAspectRatioParam<T>(half_l1, half_l2, R);
    T n1_dot_n2 = n1.dot(n2);
    T arg = 1.0 - chi2 * n1_dot_n2 * n1_dot_n2;
    T eps = pow(arg, -0.5);

    // Get the partial derivative of the anisotropy parameter w.r.t each 
    // coordinate 
    Matrix<T, 2, 4> derivs = Matrix<T, 2, 4>::Zero();

    // Compute the prefactor for each partial derivative 
    T prefactor = exp * pow(eps, exp - 1) * chi2 * pow(arg, -1.5) * n1_dot_n2; 

    // The partial derivatives w.r.t the cell center coordinates are
    // uniformly zero 
    //
    // Partial derivative w.r.t x-coordinate of cell 1 orientation
    derivs(0, 2) = prefactor * n2(0);

    // Partial derivative w.r.t y-coordinate of cell 1 orientation 
    derivs(0, 3) = prefactor * n2(1); 

    // Partial derivative w.r.t x-coordinate of cell 2 orientation 
    derivs(1, 2) = prefactor * n1(0); 

    // Partial derivative w.r.t y-coordinate of cell 2 orientation 
    derivs(1, 3) = prefactor * n1(1); 

    return std::make_pair(eps, derivs); 
}

/**
 * TODO Write
 */
template <typename T>
std::pair<T, Matrix<T, 2, 4> > anisotropyParamsGBK2(const Ref<const Matrix<T, 2, 1> >& r1, 
                                                    const Ref<const Matrix<T, 2, 1> >& n1,
                                                    const T half_l1,
                                                    const Ref<const Matrix<T, 2, 1> >& r2, 
                                                    const Ref<const Matrix<T, 2, 1> >& n2,
                                                    const T half_l2, const T R,
                                                    const Ref<const Matrix<T, 2, 1> >& d12,
                                                    const T exp, const T kappa0)
{
    // Compute the anisotropy parameter ... 
    //
    // First compute the well-depth parameters \kappa and \chi'
    T kappa = pow(kappa0 * ((min(half_l1, half_l2) / R) + 1), 1 / exp); 
    T chi = (kappa - 1) / (kappa + 1);

    // Compute the vector r12 from r1 to r2 and its normalized vector 
    Matrix<T, 2, 1> r12 = r2 - r1;
    T r12_norm = r12.norm(); 
    Matrix<T, 2, 1> r12n = r12 / r12_norm; 

    // Compute the dot products (n1, n2), (r12n, n1), and (r12n, n2)
    T n1_dot_n2 = n1.dot(n2);
    T r12n_dot_n1 = r12n.dot(n1); 
    T r12n_dot_n2 = r12n.dot(n2);

    // Compute the anisotropy parameter itself 
    T r12n_dot_sum = r12n_dot_n1 + r12n_dot_n2; 
    T r12n_dot_diff = r12n_dot_n1 - r12n_dot_n2;
    T numer1 = r12n_dot_sum * r12n_dot_sum; 
    T numer2 = r12n_dot_diff * r12n_dot_diff; 
    T denom1 = 1 + chi * n1_dot_n2; 
    T denom2 = 1 - chi * n1_dot_n2; 
    T eps = 1.0 - (chi / 2) * (numer1 / denom1 + numer2 / denom2);

    // Compute the partial derivatives of (r12n, n1) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 4> derivs1 = Matrix<T, 2, 4>::Zero();
    derivs1(0, 0) = (n1(0) * (-1 + r12n(0) * r12n(0)) + n1(1) * (r12n(0) * r12n(1))) / r12_norm;
    derivs1(0, 1) = (n1(0) * (r12n(1) * r12n(0)) + n1(1) * (-1 + r12n(1) * r12n(1))) / r12_norm; 
    derivs1(0, 2) = r12n(0); 
    derivs1(0, 3) = r12n(1);
    derivs1(1, 0) = -derivs1(0, 0); 
    derivs1(1, 1) = -derivs1(0, 1); 

    // Compute the partial derivatives of (r12n, n2) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 4> derivs2 = Matrix<T, 2, 4>::Zero(); 
    derivs2(0, 0) = (n2(0) * (-1 + r12n(0) * r12n(0)) + n2(1) * (r12n(0) * r12n(1))) / r12_norm; 
    derivs2(0, 1) = (n2(0) * (r12n(1) * r12n(0)) + n2(1) * (-1 + r12n(1) * r12n(1))) / r12_norm; 
    derivs2(1, 0) = -derivs2(0, 0); 
    derivs2(1, 1) = -derivs2(0, 1);
    derivs2(1, 2) = r12n(0); 
    derivs2(1, 3) = r12n(1);

    // Compute the partial derivatives of ((r12n, n1) + (r12n, n2))^2 with
    // respect to each generalized coordinate 
    Matrix<T, 2, 4> derivs_sum2 = 2 * r12n_dot_sum * (derivs1 + derivs2);

    // Compute the partial derivatives of ((r12n, n1) - (r12n, n1))^2 with 
    // respect to each generalized coordinate 
    Matrix<T, 2, 4> derivs_diff2 = 2 * r12n_dot_diff * (derivs1 - derivs2); 

    // Compute the partial derivative of (n1, n2) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 4> derivs3 = Matrix<T, 2, 4>::Zero(); 
    derivs3(0, 2) = n2(0);
    derivs3(0, 3) = n2(1);
    derivs3(1, 2) = n1(0); 
    derivs3(1, 3) = n1(1);

    // Compute the partial derivative of the anisotropy parameter with
    // respect to each generalized coordinate
    Matrix<T, 2, 4> term1 = ((denom1 * derivs_sum2) - (numer1 * chi * derivs3)) / (denom1 * denom1);
    Matrix<T, 2, 4> term2 = ((denom2 * derivs_diff2) + (numer2 * chi * derivs3)) / (denom2 * denom2); 
    Matrix<T, 2, 4> derivs = -0.5 * chi * (term1 + term2); 

    return std::make_pair(eps, exp * pow(eps, exp - 1) * derivs); 
}

/**
 * TODO Write
 */
template <typename T>
Matrix<T, 2, 4> attractiveForcesGBK2D(const Ref<const Matrix<T, 2, 1> >& r1, 
                                      const Ref<const Matrix<T, 2, 1> >& n1, 
                                      const T half_l1, 
                                      const Ref<const Matrix<T, 2, 1> >& r2, 
                                      const Ref<const Matrix<T, 2, 1> >& n2, 
                                      const T half_l2, const T R, 
                                      const Ref<const Matrix<T, 2, 1> >& d12, 
                                      const T s, const T t, const T exp1,
                                      const T exp2, const T kappa0)
{
    // Get the anisotropy parameters and their partial derivatives 
    auto result1 = anisotropyParamsGBK1<T>(
        r1, n1, half_l1, r2, n2, half_l2, R, exp1
    );
    T eps1 = result1.first; 
    Matrix<T, 2, 4> deps1 = result1.second; 
    auto result2 = anisotropyParamsGBK2<T>(
        r1, n1, half_l1, r2, n2, half_l2, R, d12, exp2, kappa0
    );
    T eps2 = result2.first; 
    Matrix<T, 2, 4> deps2 = result2.second; 

    // Use the product rule to get the partial derivatives of the combined
    // anisotropy parameter
    Matrix<T, 2, 4> deps_combined = eps1 * deps2 + eps2 * deps1;

    // Get the attractive Kihara potential and its partial derivatives 
    T dist = d12.norm(); 
    T potential = -pow(dist, -6.0);
    Matrix<T, 2, 4> forces = attractiveForcesKihara2D<T>(
        r1, n1, half_l1, r2, n2, half_l2, d12, s, t
    ); 

    // Use the product rule to get the overall partial derivatives 
    return potential * deps_combined + eps1 * eps2 * forces; 
}

#endif 
