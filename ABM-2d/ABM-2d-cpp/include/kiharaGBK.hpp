/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/7/2024
 */

#ifndef KIHARA_GBK_POTENTIAL_FORCES_HPP
#define KIHARA_GBK_POTENTIAL_FORCES_HPP

#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>

using namespace Eigen;

using std::pow; 
using boost::multiprecision::pow;
using std::min;
using boost::multiprecision::min;

/* --------------------------------------------------------------------- //
 *                          AUXILIARY FUNCTIONS                          //
 * --------------------------------------------------------------------- */
/**
 * Compute the squared aspect ratio parameter in the first anisotropy 
 * parameter of the Gay-Berne-Kihara potential.  
 *
 * @param half_l1 Half of length of cell 1.
 * @param half_l2 Half of length of cell 2. 
 * @param Rcell Cell radius, excluding the EPS.
 * @returns Squared aspect ratio parameter. 
 */
template <typename T>
T squaredAspectRatioParam(const T half_l1, const T half_l2, const T Rcell)
{
    T total_l1 = half_l1 + Rcell; 
    T total_l2 = half_l2 + Rcell; 
    T width = Rcell;
    T total_l1_sq = total_l1 * total_l1; 
    T total_l2_sq = total_l2 * total_l2; 
    T width_sq = width * width; 
    T chi2_term1 = total_l1_sq - width_sq; 
    T chi2_term2 = total_l2_sq - width_sq;
    T chi2_term3 = total_l2_sq + width_sq; 
    T chi2_term4 = total_l1_sq + width_sq;
    
    return (chi2_term1 * chi2_term2) / (chi2_term3 * chi2_term4);
}

/* --------------------------------------------------------------------- //
 *                              POTENTIALS                               //
 * --------------------------------------------------------------------- */
/**
 * Compute the shifted Kihara potential between two neighboring cells in
 * arbitrary dimensions (2 or 3).
 *
 * @param dist Shortest distance from cell 1 to cell 2.
 * @param R Cell radius, including the EPS. 
 * @param exp Exponent in Kihara potential.
 * @param dmin Minimum distance at which the Kihara potential is nonzero.
 * @returns Shifted Kihara potential at the given cell-cell distance. 
 */
template <typename T>
T potentialKihara(const T dist, const T R, const T exp, const T dmin)
{
    // If the distance is less than dmin, then return the corresponding
    // shift term 
    if (dist <= dmin)
        return -1.0 / pow(dmin, exp) + 1.0 / pow(2 * R, exp); 
    // If the distance is greater than dmin and less than 2 * R, then
    // evaluate the potential (plus the corresponding shift term)
    else if (dist > dmin && dist <= 2 * R)
        return -1.0 / pow(dist, exp) + 1.0 / pow(2 * R, exp); 
    // If the distance is greater than 2 * R, return zero 
    else 
        return 0.0;
}

/**
 * Compute the first anisotropy parameter in the Gay-Berne-Kihara potential
 * in arbitrary dimensions (2 or 3). 
 *
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param exp Exponent of anisotropy parameter. 
 * @returns The first anisotropy parameter and its partial derivatives with 
 *          respect to the cell coordinates. 
 */
template <typename T, int Dim>
T anisotropyParamGBK1(const Ref<const Matrix<T, Dim, 1> >& n1, const T half_l1,
                      const Ref<const Matrix<T, Dim, 1> >& n2, const T half_l2,
                      const T Rcell, const T exp)
{
    // Is the exponent zero? 
    if (exp == 0)
        return 1.0;

    // Compute the anisotropy parameter
    T chi2 = squaredAspectRatioParam<T>(half_l1, half_l2, Rcell);
    T n1_dot_n2 = n1.dot(n2);
    T arg = 1.0 - chi2 * n1_dot_n2 * n1_dot_n2;
    return pow(arg, -0.5 * exp);
}

/**
 * Compute the second anisotropy parameter in the Gay-Berne-Kihara potential
 * in arbitrary dimensions (2 or 3). 
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param exp Exponent of anisotropy parameter.
 * @param kappa0 Constant multiplier for the fold-difference in well-depths
 *               between the side-by-side and head-to-head parallel cell-cell
 *               configurations.
 * @returns The second anisotropy parameter and its partial derivatives with 
 *          respect to the cell coordinates. 
 */
template <typename T, int Dim>
T anisotropyParamGBK2(const Ref<const Matrix<T, Dim, 1> >& r1,
                      const Ref<const Matrix<T, Dim, 1> >& n1, const T half_l1,
                      const Ref<const Matrix<T, Dim, 1> >& r2,
                      const Ref<const Matrix<T, Dim, 1> >& n2, const T half_l2,
                      const T Rcell, const T exp, const T kappa0)
{
    // Is the exponent zero? 
    if (exp == 0)
        return 1.0;

    // Compute the anisotropy parameter ... 
    //
    // First compute the well-depth parameters \kappa and \chi'
    T kappa = pow(kappa0 * ((min(half_l1, half_l2) / Rcell) + 1), 1 / exp); 
    T chi = (kappa - 1) / (kappa + 1);

    // Compute the vector r12 from r1 to r2 and its normalized vector 
    Matrix<T, Dim, 1> r12 = r2 - r1;
    T r12_norm = r12.norm(); 
    Matrix<T, Dim, 1> r12n = r12 / r12_norm; 

    // Compute the dot products (n1, n2), (r12n, n1), and (r12n, n2)
    T n1_dot_n2 = n1.dot(n2);
    T r12n_dot_n1 = r12n.dot(n1); 
    T r12n_dot_n2 = r12n.dot(n2);

    // Compute the anisotropy parameter 
    T r12n_dot_sum = r12n_dot_n1 + r12n_dot_n2; 
    T r12n_dot_diff = r12n_dot_n1 - r12n_dot_n2;
    T numer1 = r12n_dot_sum * r12n_dot_sum; 
    T numer2 = r12n_dot_diff * r12n_dot_diff; 
    T denom1 = 1 + chi * n1_dot_n2; 
    T denom2 = 1 - chi * n1_dot_n2; 
    return pow(1 - (chi / 2) * (numer1 / denom1 + numer2 / denom2), exp);
}

/**
 * Compute the shifted Gay-Berne-Kihara potential between two neighboring
 * cells in arbitrary dimensions (2 or 3).
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param dist Shortest distance from cell 1 to cell 2.
 * @param expd Exponent determining the distance dependence in the
 *             Gay-Berne-Kihara potential.
 * @param exp1 Exponent of first anisotropy parameter.
 * @param exp2 Exponent of second anisotropy parameter. 
 * @param kappa0 Constant multiplier for the fold-difference in well-depths
 *               between the side-by-side and head-to-head parallel cell-cell
 *               configurations.
 * @param dmin Minimum distance at which the Kihara potential is nonzero.
 * @returns Shifted Gay-Berne-Kihara potential for the given cell-cell 
 *          configuration. 
 */
template <typename T, int Dim>
T potentialGBK(const Ref<const Matrix<T, Dim, 1> >& r1,
               const Ref<const Matrix<T, Dim, 1> >& n1, const T half_l1,
               const Ref<const Matrix<T, Dim, 1> >& r2, 
               const Ref<const Matrix<T, Dim, 1> >& n2, const T half_l2,
               const T R, const T Rcell, const T dist, const T expd,
               const T exp1, const T exp2, const T kappa0, const T dmin)
{
    // If the distance is greater than 2 * R, return zero 
    if (dist > 2 * R) 
        return 0.0;

    // Otherwise, get the anisotropy parameters 
    T eps1 = anisotropyParamGBK1<T, Dim>(n1, half_l1, n2, half_l2, Rcell, exp1);
    T eps2 = anisotropyParamGBK2<T, Dim>(
        r1, n1, half_l1, r2, n2, half_l2, Rcell, exp2, kappa0
    );
    // If the distance is less than dmin, then return the corresponding
    // shift term 
    if (dist <= dmin)
        return eps1 * eps2 * (-1.0 / pow(dmin, expd) + 1.0 / pow(2 * R, expd)); 
    // If the distance is greater than dmin and less than 2 * R, then
    // evaluate the potential (plus the corresponding shift term)
    else    // dist > dmin && dist <= 2 * R
        return eps1 * eps2 * (-1.0 / pow(dist, expd) + 1.0 / pow(2 * R, exp)); 
}

/* --------------------------------------------------------------------- //
 *                      FORCES IN 2 OR 3 DIMENSIONS                      //
 * --------------------------------------------------------------------- */
/**
 * Compute the generalized forces between two neighboring cells that arise
 * from the shifted Kihara potential in arbitrary dimensions (2 or 3).
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param R Cell radius, including the EPS. 
 * @param d12 Shortest distance vector from cell 1 to cell 2.
 * @param s Cell-body coordinate along cell 1 at which shortest distance is 
 *          achieved. 
 * @param t Cell-body coordinate along cell 2 at which shortest distance is
 *          achieved. 
 * @param exp Exponent in Kihara potential.
 * @param dmin Minimum distance at which the Kihara potential is nonzero.
 * @returns Matrix of generalized forces arising from the Kihara potential. 
 */
template <typename T, int Dim>
Array<T, 2, 2 * Dim> forcesKihara(const Ref<const Matrix<T, Dim, 1> >& r1, 
                                  const Ref<const Matrix<T, Dim, 1> >& n1,
                                  const T half_l1,
                                  const Ref<const Matrix<T, Dim, 1> >& r2,
                                  const Ref<const Matrix<T, Dim, 1> >& n2,
                                  const T half_l2, const T R, 
                                  const Ref<const Matrix<T, Dim, 1> >& d12,
                                  const T s, const T t, const T exp,
                                  const T dmin)
{
    Matrix<T, 2, 2 * Dim> dEdq = Matrix<T, 2, 2 * Dim>::Zero();

    // Normalize the distance vector 
    T dist = d12.norm(); 
    Matrix<T, Dim, 1> d12n = d12 / dist; 

    // If the distance falls within the desired range ... 
    if (dist > dmin && dist <= 2 * R)
    {
        // Get the derivative of the potential with respect to the cell-cell
        // distance
        T deriv = exp / pow(dist, exp + 1);

        // Use the above to get the partial derivative of the potential with
        // respect to each coordinate
        //
        // Partial derivatives w.r.t cell 1 center 
        dEdq(0, Eigen::seq(0, Dim - 1)) = -deriv * d12n; 

        // Partial derivatives w.r.t cell 1 orientation 
        dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = -deriv * d12n * s; 

        // Partial derivatives w.r.t cell 2 center 
        dEdq(1, Eigen::seq(0, Dim - 1)) = -dEdq(0, Eigen::seq(0, Dim - 1)); 

        // Partial derivatives w.r.t cell 2 orientation 
        dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = deriv * d12n * t;
    }

    return dEdq.array(); 
}

/**
 * Compute the first anisotropy parameter in the Gay-Berne-Kihara potential
 * in arbitrary dimensions (2 or 3), as well as its partial derivatives with
 * respect to the cell coordinates. 
 *
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param exp Exponent of anisotropy parameter. 
 * @returns The first anisotropy parameter and its partial derivatives with 
 *          respect to the cell coordinates. 
 */
template <typename T, int Dim>
std::pair<T, Matrix<T, 2, 2 * Dim> > anisotropyParamWithDerivsGBK1(const Ref<const Matrix<T, Dim, 1> >& n1,
                                                                   const T half_l1,
                                                                   const Ref<const Matrix<T, Dim, 1> >& n2,
                                                                   const T half_l2,
                                                                   const T Rcell,
                                                                   const T exp)
{
    // Is the exponent zero? 
    if (exp == 0)
        return std::make_pair(1.0, Matrix<T, 2, 2 * Dim>::Zero()); 

    // Compute the anisotropy parameter
    T chi2 = squaredAspectRatioParam<T>(half_l1, half_l2, Rcell);
    T n1_dot_n2 = n1.dot(n2);
    T arg = 1.0 - chi2 * n1_dot_n2 * n1_dot_n2;
    T eps = pow(arg, -0.5 * exp);

    // Get the partial derivative of the anisotropy parameter w.r.t each 
    // coordinate 
    Matrix<T, 2, 2 * Dim> derivs = Matrix<T, 2, 2 * Dim>::Zero();

    // Compute the prefactor for each partial derivative 
    T prefactor = exp * pow(eps, exp - 1) * chi2 * pow(arg, -1.5) * n1_dot_n2; 

    // The partial derivatives w.r.t the cell center coordinates are
    // uniformly zero 
    //
    // Partial derivatives w.r.t cell 1 orientation coordinates
    derivs(0, Eigen::seq(Dim, 2 * Dim - 1)) = prefactor * n2; 

    // Partial derivatives w.r.t cell 2 orientation coordinates
    derivs(1, Eigen::seq(Dim, 2 * Dim - 1)) = prefactor * n1; 
    
    return std::make_pair(eps, derivs); 
}

/**
 * Compute the second anisotropy parameter in the Gay-Berne-Kihara potential
 * in arbitrary dimensions (2 or 3), as well as its partial derivatives with
 * respect to the cell coordinates. 
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param Rcell Cell radius, excluding the EPS.
 * @param exp Exponent of anisotropy parameter.
 * @param kappa0 Constant multiplier for the fold-difference in well-depths
 *               between the side-by-side and head-to-head parallel cell-cell
 *               configurations.
 * @returns The second anisotropy parameter and its partial derivatives with 
 *          respect to the cell coordinates. 
 */
template <typename T, int Dim>
std::pair<T, Matrix<T, 2, 2 * Dim> > anisotropyParamWithDerivsGBK2(const Ref<const Matrix<T, Dim, 1> >& r1, 
                                                                   const Ref<const Matrix<T, Dim, 1> >& n1,
                                                                   const T half_l1,
                                                                   const Ref<const Matrix<T, Dim, 1> >& r2, 
                                                                   const Ref<const Matrix<T, Dim, 1> >& n2,
                                                                   const T half_l2,
                                                                   const T Rcell,
                                                                   const T exp,
                                                                   const T kappa0)
{
    // Is the exponent zero? 
    if (exp == 0)
        return std::make_pair(1.0, Matrix<T, 2, 2 * Dim>::Zero()); 

    // Compute the anisotropy parameter ... 
    //
    // First compute the well-depth parameters \kappa and \chi'
    T kappa = pow(kappa0 * ((min(half_l1, half_l2) / Rcell) + 1), 1 / exp); 
    T chi = (kappa - 1) / (kappa + 1);

    // Compute the vector r12 from r1 to r2 and its normalized vector 
    Matrix<T, Dim, 1> r12 = r2 - r1;
    T r12_norm = r12.norm(); 
    Matrix<T, Dim, 1> r12n = r12 / r12_norm; 

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
    T eps = pow(1 - (chi / 2) * (numer1 / denom1 + numer2 / denom2), exp);

    // Compute the partial derivatives of (r12n, n1) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 2 * Dim> derivs1 = Matrix<T, 2, 2 * Dim>::Zero();
    for (int i = 0; i < Dim; ++i)
    {
        Matrix<T, Dim, 1> v = Matrix<T, Dim, 1>::Zero();
        for (int j = 0; j < Dim; ++j)
            v(j) = (i == j ? -1 : 0) + r12n(i) * r12n(j);
        derivs1(0, i) = n1.dot(v) / r12_norm;
    }
    derivs1(0, Eigen::seq(Dim, 2 * Dim - 1)) = r12n; 
    derivs1(1, Eigen::seq(0, Dim - 1)) = -derivs1(0, Eigen::seq(0, Dim - 1)); 

    // Compute the partial derivatives of (r12n, n2) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 2 * Dim> derivs2 = Matrix<T, 2, 2 * Dim>::Zero();
    for (int i = 0; i < Dim; ++i)
    {
        Matrix<T, Dim, 1> v = Matrix<T, Dim, 1>::Zero(); 
        for (int j = 0; j < Dim; ++j)
            v(j) = (i == j ? -1 : 0) + r12n(i) * r12n(j);
        derivs2(0, i) = n2.dot(v) / r12_norm;
    }
    derivs2(1, Eigen::seq(0, Dim - 1)) = -derivs2(0, Eigen::seq(0, Dim - 1)); 
    derivs2(1, Eigen::seq(Dim, 2 * Dim - 1)) = r12n; 

    // Compute the partial derivatives of ((r12n, n1) + (r12n, n2))^2 with
    // respect to each generalized coordinate 
    Matrix<T, 2, 2 * Dim> derivs_sum2 = 2 * r12n_dot_sum * (derivs1 + derivs2);

    // Compute the partial derivatives of ((r12n, n1) - (r12n, n1))^2 with 
    // respect to each generalized coordinate 
    Matrix<T, 2, 2 * Dim> derivs_diff2 = 2 * r12n_dot_diff * (derivs1 - derivs2); 

    // Compute the partial derivative of (n1, n2) with respect to each 
    // generalized coordinate 
    Matrix<T, 2, 2 * Dim> derivs3 = Matrix<T, 2, 2 * Dim>::Zero(); 
    derivs3(0, Eigen::seq(Dim, 2 * Dim - 1)) = n2;
    derivs3(1, Eigen::seq(Dim, 2 * Dim - 1)) = n1; 

    // Compute the partial derivative of the anisotropy parameter with
    // respect to each generalized coordinate
    Matrix<T, 2, 2 * Dim> term1 = ((denom1 * derivs_sum2) - (numer1 * chi * derivs3)) / (denom1 * denom1);
    Matrix<T, 2, 2 * Dim> term2 = ((denom2 * derivs_diff2) + (numer2 * chi * derivs3)) / (denom2 * denom2); 
    Matrix<T, 2, 2 * Dim> derivs = -0.5 * chi * (term1 + term2); 

    return std::make_pair(eps, exp * pow(eps, exp - 1) * derivs); 
}

/**
 * Compute the generalized forces between two neighboring cells that arise
 * from the shifted Gay-Berne-Kihara potential in arbitrary dimensions (2
 * or 3).
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param d12 Shortest distance vector from cell 1 to cell 2.
 * @param s Cell-body coordinate along cell 1 at which shortest distance is 
 *          achieved. 
 * @param t Cell-body coordinate along cell 2 at which shortest distance is
 *          achieved. 
 * @param expd Exponent determining the distance dependence in the
 *             Gay-Berne-Kihara potential.
 * @param exp1 Exponent of first anisotropy parameter.
 * @param exp2 Exponent of second anisotropy parameter. 
 * @param kappa0 Constant multiplier for the fold-difference in well-depths
 *               between the side-by-side and head-to-head parallel cell-cell
 *               configurations.
 * @param dmin Minimum distance at which the Kihara potential is nonzero.
 * @returns Matrix of generalized forces arising from the Kihara potential. 
 */
template <typename T, int Dim>
Array<T, 2, 2 * Dim> forcesGBK(const Ref<const Matrix<T, Dim, 1> >& r1,
                               const Ref<const Matrix<T, Dim, 1> >& n1, 
                               const T half_l1,
                               const Ref<const Matrix<T, Dim, 1> >& r2, 
                               const Ref<const Matrix<T, Dim, 1> >& n2,
                               const T half_l2, const T R, const T Rcell,
                               const Ref<const Matrix<T, Dim, 1> >& d12, 
                               const T s, const T t, const T expd, const T exp1,
                               const T exp2, const T kappa0, const T dmin)
{
    // Get the anisotropy parameters and their partial derivatives 
    auto result1 = anisotropyParamWithDerivsGBK1<T, Dim>(
        n1, half_l1, n2, half_l2, Rcell, exp1
    );
    T eps1 = result1.first; 
    Matrix<T, 2, 2 * Dim> deps1 = result1.second; 
    auto result2 = anisotropyParamWithDerivsGBK2<T, Dim>(
        r1, n1, half_l1, r2, n2, half_l2, Rcell, exp2, kappa0
    );
    T eps2 = result2.first; 
    Matrix<T, 2, 2 * Dim> deps2 = result2.second; 

    // Use the product rule to get the partial derivatives of the combined
    // anisotropy parameter
    Matrix<T, 2, 2 * Dim> deps_combined = eps1 * deps2 + eps2 * deps1;

    // Compute the forces ...
    Matrix<T, 2, 2 * Dim> dEdq;
    T dist = d12.norm(); 
    if (dist > 0 && dist <= dmin)
    {
        dEdq = deps_combined * (-pow(dmin, -expd) + pow(2 * R, -expd));
    }
    else if (dist > dmin && dist <= 2 * R)
    {
        Matrix<T, Dim, 1> d12n = d12 / dist;
        dEdq(0, Eigen::seq(0, Dim - 1)) = -d12n;
        dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = -s * d12n; 
        dEdq(1, Eigen::seq(0, Dim - 1)) = d12n;
        dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = t * d12n;
        dEdq *= (eps1 * eps2 * expd / pow(dist, expd + 1)); 
        dEdq += deps_combined * (-pow(dist, -expd) + pow(2 * R, -expd)); 
    }
    else 
    {
        dEdq = Matrix<T, 2, 2 * Dim>::Zero(); 
    }

    return dEdq.array(); 
}

#endif 
