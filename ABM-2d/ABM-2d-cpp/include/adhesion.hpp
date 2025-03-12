/**
 * Updated Kihara- and Gay-Berne-Kihara-based attractive potentials for 
 * modeling cell-cell adhesion. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/12/2025
 */

#ifndef ADHESION_POTENTIAL_FORCES_HPP
#define ADHESION_POTENTIAL_FORCES_HPP

#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>

using namespace Eigen;

using std::pow; 
using boost::multiprecision::pow;
using std::sqrt;
using boost::multiprecision::sqrt;
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
 * Compute the Hertzian contact potential between two neighboring cells
 * in arbitrary dimensions (2 or 3).
 *
 * @param dist Shortest distance from cell 1 to cell 2.
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell.  
 * @returns Hertzian contact potential at the given cell-cell distance. 
 */
template <typename T>
T potentialHertz(const T dist, const T R, const T Rcell, const T E0, const T Ecell)
{
    // If the distance is less than R + Rcell ...
    if (dist <= R + Rcell)
    {
        T d0 = (7 * R + 3 * Rcell) / 5.0; 
        T term1 = 2.5 * E0 * pow(R - Rcell, 1.5) * (d0 - dist);
        T term2 = Ecell * pow(R + Rcell - dist, 2.5);
        return sqrt(R) * (term1 + term2); 
    }
    // If the distance is between R + Rcell and 2 * R ... 
    else if (dist <= 2 * R)
    {
        return E0 * sqrt(R) * pow(2 * R - dist, 2.5);
    }
    // Otherwise, return zero
    else
    {
        return 0.0;
    }
}

/**
 * Compute a simplified JKR contact potential between two neighboring 
 * cells in arbitrary dimensions (2 or 3).
 *
 * @param dist Shortest distance from cell 1 to cell 2. 
 * @param R Cell radius, including the EPS. 
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Shifted JKR potential at the given cell-cell distance. 
 */
template <typename T>
T potentialJKR(const T dist, const T R, const T dmin)
{
    // If the distance is less than dmin, then return the corresponding 
    // shift term 
    if (dist <= dmin)
    {
        T d0 = (2 * R + dmin) / 2.0;
        return -boost::math::constants::pi<T>() * R * (2 * R - dmin) * (d0 - dist); 
    }
    // If the distance is greater than dmin and less than 2 * R, then 
    // evaluate the potential (plus the corresponding shift term)
    else if (dist <= 2 * R)
    {
        T overlap = 2 * R - dist; 
        return -0.5 * boost::math::constants::pi<T>() * R * overlap * overlap;
    }
    // If the distance is greater than 2 * R, return zero
    else
    {
        return 0.0;
    }
}

/**
 * Compute the shifted Kihara potential between two neighboring cells
 * in arbitrary dimensions (2 or 3).
 *
 * @param dist Shortest distance from cell 1 to cell 2.
 * @param R Cell radius, including the EPS.
 * @param exp Exponent in Kihara potential.
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Shifted Kihara potential at the given cell-cell distance. 
 */
template <typename T>
T potentialKihara(const T dist, const T R, const T exp, const T dmin)
{
    // If the distance is less than dmin, then return the corresponding
    // shift term 
    if (dist <= dmin)
    {
        T denom1 = pow(dmin, exp); 
        T denom2 = pow(2 * R, exp); 
        T term1 = exp * dist * ((1.0 / (denom1 * dmin)) - (1.0 / (denom2 * 2 * R)));
        T term2 = (exp + 1) * ((1.0 / denom2) - (1.0 / denom1));
        return term1 + term2; 
    }
    // If the distance is greater than dmin and less than 2 * R, then
    // evaluate the potential (plus the corresponding shift term)
    else if (dist <= 2 * R)
    {
        T term1 = -1.0 / pow(dist, exp); 
        T denom = pow(2 * R, exp); 
        T term2 = -exp * dist / (denom * 2 * R);
        T term3 = (exp + 1) / denom;
        return term1 + term2 + term3; 
    }
    // If the distance is greater than 2 * R, return zero 
    else
    {
        return 0.0;
    }
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
 * The second anisotropy parameter exponent is assumed to be zero. 
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
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Shifted Gay-Berne-Kihara potential for the given cell-cell 
 *          configuration. 
 */
template <typename T, int Dim>
T potentialGBK(const Ref<const Matrix<T, Dim, 1> >& r1,
               const Ref<const Matrix<T, Dim, 1> >& n1, const T half_l1,
               const Ref<const Matrix<T, Dim, 1> >& r2, 
               const Ref<const Matrix<T, Dim, 1> >& n2, const T half_l2,
               const T R, const T Rcell, const T dist, const T expd,
               const T exp1, const T dmin)
{
    // If the distance is greater than 2 * R, return zero 
    if (dist > 2 * R) 
        return 0.0;

    // Otherwise, get the anisotropy parameter (second parameter is fixed to 1) 
    T eps1 = anisotropyParamGBK1<T, Dim>(n1, half_l1, n2, half_l2, Rcell, exp1);
    
    // If the distance is less than dmin, then return the corresponding
    // shift term 
    if (dist <= dmin)
    {
        T denom1 = pow(dmin, expd); 
        T denom2 = pow(2 * R, expd); 
        T term1 = expd * dist * ((1.0 / (denom1 * dmin)) - (1.0 / (denom2 * 2 * R)));
        T term2 = (expd + 1) * ((1.0 / denom2) - (1.0 / denom1));
        return eps1 * (term1 + term2); 
    }
    // If the distance is greater than dmin and less than 2 * R, then
    // evaluate the potential (plus the corresponding shift term)
    else
    {
        T denom = pow(2 * R, expd); 
        T term1 = -1.0 / pow(dist, expd); 
        T term2 = -expd * dist / (denom * 2 * R); 
        T term3 = (expd + 1) / denom;
        return eps1 * (term1 + term2 + term3); 
    }
}

/* --------------------------------------------------------------------- //
 *           LAGRANGIAN GENERALIZED FORCES IN 2 OR 3 DIMENSIONS          //
 * --------------------------------------------------------------------- */
/**
 * Compute the Lagrangian generalized forces between two neighboring cells
 * that arise from the shifted Kihara potential in arbitrary dimensions (2
 * or 3).
 *
 * Note that this function technically calculates the *negatives* of the
 * generalized forces.
 *
 * @param n1 Orientation of cell 1.
 * @param n2 Orientation of cell 2. 
 * @param d12 Shortest distance vector from cell 1 to cell 2.
 * @param R Cell radius, including the EPS. 
 * @param s Cell-body coordinate along cell 1 at which shortest distance is 
 *          achieved. 
 * @param t Cell-body coordinate along cell 2 at which shortest distance is
 *          achieved. 
 * @param exp Exponent in Kihara potential.
 * @param dmin Minimum distance at which the potential is nonzero.
 * @param include_constraint If true, enforce the orientation vector norm 
 *                           constraint on the generalized torques. 
 * @returns Matrix of generalized forces arising from the Kihara potential. 
 */
template <typename T, int Dim>
Array<T, 2, 2 * Dim> forcesKiharaLagrange(const Ref<const Matrix<T, Dim, 1> >& n1, 
                                          const Ref<const Matrix<T, Dim, 1> >& n2, 
                                          const Ref<const Matrix<T, Dim, 1> >& d12,
                                          const T R, const T s, const T t,
                                          const T exp, const T dmin, 
                                          const bool include_constraint = true)
{
    Matrix<T, 2, 2 * Dim> dEdq = Matrix<T, 2, 2 * Dim>::Zero();
    const T dist = d12.norm(); 

    // If the distance is less than 2 * R ... 
    if (dist <= 2 * R)
    {
        // Normalize the distance vector 
        Matrix<T, Dim, 1> d12n = d12 / dist;

        // Get the terms that contribute to each generalized force 
        T term1 = (dist <= dmin ? 1.0 / pow(dmin, exp + 1) : 1.0 / pow(dist, exp + 1)); 
        T term2 = 1.0 / pow(2 * R, exp + 1);
        Matrix<T, Dim, 1> v = exp * (term1 - term2) * d12n;
        
        // Partial derivatives w.r.t cell 1 center 
        dEdq(0, Eigen::seq(0, Dim - 1)) = -v; 

        // Partial derivatives w.r.t cell 2 center 
        dEdq(1, Eigen::seq(0, Dim - 1)) = v;

        // Partial derivatives w.r.t cell orientations 
        if (!include_constraint)
        {
            dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = -s * v; 
            dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = t * v;
        }
        else    // Correct torques to account for orientation norm constraint 
        {
            T w1 = n1.dot(-v);
            T w2 = n2.dot(-v);  
            dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = s * (-w1 * n1 - v);
            dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = t * (w2 * n2 + v);  
        }
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
    //T prefactor = exp * pow(eps, exp - 1) * chi2 * pow(arg, -1.5) * n1_dot_n2;
    T prefactor = exp * eps * chi2 * pow(arg, -1) * n1_dot_n2; 

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
 * The second anisotropy parameter exponent is assumed to be zero.
 *
 * Note that this function technically calculates the *negatives* of the
 * generalized forces.
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
 * @param dmin Minimum distance at which the potential is nonzero.
 * @param include_constraint If true, enforce the orientation vector norm 
 *                           constraint on the generalized torques. 
 * @returns Matrix of generalized forces arising from the Kihara potential. 
 */
template <typename T, int Dim>
Array<T, 2, 2 * Dim> forcesGBKLagrange(const Ref<const Matrix<T, Dim, 1> >& r1,
                                       const Ref<const Matrix<T, Dim, 1> >& n1, 
                                       const T half_l1,
                                       const Ref<const Matrix<T, Dim, 1> >& r2, 
                                       const Ref<const Matrix<T, Dim, 1> >& n2,
                                       const T half_l2, const T R, const T Rcell,
                                       const Ref<const Matrix<T, Dim, 1> >& d12, 
                                       const T s, const T t, const T expd,
                                       const T exp1, const T dmin,
                                       const bool include_constraint = true)
{
    // If the distance is greater than 2 * R, return zero
    const T dist = d12.norm(); 
    if (dist > 2 * R)
        return Matrix<T, 2, 2 * Dim>::Zero(); 

    // Get the first anisotropy parameter and its partial derivatives (second
    // anisotropy parameter is fixed to 1)
    auto result1 = anisotropyParamWithDerivsGBK1<T, Dim>(
        n1, half_l1, n2, half_l2, Rcell, exp1
    );
    T eps1 = result1.first; 
    Matrix<T, 2, 2 * Dim> deps1 = result1.second; 

    // Compute the forces ...
    Matrix<T, 2, 2 * Dim> dEdq = Matrix<T, 2, 2 * Dim>::Zero();

    // Normalize the distance vector 
    Matrix<T, Dim, 1> d12n = d12 / dist;

    // If the distance is less than dmin ... 
    if (dist <= dmin)
    {
        T denom1 = pow(dmin, expd); 
        T denom2 = pow(2 * R, expd);
        T denom3 = denom1 * dmin; 
        T denom4 = denom2 * 2 * R; 
        T term1 = eps1 * expd * ((1.0 / denom3) - (1.0 / denom4));
        T term2 = expd * dist * ((1.0 / denom3) - (1.0 / denom4));
        T term3 = (expd + 1) * ((1.0 / denom2) - (1.0 / denom1));
        Matrix<T, Dim, 1> v = term1 * d12n;
        Matrix<T, Dim, 1> w1 = -(term2 + term3) * deps1(0, Eigen::seq(Dim, 2 * Dim - 1));
        Matrix<T, Dim, 1> w2 = -(term2 + term3) * deps1(1, Eigen::seq(Dim, 2 * Dim - 1)); 
        dEdq(0, Eigen::seq(0, Dim - 1)) = -v; 
        dEdq(1, Eigen::seq(0, Dim - 1)) = v;
        dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = -(w1 + s * v);
        dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = -(w2 - t * v);
    }
    // Otherwise, if the distance is less than 2 * R ...  
    else if (dist <= 2 * R)
    {
        T denom1 = pow(dist, expd); 
        T denom2 = pow(2 * R, expd);
        T denom3 = denom1 * dist; 
        T denom4 = denom2 * 2 * R;
        T term1 = eps1 * expd * ((1.0 / denom3) - (1.0 / denom4));
        T term2 = -(1.0 / denom1) - (expd * dist / denom4) + ((expd + 1) / denom2);
        Matrix<T, Dim, 1> v = term1 * d12n;
        Matrix<T, Dim, 1> w1 = -term2 * deps1(0, Eigen::seq(Dim, 2 * Dim - 1)); 
        Matrix<T, Dim, 1> w2 = -term2 * deps1(1, Eigen::seq(Dim, 2 * Dim - 1)); 
        dEdq(0, Eigen::seq(0, Dim - 1)) = -v; 
        dEdq(1, Eigen::seq(0, Dim - 1)) = v;
        dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) = -(w1 + s * v); 
        dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) = -(w2 - t * v);
    }

    // Enforce the orientation vector norm constraint if desired 
    if (include_constraint) 
    {
        T lambda1 = n1.dot(dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)));
        T lambda2 = n2.dot(dEdq(1, Eigen::seq(Dim, 2 * Dim - 1))); 
        dEdq(0, Eigen::seq(Dim, 2 * Dim - 1)) -= lambda1 * n1; 
        dEdq(1, Eigen::seq(Dim, 2 * Dim - 1)) -= lambda2 * n2;  
    }            

    return dEdq.array(); 
}

/* --------------------------------------------------------------------- //
 *                  NEWTONIAN FORCES IN 2 OR 3 DIMENSIONS                //
 * --------------------------------------------------------------------- */
/**
 * Compute the Newtonian forces between two neighboring cells that arise
 * from the shifted Kihara potential in arbitrary dimensions (2 or 3).
 *
 * Note that this function calculates the force on cell 1 due to cell 2. 
 *
 * @param d12 Shortest distance vector from cell 1 to cell 2.
 * @param R Cell radius, including the EPS. 
 * @param s Cell-body coordinate along cell 1 at which shortest distance is 
 *          achieved. 
 * @param t Cell-body coordinate along cell 2 at which shortest distance is
 *          achieved. 
 * @param exp Exponent in Kihara potential.
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Force on cell 1 due to cell 2 arising from the Kihara potential. 
 */
template <typename T, int Dim>
Array<T, Dim, 1> forceKiharaNewton(const Ref<const Matrix<T, Dim, 1> >& d12,
                                   const T R, const T exp, const T dmin)
{
    Matrix<T, Dim, 1> force = Matrix<T, Dim, 1>::Zero();
    const T dist = d12.norm(); 

    // If the distance is less than 2 * R ... 
    if (dist <= 2 * R)
    {
        // Normalize the distance vector 
        Matrix<T, Dim, 1> d12n = d12 / dist;

        // Get the terms that contribute to the force on cell 1 due to cell 2 
        T term1 = (dist <= dmin ? 1.0 / pow(dmin, exp + 1) : 1.0 / pow(dist, exp + 1)); 
        T term2 = 1.0 / pow(2 * R, exp + 1);
        force = exp * (term1 - term2) * d12n; 
    }
    
    return force; 
}

/**
 * Compute the Newtonian force on one cell due to another neighboring cell,
 * arising from the shifted Gay-Berne-Kihara potential in arbitrary dimensions
 * (2 or 3).
 *
 * The second anisotropy parameter exponent is assumed to be zero.
 *
 * Note that this function calculates the force on cell 1 due to cell 2. 
 *
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
 * @param n2 Orientation of cell 2.
 * @param half_l2 Half of length of cell 2. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param d12 Shortest distance vector from cell 1 to cell 2.
 * @param expd Exponent determining the distance dependence in the
 *             Gay-Berne-Kihara potential.
 * @param exp1 Exponent of first anisotropy parameter.
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Force on cell 1 due to cell 2 arising from the Gay-Berne-Kihara
 *          potential. 
 */
template <typename T, int Dim>
Array<T, Dim, 1> forceGBKNewton(const Ref<const Matrix<T, Dim, 1> >& n1, 
                                const T half_l1,
                                const Ref<const Matrix<T, Dim, 1> >& n2,
                                const T half_l2, const T R, const T Rcell,
                                const Ref<const Matrix<T, Dim, 1> >& d12, 
                                const T expd, const T exp1, const T dmin)
{
    // If the distance is greater than 2 * R, return zero
    const T dist = d12.norm(); 
    if (dist > 2 * R)
        return Matrix<T, Dim, 1>::Zero();

    // Get the first anisotropy parameter and its partial derivatives (second
    // anisotropy parameter is fixed to 1)
    auto result1 = anisotropyParamWithDerivsGBK1<T, Dim>(
        n1, half_l1, n2, half_l2, Rcell, exp1
    );
    T eps1 = result1.first; 

    // If the distance is less than 2 * R ... 
    Matrix<T, Dim, 1> force = Matrix<T, Dim, 1>::Zero();
    if (dist <= 2 * R)
    {
        // Normalize the distance vector 
        Matrix<T, Dim, 1> d12n = d12 / dist;

        // Get the terms that contribute to the force on cell 1 due to cell 2 
        T term1 = (dist <= dmin ? 1.0 / pow(dmin, expd + 1) : 1.0 / pow(dist, expd + 1)); 
        T term2 = 1.0 / pow(2 * R, expd + 1); 
        force = eps1 * expd * (term1 - term2) * d12n; 
    }

    return force;
}

/**
 * Compute the Newtonian torque on one cell due to another neighboring cell,
 * arising from the shifted Gay-Berne-Kihara potential in arbitrary dimensions
 * (2 or 3).
 *
 * The second anisotropy parameter exponent is assumed to be zero.
 *
 * Note that this function calculates the torque on cell 1 due to cell 2 
 * according to the prescription given by Allen and Tildesley (Appendix C).
 *
 * If Dim == 2, then this vector should be nonzero only in the z-coordinate; 
 * if Dim == 3, then this vector can be nonzero in all three coordinates. 
 *
 * @param n1 Orientation of cell 1.
 * @param half_l1 Half of length of cell 1.
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
 * @param dmin Minimum distance at which the potential is nonzero.
 * @returns Force on cell 1 due to cell 2 arising from the Gay-Berne-Kihara
 *          potential. 
 */
template <typename T, int Dim>
Array<T, 3, 1> torqueGBKNewton(const Ref<const Matrix<T, Dim, 1> >& n1,
                               const T half_l1,
                               const Ref<const Matrix<T, Dim, 1> >& n2,
                               const T half_l2, const T R, const T Rcell,
                               const Ref<const Matrix<T, Dim, 1> >& d12, 
                               const T s, const T t, const T expd, const T exp1,
                               const T dmin)
{
    // If the distance is greater than 2 * R, return zero
    const T dist = d12.norm(); 
    if (dist > 2 * R)
        return Array<T, 3, 1>::Zero(); 

    // Get the first anisotropy parameter and its partial derivatives (second
    // anisotropy parameter is fixed to 1)
    auto result1 = anisotropyParamWithDerivsGBK1<T, Dim>(
        n1, half_l1, n2, half_l2, Rcell, exp1
    );
    T eps1 = result1.first; 
    Matrix<T, 2, 2 * Dim> deps1 = result1.second; 

    // Normalize the distance vector 
    Matrix<T, Dim, 1> d12n = d12 / dist;

    // If the distance is less than dmin ...
    Matrix<T, Dim, 1> torque; 
    if (dist <= dmin)
    {
        T denom1 = pow(dmin, expd); 
        T denom2 = pow(2 * R, expd);
        T denom3 = denom1 * dmin; 
        T denom4 = denom2 * 2 * R; 
        T term1 = eps1 * expd * ((1.0 / denom3) - (1.0 / denom4));
        T term2 = expd * dist * ((1.0 / denom3) - (1.0 / denom4));
        T term3 = (expd + 1) * ((1.0 / denom2) - (1.0 / denom1));
        Matrix<T, Dim, 1> v = term1 * d12n;
        Matrix<T, Dim, 1> w1 = -(term2 + term3) * deps1(0, Eigen::seq(Dim, 2 * Dim - 1));
        torque = w1 + s * v;
    }
    // Otherwise, if the distance is less than 2 * R ...  
    else if (dist <= 2 * R)
    {
        T denom1 = pow(dist, expd); 
        T denom2 = pow(2 * R, expd);
        T denom3 = denom1 * dist; 
        T denom4 = denom2 * 2 * R;
        T term1 = eps1 * expd * ((1.0 / denom3) - (1.0 / denom4));
        T term2 = -(1.0 / denom1) - (expd * dist / denom4) + ((expd + 1) / denom2);
        Matrix<T, Dim, 1> v = term1 * d12n;
        Matrix<T, Dim, 1> w1 = -term2 * deps1(0, Eigen::seq(Dim, 2 * Dim - 1)); 
        torque = w1 + s * v; 
    }

    // Enforce the orientation vector norm constraint
    T lambda1 = n1.dot(-torque);
    torque += lambda1 * n1;

    // Take the cross product with the orientation vector 
    Matrix<T, 3, 1> u = Matrix<T, 3, 1>::Zero();
    Matrix<T, 3, 1> v = Matrix<T, 3, 1>::Zero();
    for (int i = 0; i < Dim; ++i)
    {
        u(i) = n1(i); 
        v(i) = torque(i);
    }
    return u.cross(v).array(); 
}

#endif 
