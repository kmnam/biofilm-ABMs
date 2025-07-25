/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/15/2024
 */

#ifndef ROD_LENNARD_JONES_POTENTIALS_2D_HPP
#define ROD_LENNARD_JONES_POTENTIALS_2D_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>

using namespace Eigen;

using std::abs;
using boost::multiprecision::abs;
using std::pow;
using boost::multiprecision::pow;
using std::sqrt;
using boost::multiprecision::sqrt; 
using std::sin;
using boost::multiprecision::sin; 
using std::tan;
using boost::multiprecision::tan;
using std::acos; 
using boost::multiprecision::acos;
using std::atan;
using boost::multiprecision::atan;

// -------------------------------------------------------------------- //
//                           HELPER FUNCTIONS
// -------------------------------------------------------------------- //
/**
 * Get the perpendicular distance between two parallel lines in 2-D,
 * where the two lines are parametrized as r1 + s * n and r2 + s * n.
 */
template <typename T>
T distBetweenLines(const Ref<const Matrix<T, 2, 1> >& r1,
                   const Ref<const Matrix<T, 2, 1> >& r2, 
                   const Ref<const Matrix<T, 2, 1> >& n)
{
    T m = n(1) / n(0); 
    T b1 = r1(1) - m * r1(0); 
    T b2 = r2(1) - m * r2(0); 
    return abs(b2 - b1) / sqrt(m * m + 1);
}

/**
 * Get the projection of the given point `p` onto the line passing through
 * `q` which is parallel to the unit vector `n`. 
 */
template <typename T>
Matrix<T, 2, 1> projectPointOntoLine(const Ref<const Matrix<T, 2, 1> >& p,
                                     const Ref<const Matrix<T, 2, 1> >& q,
                                     const Ref<const Matrix<T, 2, 1> >& n)
{
    return q + (p - q).dot(n) * n;
}

// -------------------------------------------------------------------- //
//                ATTRACTIVE POTENTIAL FOR COLLINEAR RODS               //
// -------------------------------------------------------------------- //
/**
 * Compute the four terms that contribute to the 2-D attractive potential 
 * between two collinear spherocylindrical rods. 
 */
template <typename T>
Matrix<T, 4, 1> attractivePotentialCollinearTerms2D(const T x0, const T p0,
                                                    const T half_l1, 
                                                    const T half_l2)
{
    // Compute the terms that contribute to the potential 
    T offset = x0 - p0; 
    Matrix<T, 4, 1> terms;
    terms << pow(offset + half_l1 + half_l2, -4.0),
             -pow(offset + half_l1 - half_l2, -4.0),
             -pow(offset - half_l1 + half_l2, -4.0),
             pow(offset - half_l1 - half_l2, -4.0);

    return terms;
}

/**
 * Compute the 2-D attractive potential between two collinear spherocylindrical 
 * rods. 
 */
template <typename T>
T attractivePotentialCollinear2D(const Ref<const Matrix<T, 2, 1> >& r1, 
                                 const T half_l1, 
                                 const Ref<const Matrix<T, 2, 1> >& r2, 
                                 const T half_l2)
{
    // Compute the arguments to pass to the helper function
    T x0 = 0; 
    T p0 = (r2 - r1).norm();

    // Compute the terms contributing to the potential in T
    Matrix<T, 4, 1> terms = attractivePotentialCollinearTerms2D<T>(
        x0, p0, half_l1, half_l2
    );
    T sum = terms.sum();

    return -sum / 20.0;
}

// -------------------------------------------------------------------- //
//                ATTRACTIVE POTENTIAL FOR PARALLEL RODS                //
// -------------------------------------------------------------------- //
/**
 * Compute the eight terms that contribute to the 2-D attractive potential 
 * between two parallel spherocylindrical rods. 
 */
template <typename T>
Matrix<T, 8, 1> attractivePotentialParallelTerms2D(const T x0, const T p0,
                                                   const T r, const T half_l1,
                                                   const T half_l2)
{
    // Compute the terms that contribute to the potential (4 groups of 2)
    Matrix<T, 8, 1> terms = Matrix<T, 8, 1>::Zero();
    std::vector<T> delta1 { half_l1, -half_l1 }; 
    std::vector<T> delta2 { half_l2, -half_l2 };
    int i = 0;
    for (const T& d1 : delta1)
    {
        for (const T& d2 : delta2)
        {
            T x = x0 + d1; 
            T a = p0 + d2;
            T r2 = r * r; 
            T offset = a - x;
            T term1 = 1.0 / (8 * r2 * (pow(offset, 2) + r2));
            T term2 = -3 * offset / (8 * pow(r, 5)) * atan(offset / r);
            terms(i) = term1; 
            terms(4 + i) = term2;
            i++;
        }
    }

    // Negate half the terms and re-organize into a vector 
    terms(1) *= -1;
    terms(2) *= -1; 
    terms(5) *= -1; 
    terms(6) *= -1;
    return terms;
}

/**
 * Compute the 2-D attractive potential between two parallel spherocylindrical 
 * rods.
 *
 * Here, the orientation vector of spherocylinder 2 is assumed to be the same 
 * as the orientation vector of spherocylinder 1, given here as `n1`.
 */
template <typename T>
T attractivePotentialParallel2D(const Ref<const Matrix<T, 2, 1> >& r1, 
                                const Ref<const Matrix<T, 2, 1> >& n1, 
                                const T half_l1,
                                const Ref<const Matrix<T, 2, 1> >& r2,
                                const T half_l2, const T dist)
{
    // Compute the arguments to pass to the helper function
    //
    // Set x0 to 0 and set p0 to the position of rod 2 when projected 
    // onto the axis of rod 1
    T x0 = 0;
    Matrix<T, 2, 1> v2 = projectPointOntoLine<T>(r2, r1, n1);
    T p0 = (v2 - r1).norm();

    // Compute the terms contributing to the potential in MainType
    Matrix<T, 8, 1> terms = attractivePotentialParallelTerms2D<T>(
        x0, p0, dist, half_l1, half_l2
    );
    T sum = terms.sum();

    return -sum;
}

// -------------------------------------------------------------------- //
//                  ATTRACTIVE POTENTIAL FOR SKEW RODS                  //
// -------------------------------------------------------------------- //
/**
 * Compute the 8 terms that contribute to the 2-D attractive potential
 * between two skew spherocylindrical rods.
 *
 * The terms are arranged as a vector, with elements:
 *
 * [  G1(x2, a2, b) - G1(x1, a2, b)  ]
 * [  G1(x1, a1, b) - G1(x2, a1, b)  ]
 * [         -G2(x1, a1, b)          ]
 * [         +G2(x1, a2, b)          ]
 * [         +G2(x2, a1, b)          ]
 * [         -G2(x2, a2, b)          ]
 * [  G3(x2, a2, b) - G3(x2, a1, b)  ]
 * [  G3(x1, a1, b) - G3(x1, a2, b)  ],
 *
 * where G1, G2, and G3 are the three terms in the function G(x, a, b).
 */
template <typename T>
Matrix<T, 8, 1> attractivePotentialSkewTerms2D(const T x0, const T p0,
                                               const T sin_theta,
                                               const T cot_theta,
                                               const T half_l1,
                                               const T half_l2)
{
    // Compute the terms that contribute to the potential
    Matrix<T, 8, 1> terms = Matrix<T, 8, 1>::Zero();
    const T x1 = x0 - half_l1; 
    const T x2 = x0 + half_l1;
    const T a1 = (p0 - half_l2) * sin_theta; 
    const T a2 = (p0 + half_l2) * sin_theta; 
    const T b = cot_theta; 
    const T b2_plus1 = b * b + 1; 
    const T b2_plus1_sq = b2_plus1 * b2_plus1;

    // The first two terms are differences of arctangents
    const T x1_2 = x1 * x1; 
    const T x1_3 = x1_2 * x1; 
    const T x1_4 = x1_3 * x1;
    const T x2_2 = x2 * x2; 
    const T x2_3 = x2_2 * x2; 
    const T x2_4 = x2_3 * x2;
    const T a1_2 = a1 * a1; 
    const T a1_3 = a1_2 * a1; 
    const T a1_4 = a1_3 * a1; 
    const T a2_2 = a2 * a2; 
    const T a2_3 = a2_2 * a2; 
    const T a2_4 = a2_3 * a2;
    const T arg1 = b - (x2 / a2); 
    const T arg2 = b - (x1 / a2);
    const T arg3 = b - (x1 / a1); 
    const T arg4 = b - (x2 / a1);
    terms(0) = 3 * (atan(arg1) - atan(arg2)) / a2_4;
    terms(1) = 3 * (atan(arg3) - atan(arg4)) / a1_4;

    // The last two terms are also differences of arctangents
    const T arg5 = -(b2_plus1 * a2 / x2 - b); 
    const T arg6 = -(b2_plus1 * a1 / x2 - b);
    const T arg7 = -(b2_plus1 * a1 / x1 - b);
    const T arg8 = -(b2_plus1 * a2 / x1 - b);
    terms(6) = 3 * b2_plus1_sq * (atan(arg5) - atan(arg6)) / x2_4;
    terms(7) = 3 * b2_plus1_sq * (atan(arg7) - atan(arg8)) / x1_4;

    // Compute the middle four terms
    //
    // The first middle term is for x = x1, a = a1
    const T numer1 = 3 * b2_plus1_sq * a1_4 - 3 * b * b2_plus1 * a1_3 * x1 + 2 * a1_2 * x1_2 - 3 * b * a1 * x1_3 + 3 * x1_4;
    const T denom1 = a1_3 * x1_3 * (a1_2 * b2_plus1 - 2 * b * a1 * x1 + x1_2);
    terms(2) = -numer1 / denom1;

    // The second middle term is for x = x1, a = a2
    const T numer2 = 3 * b2_plus1_sq * a2_4 - 3 * b * b2_plus1 * a2_3 * x1 + 2 * a2_2 * x1_2 - 3 * b * a2 * x1_3 + 3 * x1_4; 
    const T denom2 = a2_3 * x1_3 * (a2_2 * b2_plus1 - 2 * b * a2 * x1 + x1_2); 
    terms(3) = numer2 / denom2; 

    // The third middle term is for x = x2, a = a1
    const T numer3 = 3 * b2_plus1_sq * a1_4 - 3 * b * b2_plus1 * a1_3 * x2 + 2 * a1_2 * x2_2 - 3 * b * a1 * x2_3 + 3 * x2_4;
    const T denom3 = a1_3 * x2_3 * (a1_2 * b2_plus1 - 2 * b * a1 * x2 + x2_2);
    terms(4) = numer3 / denom3;

    // The fourth middle term is for x = x2, a = a2
    const T numer4 = 3 * b2_plus1_sq * a2_4 - 3 * b * b2_plus1 * a2_3 * x2 + 2 * a2_2 * x2_2 - 3 * b * a2 * x2_3 + 3 * x2_4; 
    const T denom4 = a2_3 * x2_3 * (a2_2 * b2_plus1 - 2 * b * a2 * x2 + x2_2); 
    terms(5) = -numer4 / denom4;

    return terms; 
}

/**
 * Compute the 2-D attractive potential between two skew spherocylindrical rods.
 */
template <typename MainType, typename PreciseType>
MainType attractivePotentialSkew2D(const Ref<const Matrix<MainType, 2, 1> >& r1,
                                   const Ref<const Matrix<MainType, 2, 1> >& n1,
                                   const MainType half_l1,
                                   const Ref<const Matrix<MainType, 2, 1> >& r2,
                                   const Ref<const Matrix<MainType, 2, 1> >& n2,
                                   const MainType half_l2)
{
    // Identify the intersection of the lines spanned by the orientation vectors
    Matrix<MainType, 2, 2> A; 
    A << n1(0), -n2(0),
         n1(1), -n2(1); 
    Matrix<MainType, 2, 1> b = r2 - r1;
    Matrix<MainType, 2, 1> y = A.partialPivLu().solve(b);
    Matrix<MainType, 2, 1> origin = r1 + y(0) * n1; 

    // Find the angle formed by the two rods with respect to the origin
    Matrix<MainType, 2, 1> v1 = r1 - origin; 
    Matrix<MainType, 2, 1> v2 = r2 - origin;
    MainType x0 = v1.norm(); 
    MainType p0 = v2.norm();
    MainType theta = acos(v1.dot(v2) / (x0 * p0));
    MainType cot_theta = 1.0 / tan(theta); 
    MainType sin_theta = sin(theta);

    // First compute the terms contributing to the potential in MainType
    Matrix<MainType, 8, 1> terms = attractivePotentialSkewTerms2D<MainType>(
        x0, p0, sin_theta, cot_theta, half_l1, half_l2
    );

    // What is the gap between the largest and smallest term?
    Matrix<MainType, 8, 1> abs_terms = terms.cwiseAbs(); 
    MainType max_term = abs_terms.maxCoeff(); 
    MainType min_term = abs_terms.minCoeff();
    bool large_diff = (max_term - min_term > 1e+10);

    // If so, re-compute the terms in PreciseType and calculate the corresponding
    // potential 
    MainType sum = 0;
    if (large_diff)
    {
        Matrix<PreciseType, 8, 1> terms2 = attractivePotentialSkewTerms2D<PreciseType>(
            static_cast<PreciseType>(x0),
            static_cast<PreciseType>(p0),
            static_cast<PreciseType>(sin_theta),
            static_cast<PreciseType>(cot_theta),
            static_cast<PreciseType>(half_l1),
            static_cast<PreciseType>(half_l2)
        );
        sum = static_cast<MainType>(terms2.sum());
    }
    else 
    {
        sum = terms.sum();
    }
    MainType potential = -sum / (32 * sin_theta);

    return potential;
}

// -------------------------------------------------------------------- //
//             COMPOSITE FUNCTIONS FOR POTENTIAL AND FORCES             //
// -------------------------------------------------------------------- //
/**
 * Compute the 2-D attractive potential between two spherocylindrical rods
 * from their center orientations, orientation vectors, and lengths. 
 */
template <typename MainType, typename PreciseType>
MainType attractivePotentialLJ2D(const Ref<const Matrix<MainType, 2, 1> >& r1, 
                                 const Ref<const Matrix<MainType, 2, 1> >& n1,
                                 const MainType half_l1, 
                                 const Ref<const Matrix<MainType, 2, 1> >& r2, 
                                 const Ref<const Matrix<MainType, 2, 1> >& n2,
                                 const MainType half_l2,
                                 const MainType cos_eps_parallel,
                                 const MainType eps_collinear)
{
    // Get the angle formed by the two orientation vectors
    MainType cos_theta = n1.dot(n2); 
    if (cos_theta <= -1)    // Note that cos(theta) should lie between -1 and 1
        cos_theta = -1;
    else if (cos_theta >= 1)
        cos_theta = 1;

    // Are the two rods parallel?
    if (cos_theta > cos_eps_parallel || cos_theta < -cos_eps_parallel)
    {
        // If so, get the perpendicular distance between the axes of the two rods
        MainType dist1 = distBetweenLines<MainType>(r1, r2, n1);
        MainType dist2 = distBetweenLines<MainType>(r1, r2, n2);
        MainType dist = (dist1 + dist2) / 2.0;

        // Is the distance close to zero (are the axes actually collinear)? 
        if (dist < eps_collinear)
        {
            return attractivePotentialCollinear2D<MainType>(
                r1, half_l1, r2, half_l2
            );
        }
        else 
        {
            return attractivePotentialParallel2D<MainType>(
                r1, n1, half_l1, r2, half_l2, dist
            );
        }
    }
    // Otherwise, the two rods are skew 
    else 
    {
        return attractivePotentialSkew2D<MainType, PreciseType>(
            r1, n1, half_l1, r2, n2, half_l2
        );
    }
}

/**
 * Compute the generalized forces arising from the 2-D attractive potential 
 * between two spherocylindrical rods. 
 */
template <typename MainType, typename PreciseType>
Matrix<MainType, 2, 4> attractiveForcesLJ2D(const Ref<const Matrix<MainType, 2, 1> >& r1,
                                            const Ref<const Matrix<MainType, 2, 1> >& n1, 
                                            const MainType half_l1, 
                                            const Ref<const Matrix<MainType, 2, 1> >& r2, 
                                            const Ref<const Matrix<MainType, 2, 1> >& n2, 
                                            const MainType half_l2,
                                            const MainType cos_eps_parallel, 
                                            const MainType eps_collinear,
                                            const MainType delta) 
{
    Matrix<MainType, 2, 4> dEdq; 
    
    // Compute each partial derivative as a finite-difference approximation
    Matrix<MainType, 6, 1> x;
    x << r1(0), r1(1), n1(0), n1(1), n2(0), n2(1);
    for (int i = 0; i < 6; ++i)
    {
        x(i) += delta;
        Matrix<MainType, 2, 1> rt1 = x(Eigen::seq(0, 1)); 
        Matrix<MainType, 2, 1> nt1 = x(Eigen::seq(2, 3)) / x(Eigen::seq(2, 3)).norm(); 
        Matrix<MainType, 2, 1> nt2 = x(Eigen::seq(4, 5)) / x(Eigen::seq(4, 5)).norm();
        MainType term1 = attractivePotentialLJ2D<MainType, PreciseType>(
            rt1, nt1, half_l1, r2, nt2, half_l2, cos_eps_parallel,
            eps_collinear
        );
        x(i) -= 2 * delta;
        rt1 = x(Eigen::seq(0, 1)); 
        nt1 = x(Eigen::seq(2, 3)) / x(Eigen::seq(2, 3)).norm(); 
        nt2 = x(Eigen::seq(4, 5)) / x(Eigen::seq(4, 5)).norm();
        MainType term2 = attractivePotentialLJ2D<MainType, PreciseType>(
            rt1, nt1, half_l1, r2, nt2, half_l2, cos_eps_parallel,
            eps_collinear
        );
        MainType deriv = (term1 - term2) / (2 * delta); 
        if (i == 0 || i == 1)
        {
            dEdq(0, i) = deriv; 
            dEdq(1, i) = -deriv;
        }
        else if (i == 2 || i == 3)
        {
            dEdq(0, i) = deriv; 
        }
        else    // i == 4 || i == 5
        {
            dEdq(1, i - 2) = deriv; 
        }
        x(i) += delta;
    }

    return dEdq;
}

#endif
