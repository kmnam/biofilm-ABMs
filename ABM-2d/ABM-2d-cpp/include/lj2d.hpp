/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/22/2024
 */

#ifndef ROD_LENNARD_JONES_POTENTIALS_2D_HPP
#define ROD_LENNARD_JONES_POTENTIALS_2D_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "distances.hpp"

using namespace Eigen;

using std::abs;
using boost::multiprecision::abs;
using std::pow;
using boost::multiprecision::pow;
using std::sin;
using boost::multiprecision::sin; 
using std::tan;
using boost::multiprecision::tan;
using std::acos; 
using boost::multiprecision::acos;
using std::atan;
using boost::multiprecision::atan;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;

/**
 * Compute the floating-point sum of two numbers and the corresponding 
 * round-off error via the Fast2Sum algorithm. 
 */
template <typename T>
std::pair<T, T> fast2Sum(const T a, const T b)
{
    T s = a + b; 
    T t = b - (s - a);
    return std::make_pair(s, t);
}

/**
 * Compute the recursive sum of the given vector of floating-point numbers.
 */
template <typename T>
T vecSum(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    T sum = 0;    // Running sum
    for (const T& a : x)
        sum += a;
    return sum;
}

/**
 * Compute the sum of the given vector of floating-point numbers using 
 * Kahan summation. 
 */
template <typename T>
T kahanSum(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    T sum = 0;    // Running sum 
    T c = 0;      // Running compensation
    for (const T& a : x)
    {
        T y = a + c; 
        auto result = fast2Sum(sum, y);
        sum = result.first; 
        c = result.second;
    }

    return sum; 
}

/**
 * Compute the sum of the given vector of floating-point numbers using 
 * Neumaier summation. 
 */
template <typename T>
T neumaierSum(const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    T sum = 0;    // Running sum
    T c = 0;      // Running compensation
    for (const T& a : x)
    {
        T y = sum + a;
        if (abs(sum) >= abs(a))
            c += (sum - y) + a;
        else    // abs(sum) < abs(a)
            c += (a - y) + sum;
        sum = y;
    }

    return sum + c;
}

/**
 * Compute the helper function for the 2-D attractive potential between 
 * two parallel spherocylindrical rods. 
 */
template <typename T>
T attractivePotentialParallelHelper2D(const T x, const T a, const T r)
{
    T term1 = 1.0 / (8 * r * r * (pow(a - x, 2) + r * r));
    T term2 = -3 * (a - x) / (8 * pow(r, 5)) * atan((a - x) / r);
    return term1 + term2;
}

/**
 * Compute the 2-D attractive potential between two parallel spherocylindrical 
 * rods. 
 */
template <typename T>
T attractivePotentialParallel2D(const T x0, const T p0, const T r, const T half_l1,
                                const T half_l2)
{
    T term1 = attractivePotentialParallelHelper2D<T>(x0 + half_l1, p0 + half_l2, r); 
    T term2 = attractivePotentialParallelHelper2D<T>(x0 + half_l1, p0 - half_l2, r); 
    T term3 = attractivePotentialParallelHelper2D<T>(x0 - half_l1, p0 + half_l2, r); 
    T term4 = attractivePotentialParallelHelper2D<T>(x0 - half_l1, p0 - half_l2, r);
    return -(term1 - term2 - term3 + term4);
}

/**
 * Compute the 12 terms that contribute to the 2-D attractive potential
 * between two skew spherocylindrical rods. 
 */
template <typename T>
Matrix<T, Dynamic, 1> attractivePotentialHelper2D(const T x0, const T p0,
                                                  const T sin_theta,
                                                  const T cot_theta,
                                                  const T half_l1,
                                                  const T half_l2)
{
    // Compute the terms that contribute to the potential (4 groups of 3)
    Matrix<T, Dynamic, Dynamic> terms = Matrix<T, Dynamic, Dynamic>::Zero(4, 3);
    std::vector<T> delta1 { half_l1, -half_l1 }; 
    std::vector<T> delta2 { half_l2, -half_l2 };
    const T b = cot_theta; 
    const T c = sin_theta;
    int i = 0;
    for (const T& d1 : delta1)
    {
        for (const T& d2 : delta2)
        {
            T x = x0 + d1; 
            T a = (p0 + d2) * c;
            T arg1 = b - (x / a);
            T arg2 = -((b * b + 1) * a - b * x) / x;
            T term1 = 3 * atan(arg1) / pow(a, 4);
            T term2a = b * b + 1;
            T term2b = 3 * pow(term2a, 2) * pow(a, 4);
            T term2c = -3 * b * term2a * pow(a, 3) * x; 
            T term2d = 2 * a * a * x * x; 
            T term2e = -3 * b * a * pow(x, 3); 
            T term2f = 3 * pow(x, 4);
            T term2g = pow(a, 3) * pow(x, 3); 
            T term2h = a * a * term2a; 
            T term2i = -2 * b * a * x; 
            T term2j = x * x;
            Matrix<T, Dynamic, 1> numer(5);
            Matrix<T, Dynamic, 1> denom(3); 
            numer << term2b, term2c, term2d, term2e, term2f; 
            denom << term2h, term2i, term2j; 
            T term2 = -vecSum<T>(numer) / (term2g * vecSum<T>(denom));
            T term3 = 3 * pow(term2a, 2) * atan(arg2) / pow(x, 4);
            terms(i, 0) = term1; 
            terms(i, 1) = term2; 
            terms(i, 2) = term3;
            i++;
        }
    }

    // Negate half the terms and re-organize into a vector 
    terms.row(1) *= -1; 
    terms.row(2) *= -1;
    return terms.reshaped(12, 1);
}

/**
 * Compute the 2-D attractive potential between two skew spherocylindrical rods.
 */
template <typename MainType, typename PreciseType>
MainType attractivePotential2D(const MainType x0, const MainType p0, const MainType theta,
                               const MainType half_l1, const MainType half_l2)
{
    MainType cot_theta = 1.0 / tan(theta); 
    MainType sin_theta = sin(theta);

    // First compute the terms contributing to the potential in MainType
    Matrix<MainType, Dynamic, 1> terms = attractivePotentialHelper2D<MainType>(
        x0, p0, sin_theta, cot_theta, half_l1, half_l2
    );
    std::cout << "terms :\n" << terms.transpose() << std::endl; 

    // Are any of the terms large in magnitude? 
    bool large = false; 
    for (const MainType& term : terms)
    {
        if (abs(term) > 1e+8)
        {
            large = true; 
            break;
        }
    }
    std::cout << "large? " << large << std::endl; 

    // If so, re-compute the terms in PreciseType and calculate the corresponding
    // potential 
    MainType sum = 0;
    if (large)
    {
        std::cout << "in precise mode\n";
        Matrix<PreciseType, Dynamic, 1> terms2 = attractivePotentialHelper2D<PreciseType>(
            static_cast<PreciseType>(x0),
            static_cast<PreciseType>(p0),
            static_cast<PreciseType>(sin_theta),
            static_cast<PreciseType>(cot_theta),
            static_cast<PreciseType>(half_l1),
            static_cast<PreciseType>(half_l2)
        );
        std::cout << "terms2 : \n" << terms2.transpose() << std::endl;  
        sum = static_cast<MainType>(neumaierSum<PreciseType>(terms2));
    }
    else 
    {
        sum = neumaierSum<MainType>(terms);
    }
    MainType potential = -sum / (32 * sin_theta);
    std::cout << "sum = " << sum << std::endl; 
    std::cout << "potential = " << potential << std::endl; 

    return potential;
}

/**
 * Compute the 2-D attractive potential between two spherocylindrical rods
 * from their center orientations, orientation vectors, and lengths. 
 */
template <typename MainType, typename PreciseType>
MainType attractivePotential2D(const Ref<const Matrix<MainType, 3, 1> >& r1, 
                               const Ref<const Matrix<MainType, 3, 1> >& n1,
                               const MainType half_l1, 
                               const Ref<const Matrix<MainType, 3, 1> >& r2, 
                               const Ref<const Matrix<MainType, 3, 1> >& n2,
                               const MainType half_l2, const K& kernel)
{
    std::cout << r1.transpose() << std::endl; 
    std::cout << n1.transpose() << std::endl; 
    std::cout << half_l1 << std::endl; 
    std::cout << r2.transpose() << std::endl; 
    std::cout << n2.transpose() << std::endl; 
    std::cout << half_l2 << std::endl; 

    // Are the two rods parallel?
    MainType cos_theta = n1.dot(n2);
    std::cout << "cos_theta = " << cos_theta << std::endl; 
    if (cos_theta == 1)
    {
        // If so, get the separation between the two rods
        Matrix<double, 3, 1> p1 = (r1 - half_l1 * n1).template cast<double>(); 
        Matrix<double, 3, 1> q1 = (r1 + half_l1 * n1).template cast<double>();
        Matrix<double, 3, 1> p2 = (r2 - half_l2 * n2).template cast<double>();
        Matrix<double, 3, 1> q2 = (r2 + half_l2 * n2).template cast<double>();
        Point_3 p1_(p1(0), p1(1), p1(2)); 
        Point_3 q1_(q1(0), q1(1), q1(2)); 
        Point_3 p2_(p2(0), p2(1), p2(2)); 
        Point_3 q2_(q2(0), q2(1), q2(2)); 
        Segment_3 seg1(p1_, q1_), seg2(p2_, q2_);
        auto result = distBetweenCells<MainType>(
            seg1, seg2, r1.head(2), n1.head(2), half_l1, r2.head(2), n2.head(2),
            half_l2, kernel
        );
        MainType dist = static_cast<MainType>(std::get<0>(result).norm());

        // Set x0 to 0 and set p0 to the position of rod 2 when projected 
        // onto the axis of rod 1
        MainType x0 = 0;
        MainType p0;    // TODO What to set for p0?
        return attractivePotentialParallel2D<MainType>(x0, p0, dist, half_l1, half_l2); 
    }
    else 
    {
        // If not, identify the intersection of the lines spanned by the
        // orientation vectors
        Matrix<MainType, 3, 1> origin;
        Matrix<MainType, 2, 2> A; 
        A << n1(0), -n2(0),
             n1(1), -n2(1); 
        Matrix<MainType, 2, 1> b = r2.head(2) - r1.head(2); 
        Matrix<MainType, 2, 1> y = A.partialPivLu().solve(b);
        Matrix<MainType, 2, 1> z = r1.head(2) + y(0) * n1.head(2); 
        origin << z(0), z(1), 0;

        // Find the angle formed by the two rods with respect to the origin
        Matrix<MainType, 3, 1> q1 = r1 - origin; 
        Matrix<MainType, 3, 1> q2 = r2 - origin;
        MainType x0 = q1.norm(); 
        MainType p0 = q2.norm();
        MainType phi = acos(q1.dot(q2) / (x0 * p0));

        // Compute the corresponding potential
        std::cout << "x0 = " << x0 << std::endl; 
        std::cout << "p0 = " << p0 << std::endl; 
        std::cout << "phi = " << phi << std::endl; 
        return attractivePotential2D<MainType, PreciseType>(
            x0, p0, phi, half_l1, half_l2
        );
    }
}

/**
 * Compute the generalized forces arising from the 2-D attractive potential 
 * between two spherocylindrical rods. 
 */
template <typename MainType, typename PreciseType>
Matrix<MainType, 2, 4> attractiveForces2D(const Ref<const Matrix<MainType, 3, 1> >& r1,
                                          const Ref<const Matrix<MainType, 3, 1> >& n1, 
                                          const MainType half_l1, 
                                          const Ref<const Matrix<MainType, 3, 1> >& r2, 
                                          const Ref<const Matrix<MainType, 3, 1> >& n2, 
                                          const MainType half_l2,
                                          const MainType delta, 
                                          const K& kernel) 
{
    Matrix<MainType, 2, 4> dEdq; 
    
    // Compute each partial derivative as a finite-difference approximation
    Matrix<MainType, 8, 1> x;
    x << r1(0), r1(1), n1(0), n1(1), r2(0), r2(1), n2(0), n2(1);
    std::cout << x.transpose() << std::endl; 
    for (int i = 0; i < 8; ++i)
    {
        Matrix<MainType, 3, 1> rt1 = Matrix<MainType, 3, 1>::Zero();
        Matrix<MainType, 3, 1> nt1 = Matrix<MainType, 3, 1>::Zero();
        Matrix<MainType, 3, 1> rt2 = Matrix<MainType, 3, 1>::Zero();
        Matrix<MainType, 3, 1> nt2 = Matrix<MainType, 3, 1>::Zero();
        x(i) += delta;
        rt1.head(2) = x(Eigen::seq(0, 1)); 
        nt1.head(2) = x(Eigen::seq(2, 3)) / x(Eigen::seq(2, 3)).norm(); 
        rt2.head(2) = x(Eigen::seq(4, 5)); 
        nt2.head(2) = x(Eigen::seq(6, 7)) / x(Eigen::seq(6, 7)).norm();
        MainType term1 = attractivePotential2D<MainType, PreciseType>(
            rt1, nt1, half_l1, rt2, nt2, half_l2, kernel
        );
        x(i) -= 2 * delta;
        rt1.head(2) = x(Eigen::seq(0, 1)); 
        nt1.head(2) = x(Eigen::seq(2, 3)) / x(Eigen::seq(2, 3)).norm(); 
        rt2.head(2) = x(Eigen::seq(4, 5)); 
        nt2.head(2) = x(Eigen::seq(6, 7)) / x(Eigen::seq(6, 7)).norm();
        MainType term2 = attractivePotential2D<MainType, PreciseType>(
            rt1, nt1, half_l1, rt2, nt2, half_l2, kernel
        );
        int j = static_cast<int>(i / 4);
        int k = i % 4;
        dEdq(j, k) = (term1 - term2) / (2 * delta);
        x(i) += delta;
    }

    return dEdq;
}

#endif
