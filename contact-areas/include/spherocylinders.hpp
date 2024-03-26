/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/26/2024
 */

#ifndef SPHEROCYLINDERS_HPP
#define SPHEROCYLINDERS_HPP

#include <cmath>
#include <complex>
#include <functional>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include "quadrics.hpp"
#include "homotopies.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Point_3 Point_3; 
typedef K::Segment_3 Segment_3;

template <typename RealType>
class Spherocylinder
{
    private:
        // Center, orientation vector, length, half-length, and radius
        Matrix<RealType, 3, 1> r; 
        Matrix<RealType, 3, 1> n;
        RealType l;
        RealType half_l;
        RealType R;

        // Polynomials for the cylinder and the two spheres ... 
        MultivariatePolynomial<RealType, 3> cylinder_; 
        MultivariatePolynomial<RealType, 3> sphere1_;
        MultivariatePolynomial<RealType, 3> sphere2_; 
        
        // ... along with the planar constraints
        MultivariatePolynomial<RealType, 3> plane1_;
        MultivariatePolynomial<RealType, 3> plane2_;

    public:
        /**
         * Constructor from input center, orientation vector, length, and
         * radius.
         *
         * @param r Spherocylinder center. 
         * @param n Spherocylinder orientation vector. 
         * @param l Spherocylinder length. 
         * @param R Spherocylinder radius.  
         */
        Spherocylinder(const Ref<const Matrix<RealType, 3, 1> >& r,
                       const Ref<const Matrix<RealType, 3, 1> >& n,
                       const RealType l, const RealType R)
        {
            // Store the given data
            this->r = r;
            this->n = n;
            this->l = l;
            this->half_l = l / 2;
            this->R = R;

            // Get the centers of the spheres 
            Matrix<RealType, 3, 1> c1 = r - (l / 2) * n;
            Matrix<RealType, 3, 1> c2 = r + (l / 2) * n;

            // Get the polynomials for the cylinder and the spheres 
            this->cylinder_ = getCylinder<RealType>(r, n, R);
            this->sphere1_ = getSphere<RealType>(c1, R); 
            this->sphere2_ = getSphere<RealType>(c2, R);

            // Get the polynomials for the planar constraints 
            this->plane1_ = getPlane<RealType>(c1, n);
            this->plane2_ = getPlane<RealType>(c2, n); 
        }

        /**
         * Return the cylinder polynomial.
         *
         * @returns Polynomial for the cylinder in Cartesian coordinates. 
         */
        MultivariatePolynomial<RealType, 3> cylinder()
        {
            return this->cylinder_; 
        }

        /**
         * Return the first sphere polynomial (corresponding to the centerline
         * coordinate s = -l / 2).
         *
         * @returns Polynomial for the sphere at s = -l / 2 in Cartesian 
         *          coordinates. 
         */
        MultivariatePolynomial<RealType, 3> sphere1()
        {
            return this->sphere1_;
        }

        /**
         * Return the second sphere polynomial (corresponding to the centerline
         * coordinate s = l / 2).
         *
         * @returns Polynomial for the sphere at s = l / 2 in Cartesian 
         *          coordinates. 
         */
        MultivariatePolynomial<RealType, 3> sphere2()
        {
            return this->sphere2_;
        }

        /**
         * Return the first plane polynomial (corresponding to the centerline 
         * coordinate s = -l / 2).
         * 
         * @returns Polynomial for the plane at s = -l / 2 in Cartesian 
         *          coordinates. 
         */
        MultivariatePolynomial<RealType, 3> plane1()
        {
            return this->plane1_; 
        }

        /**
         * Return the second plane polynomial (corresponding to the centerline
         * coordinate s = l / 2).
         *
         * @returns Polynomial for the plane at s = l / 2 in Cartesian 
         *          coordinates. 
         */
        MultivariatePolynomial<RealType, 3> plane2()
        {
            return this->plane2_;
        }

        /**
         * Given a query point, return the coordinate along the spherocylinder
         * centerline that is nearest to the query point.
         *
         * @param p Query point. 
         * @returns Centerline coordinate that is nearest to p.  
         */
        RealType nearestCoord(const Ref<const Matrix<RealType, 3, 1> >& p)
        {
            RealType s = (p - this->r).dot(this->n);
            if (std::abs(s) <= this->half_l)
                return s; 
            else if (s > this->half_l)
                return this->half_l;
            else    // s < -this->half_l
                return -this->half_l;
        }

        /**
         * Given a point and normal vector, return the coordinate along the 
         * spherocylinder centerline that crosses the corresponding plane.
         *
         * Here, the centerline is assumed to extend infinitely, and the 
         * coordinate may not actually be contained within the spherocylinder. 
         *
         * If no such coordinate exists (because the centerline is parallel
         * to the plane), then NaN is returned.
         *
         * @param p Input point. 
         * @param v Input normal vector.  
         * @returns Centerline coordinate (which may not lie between -l / 2
         *          and l / 2) that crosses the plane containing p that is 
         *          normal to v. 
         */
        RealType planeIntersect(const Ref<const Matrix<RealType, 3, 1> >& p,
                                const Ref<const Matrix<RealType, 3, 1> >& v)
        {
            // If the normal vector is orthogonal to the centerline, then 
            // the plane is parallel to the centerline and there is no such
            // intersection
            RealType prod = v.dot(this->n); 
            if (v.dot(this->n) == 0)
                return std::numeric_limits<RealType>::quiet_NaN();
            // Otherwise, get the intersection 
            else 
                return v.dot(p - this->r) / prod;
        }

        /**
         * Return the centerline as a Segment_3 instance.
         *
         * @returns Centerline as a Segment_3 instance. 
         */
        Segment_3 segment()
        {
            Matrix<RealType, 3, 1> p = this->r - this->half_l * this->n; 
            Matrix<RealType, 3, 1> q = this->r + this->half_l * this->n;
            Point_3 p_(
                static_cast<double>(p(0)),
                static_cast<double>(p(1)), 
                static_cast<double>(p(2))
            );
            Point_3 q_(
                static_cast<double>(q(0)),
                static_cast<double>(q(1)),
                static_cast<double>(q(2))
            ); 
            return Segment_3(p_, q_);
        }
};

/**
 * Approximate the contact region between:
 *
 * (1) the spherocylinder centered at the origin with orientation (0, 0, 1)
 *     and of given length and radius, and
 * (2) the spherocylinder centered at the given point with the given
 *     orientation, length, and radius.
 *
 * A start system and its roots must be supplied by the user.
 *
 * @param r2 Center of spherocylinder 2.
 * @param n2 Orientation vector of spherocylinder 2.
 * @param l1 Length of spherocylinder 1.
 * @param R1 Radius of spherocylinder 1.
 * @param l2 Length of spherocylinder 2.
 * @param R2 Radius of spherocylinder 2.
 * @param g Start system of polynomials (in two variables, x and y). 
 * @param g_roots Roots to the start system.
 * @param meshsize Number of centerline coordinates at which to solve for 
 *                 roots. 
 * @param rkA Runge-Kutta matrix. 
 * @param rkb Runge-Kutta weights.
 * @param rkc Runge-Kutta nodes.  
 * @param track_tol Tracking tolerance for homotopy continuation. 
 * @param correct_tol Correction tolerance for homotopy continuation. 
 * @param max_correct_iter Maximum number of Newton corrections per iteration
 *                         during homotopy continuation. 
 * @param min_dt Minimum stepsize for homotopy continuation.
 * @param max_dt Maximum stepsize for homotopy continuation.
 * @param imag_tol Tolerance for judging whether a complex root is real or 
 *                 imaginary. 
 * @param inf_tol Tolerance for judging whether a complex root is finite or 
 *                infinite. 
 * @returns A matrix of points (x0, y0, x1, y1, z), where (x0, y0, z) and 
 *          (x1, y1, z) are points of intersection between the spherocylinders.  
 */
template <typename RealType, int NStages>
std::pair<Matrix<RealType, Dynamic, 5>, RealType>
    getContactRegion(const Ref<const Matrix<RealType, 3, 1> >& r2,
                     const Ref<const Matrix<RealType, 3, 1> >& n2,
                     const RealType l1, 
                     const RealType R1,
                     const RealType l2,
                     const RealType R2,
                     std::array<MultivariatePolynomial<RealType, 2>, 2>& g,
                     const Ref<const Matrix<std::complex<RealType>, Dynamic, 2> >& g_roots,
                     const int meshsize,
                     const Ref<const Matrix<RealType, NStages, NStages> >& rkA,
                     const Ref<const Matrix<RealType, NStages, 1> >& rkb,
                     const Ref<const Matrix<RealType, NStages, 1> >& rkc,
                     const RealType track_tol, 
                     const RealType correct_tol, 
                     const int max_correct_iter,
                     const RealType min_dt,
                     const RealType max_dt,
                     const RealType imag_tol,
                     const RealType inf_tol)
{
    Matrix<RealType, 3, 1> r1 = Matrix<RealType, 3, 1>::Zero();
    Matrix<RealType, 3, 1> n1; 
    n1 << 0, 0, 1;
    RealType half_l1 = l1 / 2;
    RealType half_l2 = l2 / 2;

    // Define the two spherocylinders 
    Spherocylinder<RealType> sc1(r1, n1, l1, R1);
    Spherocylinder<RealType> sc2(r2, n2, l2, R2);

    // Get the angle formed by the spherocylinders
    RealType nprod = n1.dot(n2);
    bool not_orthogonal = (nprod > 1e-8);

    // Get the two planes for spherocylinder 2
    MultivariatePolynomial<RealType, 3> plane1 = sc2.plane1(); 
    MultivariatePolynomial<RealType, 3> plane2 = sc2.plane2();

    // Initialize matrix of points that trace the contact region
    //
    // Each row stores a pair of points at the same z-coordinate
    int npairs = 0;
    Matrix<RealType, Dynamic, 5> region(npairs, 5);

    // Initialize enclosed area 
    RealType area = 0;

    // Define functions that check whether a point lies along the cylinder
    // or within the two hemispherical caps 
    std::function<bool(const Ref<const Matrix<std::complex<RealType>, 3, 1> >&)>
        constraint_sphere1, constraint_sphere2, constraint_cylinder; 
    constraint_sphere1 =
        [&plane1](const Ref<const Matrix<std::complex<RealType>, 3, 1> >& v) -> bool
        {
            return std::real(plane1.eval(v)) < 0;
        };
    constraint_sphere2 =
        [&plane2](const Ref<const Matrix<std::complex<RealType>, 3, 1> >& v) -> bool
        {
            return std::real(plane2.eval(v)) > 0;
        };
    constraint_cylinder = 
        [&plane1, &plane2](const Ref<const Matrix<std::complex<RealType>, 3, 1> >& v) -> bool
        {
            return std::real(plane1.eval(v)) > 0 && std::real(plane2.eval(v)) < 0;
        };

    // Initialize a straight-line homotopy with a placeholder end system 
    std::array<MultivariatePolynomial<RealType, 2>, 2> f;
    f[0] = sc1.cylinder().eval(2, 0); 
    f[1] = sc2.cylinder().eval(2, 0);
    ProjectiveStraightLineHomotopy<RealType, 2, NStages> h(g, f, g_roots, rkA, rkb, rkc);

    // For each point along the centerline of spherocylinder 1 ...
    RealType smin = -half_l1 - R1; 
    RealType smax = -smin;
    Matrix<RealType, Dynamic, 1> zmesh
        = Matrix<RealType, Dynamic, 1>::LinSpaced(meshsize, smin, smax);
    for (int i = 0; i < meshsize; ++i)
    {
        RealType z0 = zmesh(i);
        Matrix<RealType, 3, 1> p = z0 * n1;

        // Initialize array of two polynomials in two variables 
        //
        // The first polynomial is for either sphere or the cylinder in 
        // spherocylinder 1, depending on the value of z0
        if (z0 <= -half_l1)
            f[0] = sc1.sphere1().eval(2, z0);
        else if (z0 >= half_l1)
            f[0] = sc1.sphere2().eval(2, z0);
        else    // z0 > -half_l1 and z0 < half_l1
            f[0] = sc1.cylinder().eval(2, z0);
        std::cout << f[0].toString() << std::endl;

        // First, try calculating a circle-cylinder intersection
        std::cout << z0 << " getting circle-cylinder\n";
        f[1] = sc2.cylinder().eval(2, z0);
        h.setEnd(f);
        Matrix<std::complex<RealType>, Dynamic, 2> roots = h.solve(
            track_tol, correct_tol, max_correct_iter, min_dt, max_dt
        );
        std::cout << "solved\n";

        // Keep track of the finite real roots that satisfy the corresponding 
        // planar constraints pertaining to the cylinder in spherocylinder 2
        std::vector<int> idx_real; 
        for (int j = 0; j < roots.rows(); ++j)
        {
            // Check that the root is finite
            if (roots.row(j).norm() < inf_tol)
            {
                // Check that the root is real
                RealType xim = std::abs(std::imag(roots(j, 0)));
                RealType yim = std::abs(std::imag(roots(j, 1))); 
                if (xim < imag_tol && yim < imag_tol)
                {
                    // Check that the root satisfies the corresponding 
                    // planar constraints pertaining to the cylinder in 
                    // spherocylinder 2
                    Matrix<std::complex<RealType>, 3, 1> root_3d; 
                    root_3d << roots(j, 0), roots(j, 1), z0;
                    if (constraint_cylinder(root_3d))
                        idx_real.push_back(j);
                }
            }
        }

        // Are there two finite real roots?
        if (idx_real.size() == 2)
        {
            int j = idx_real[0];
            int k = idx_real[1];
            npairs++;
            region.conservativeResize(npairs, 5);
            region(npairs - 1, 0) = std::real(roots(j, 0));
            region(npairs - 1, 1) = std::real(roots(j, 1)); 
            region(npairs - 1, 2) = std::real(roots(k, 0)); 
            region(npairs - 1, 3) = std::real(roots(k, 1)); 
            region(npairs - 1, 4) = z0;
           
            // If there is a previous pair of roots that were identified, 
            // compute the area contributed by the new roots 
            if (npairs >= 2)
            {
                Matrix<RealType, 2, 1> v = region(npairs - 1, Eigen::seq(0, 1)); 
                Matrix<RealType, 2, 1> w = region(npairs - 1, Eigen::seq(2, 3));
                RealType length = (v - w).norm();
                RealType zdelta = z0 - region(npairs - 2, 4);
                area += (zdelta * length);
            }

            // Move onto the next value of z0
            continue;
        }

        // Otherwise, try calculating a circle-sphere intersection
        std::cout << z0 << " getting circle-sphere\n";
        RealType s = sc2.nearestCoord(p);
        if (s < 0)
            f[1] = sc2.sphere1().eval(2, z0); 
        else 
            f[1] = sc2.sphere2().eval(2, z0);
        h.setEnd(f);
        roots = h.solve(track_tol, correct_tol, max_correct_iter, min_dt, max_dt);
        std::cout << "solved\n";

        // Keep track of the finite real roots that satisfy the corresponding 
        // planar constraints pertaining to the either hemispherical cap in
        // spherocylinder 2
        idx_real.clear(); 
        for (int j = 0; j < roots.rows(); ++j)
        {
            // Check that the root is finite
            if (roots.row(j).norm() < inf_tol)
            {
                // Check that the root is real
                RealType xim = std::abs(std::imag(roots(j, 0)));
                RealType yim = std::abs(std::imag(roots(j, 1))); 
                if (xim < imag_tol && yim < imag_tol)
                {
                    // Check that the root satisfies the corresponding 
                    // planar constraints pertaining to either hemispherical
                    // cap in spherocylinder 2
                    Matrix<std::complex<RealType>, 3, 1> root_3d; 
                    root_3d << roots(j, 0), roots(j, 1), z0;
                    if (s < 0)
                    {
                        if (constraint_sphere1(root_3d))
                            idx_real.push_back(j);
                    }
                    else
                    { 
                        if (constraint_sphere2(root_3d))
                            idx_real.push_back(j);
                    }
                }
            }
        }

        // Are there two finite real roots?
        if (idx_real.size() == 2)
        {
            int j = idx_real[0];
            int k = idx_real[1];
            npairs++;
            region.conservativeResize(npairs, 5);
            region(npairs - 1, 0) = std::real(roots(j, 0));
            region(npairs - 1, 1) = std::real(roots(j, 1)); 
            region(npairs - 1, 2) = std::real(roots(k, 0)); 
            region(npairs - 1, 3) = std::real(roots(k, 1)); 
            region(npairs - 1, 4) = z0;
           
            // If there is a previous pair of roots that were identified, 
            // compute the area contributed by the new roots 
            if (npairs >= 2)
            {
                Matrix<RealType, 2, 1> v = region(npairs - 1, Eigen::seq(0, 1)); 
                Matrix<RealType, 2, 1> w = region(npairs - 1, Eigen::seq(2, 3));
                RealType length = (v - w).norm();
                RealType zdelta = z0 - region(npairs - 2, 4);
                area += (zdelta * length);
            }
        }
    }

    return std::make_pair(region, area);
}

#endif 
