/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/20/2024
 */

#ifndef HOMOTOPIES_HPP
#define HOMOTOPIES_HPP

#include <tuple>
#include <array>
#include <complex>
#include <Eigen/Dense>
#include "polynomials.hpp"

using namespace Eigen;

/**
 * An implementation of Euler's method. 
 *
 * This function applies one iteration of Euler's method on a differential 
 * equation of the form 
 *
 * J(x(t)) * dx/dt = u(x(t), t),
 *
 * where x has dimension N + 1, u(x, t) has dimension N, and J has dimension
 * (N, N + 1).
 *
 * Note that we are decreasing t by dt. 
 */
template <typename RealType, int N>
Matrix<std::complex<RealType>, N + 1, 1> euler(
    const std::function<
        Matrix<std::complex<RealType>, N, N + 1>(
            const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >&,
            const RealType
        )
    >& J,
    const std::function<
        Matrix<std::complex<RealType>, N, 1>(
            const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >&,
            const RealType
        )
    >& u,
    const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >& x,
    const RealType t, const RealType dt)
{
    // Evaluate J and u at the given values of x and t 
    Matrix<std::complex<RealType>, N + 1, 1> k = (-J(x, t)).householderQr().solve(u(x, t)); 

    // Return the update 
    return dt * k;
}

/**
 * An implementation of a Runge-Kutta method with a user-specified tableau.
 *
 * This function applies one iteration of the specified Runge-Kutta method 
 * on a differential equation of the form 
 *
 * J(x(t)) * dx/dt = u(x(t), t),
 *
 * where x has dimension N + 1, u(x, t) has dimension N, and J has dimension
 * (N, N + 1).
 *
 * Note that we are decreasing t by dt.
 */
template <typename RealType, int N, int S>
Matrix<std::complex<RealType>, N + 1, 1> rungeKutta(
    const std::function<
        Matrix<std::complex<RealType>, N, N + 1>(
            const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >&,
            const RealType
        )
    >& J,
    const std::function<
        Matrix<std::complex<RealType>, N, 1>(
            const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >&,
            const RealType
        )
    >& u,
    const Ref<const Matrix<std::complex<RealType>, N + 1, 1> >& x,
    const RealType t, const RealType dt,
    const Ref<const Matrix<RealType, S, S> >& A,
    const Ref<const Matrix<RealType, S, 1> >& b,
    const Ref<const Matrix<RealType, S, 1> >& c)
{
    // Compute the terms in the Runge-Kutta summation 
    Matrix<std::complex<RealType>, N + 1, S> k;
    for (int s = 0; s < S; ++s)
    {
        // The term corresponding to stage s has contributions from the 
        // previous stages 
        Matrix<std::complex<RealType>, N + 1, 1> y = Matrix<std::complex<RealType>, N + 1, 1>::Zero();
        for (int i = 0; i < s; ++i)
            y += A(s, i) * k.col(i); 
        RealType ts = t - c(s) * dt;
        Matrix<std::complex<RealType>, N + 1, 1> xs = x + y * dt;
        k.col(s) = (-J(xs, ts)).householderQr().solve(u(xs, ts));
    }

    // Return the update 
    Matrix<std::complex<RealType>, N + 1, 1> update = Matrix<std::complex<RealType>, N + 1, 1>::Zero();
    for (int s = 0; s < S; ++s)
        update += b(s) * k.col(s);
    update *= dt; 
    
    return update;
}

/**
 * Return the Butcher tableau for Euler's method.
 */
template <typename RealType>
std::tuple<Matrix<RealType, 1, 1>, Matrix<RealType, 1, 1>, Matrix<RealType, 1, 1> > eulerTableau()
{
    Matrix<RealType, 1, 1> rkA = Matrix<RealType, 1, 1>::Zero(); 
    Matrix<RealType, 1, 1> rkb = Matrix<RealType, 1, 1>::Ones(); 
    Matrix<RealType, 1, 1> rkc = Matrix<RealType, 1, 1>::Zero(); 

    return std::make_tuple(rkA, rkb, rkc);
}

/**
 * Return the Butcher tableau for the classic 4th-order Runge-Kutta method.
 */
template <typename RealType>
std::tuple<Matrix<RealType, 4, 4>, Matrix<RealType, 4, 1>, Matrix<RealType, 4, 1> > rk4Tableau()
{
    Matrix<RealType, 4, 4> rkA = Matrix<RealType, 4, 4>::Zero(); 
    rkA(1, 0) = 0.5;
    rkA(2, 1) = 0.5;
    rkA(3, 2) = 1.0;
    Matrix<RealType, 4, 1> rkb, rkc;
    rkb << 1./6., 1./3., 1./3., 1./6.;
    rkc << 0.0, 0.5, 0.5, 1.0;

    return std::make_tuple(rkA, rkb, rkc);
}

/**
 * A basic straight-line homotopy class that performs homotopy continuation
 * in projective coordinates.
 */
template <typename RealType, int NVariables, int NStages>
class ProjectiveStraightLineHomotopy
{
    typedef std::complex<RealType> ComplexType;
    typedef MultivariatePolynomial<RealType, NVariables> AffinePolynomial;
    typedef MultivariatePolynomial<RealType, NVariables + 1> HomogeneousPolynomial;

    private:
        // Homogenized start and end systems 
        std::array<HomogeneousPolynomial, NVariables> g;
        std::array<HomogeneousPolynomial, NVariables> f;

        // Roots for the homogenized start system 
        Matrix<ComplexType, Dynamic, NVariables + 1> g_roots;

        // Partial derivatives of the homogenized start and end systems 
        std::array<std::array<HomogeneousPolynomial, NVariables + 1>, NVariables> gderivs;
        std::array<std::array<HomogeneousPolynomial, NVariables + 1>, NVariables> fderivs;

        // Elements of the Runge-Kutta Butcher tableau
        Matrix<RealType, NStages, NStages> rkA; 
        Matrix<RealType, NStages, 1> rkb; 
        Matrix<RealType, NStages, 1> rkc;

    public:
        /**
         * Constructor with input polynomials and roots for the start system
         * in affine coordinates, plus the Runge-Kutta Butcher tableau.
         */
        ProjectiveStraightLineHomotopy(std::array<AffinePolynomial, NVariables>& g,
                                       std::array<AffinePolynomial, NVariables>& f,
                                       const Ref<const Matrix<ComplexType, Dynamic, NVariables> >& g_roots,
                                       const Ref<const Matrix<RealType, NStages, NStages> >& rkA,
                                       const Ref<const Matrix<RealType, NStages, 1> >& rkb,
                                       const Ref<const Matrix<RealType, NStages, 1> >& rkc)
        {
            // Homogenize the input polynomials
            for (int i = 0; i < NVariables; ++i)
            {
                this->g[i] = g[i].homogenize();
                this->f[i] = f[i].homogenize();
            }

            // Homogenize the starting roots
            int nroots = g_roots.rows();
            this->g_roots = Matrix<ComplexType, Dynamic, NVariables + 1>::Zero(nroots, NVariables + 1);
            for (int i = 0; i < nroots; ++i)
            {
                this->g_roots(i, Eigen::seqN(0, NVariables)) = g_roots.row(i); 
                this->g_roots(i, NVariables) = 1;
            }

            // Normalize the starting roots
            for (int i = 0; i < nroots; ++i)
            {
                RealType root_norm = this->g_roots.row(i).norm();
                this->g_roots.row(i) /= root_norm;
            }

            // Differentiate each polynomial with respect to each variable
            // and store them in an array of arrays
            //
            // The (i, j)-th coordinate is the derivative of polynomial i 
            // with respect to variable j, where i runs through 0, ...,
            // NVariables - 1 and j runs through 0, ..., NVariables
            for (int i = 0; i < NVariables; ++i)
            {
                for (int j = 0; j < NVariables + 1; ++j)
                {
                    this->gderivs[i][j] = this->g[i].deriv(j);
                    this->fderivs[i][j] = this->f[i].deriv(j);
                }
            }

            // Set the Runge-Kutta Butcher tableau
            this->rkA = rkA; 
            this->rkb = rkb;
            this->rkc = rkc;
        }

        /**
         * Update the start system to the given array of polynomials, and 
         * update the corresponding roots.  
         */
        void setStart(std::array<AffinePolynomial, NVariables>& g,
                      const Ref<const Matrix<ComplexType, Dynamic, NVariables> >& g_roots)
        {
            // Homogenize the input polynomials 
            for (int i = 0; i < NVariables; ++i)
                this->g[i] = g[i].homogenize(); 

            // Homogenize the starting roots 
            int nroots = g_roots.rows();
            this->g_roots = Matrix<ComplexType, Dynamic, NVariables + 1>::Zero(nroots, NVariables + 1);
            for (int i = 0; i < nroots; ++i)
            {
                this->g_roots(i, Eigen::seqN(0, NVariables)) = g_roots.row(i); 
                this->g_roots(i, NVariables) = 1;
            }

            // Normalize the starting roots
            for (int i = 0; i < nroots; ++i)
            {
                RealType root_norm = this->g_roots.row(i).norm();
                this->g_roots.row(i) /= root_norm;
            }

            // Differentiate each polynomial with respect to each variable 
            for (int i = 0; i < NVariables; ++i)
            {
                for (int j = 0; j < NVariables + 1; ++j)
                {
                    this->gderivs[i][j] = this->g[i].deriv(j);
                }
            }
        }

        /**
         * Update the end system to the given array of polynomials.
         */
        void setEnd(std::array<AffinePolynomial, NVariables>& f)
        {
            // Homogenize the input polynomials 
            for (int i = 0; i < NVariables; ++i)
                this->f[i] = f[i].homogenize(); 

            // Differentiate each polynomial with respect to each variable 
            for (int i = 0; i < NVariables; ++i)
            {
                for (int j = 0; j < NVariables + 1; ++j)
                {
                    this->fderivs[i][j] = this->f[i].deriv(j);
                }
            }
        }

        /**
         * Evaluate the straight-line homotopy at the given values of the 
         * variables (including the homogenizing variable) and t.
         */
        Matrix<ComplexType, NVariables, 1> eval(const Ref<const Matrix<ComplexType, NVariables + 1, 1> >& values,
                                                const RealType t)
        {
            // Evaluate each polynomial in each system at the given values
            Matrix<ComplexType, NVariables, 1> gvalues = Matrix<ComplexType, NVariables, 1>::Zero();
            Matrix<ComplexType, NVariables, 1> fvalues = Matrix<ComplexType, NVariables, 1>::Zero();
            for (int i = 0; i < NVariables; ++i)
            {
                gvalues(i) = this->g[i].eval(values);
                fvalues(i) = this->f[i].eval(values);
            }

            // Evaluate the homotopy 
            return t * gvalues + (1 - t) * fvalues;
        }

        /**
         * Use a Runge-Kutta predictor (with the given Butcher tableau) and a
         * Newton corrector to solve for the roots of the end system.
         */
        Matrix<ComplexType, Dynamic, NVariables> solve(
            const RealType tol, const RealType correct_tol,
            const int max_correct_iter, const RealType min_dt,
            const RealType max_dt)
        {
            int nroots = this->g_roots.rows();
            Matrix<ComplexType, Dynamic, NVariables + 1> roots_end_hom(nroots, NVariables + 1);

            // Define the Jacobian matrix function for the homotopy 
            auto jacobian = [this](
                const Ref<const Matrix<ComplexType, NVariables + 1, 1> >& x,
                const RealType t
            ) -> Matrix<ComplexType, NVariables, NVariables + 1>
            {
                Matrix<ComplexType, NVariables, NVariables + 1> J;
                for (int i = 0; i < NVariables; ++i)
                {
                    for (int j = 0; j < NVariables + 1; ++j)
                    {
                        J(i, j) = t * this->gderivs[i][j].eval(x) + (1 - t) * this->fderivs[i][j].eval(x);
                    }
                }
                return J;
            };

            // Define the partial derivative of the homotopy with respect to
            // t, as a function of the variables and t
            auto tderiv = [this](
                const Ref<const Matrix<ComplexType, NVariables + 1, 1> >& x,
                const RealType t
            ) -> Matrix<ComplexType, NVariables, 1>
            {
                Matrix<ComplexType, NVariables, 1> u; 
                for (int i = 0; i < NVariables; ++i)
                    u(i) = this->g[i].eval(x) - this->f[i].eval(x);
                return u;
            };

            // For each root ...
            for (int i = 0; i < nroots; ++i)
            {
                Matrix<ComplexType, NVariables + 1, 1> root_curr = this->g_roots.row(i).transpose();

                // Keep track of whether the last five steps were successful 
                Array<bool, 5, 1> success = Array<bool, 5, 1>::Zero();

                // Starting from t = 1 ...
                RealType t_curr = 1.0;
                RealType dt = max_dt;
                while (t_curr > tol)
                {
                    RealType t_next = t_curr - dt;

                    // Compute the Runge-Kutta predictor update
                    //
                    // Note that each root now has dimension (NVariables + 1),
                    // and so the system is not square 
                    Matrix<ComplexType, NVariables + 1, 1> update = rungeKutta<RealType, NVariables, NStages>(
                        jacobian, tderiv, root_curr, t_curr, dt, this->rkA,
                        this->rkb, this->rkc
                    );
                    Matrix<ComplexType, NVariables + 1, 1> root_next = root_curr + update;

                    // Evaluate the homotopy at the updated root 
                    Matrix<ComplexType, NVariables, 1> h_next = this->eval(root_next, t_next);

                    // Compute the Newton corrector update up to a maximum number
                    // of iterations, if necessary
                    int correct_iter = 0;
                    while (correct_iter < max_correct_iter && h_next.norm() > correct_tol)
                    {
                        // Compute the Newton corrector update
                        //
                        // Note that each root now has dimension (NVariables + 1),
                        // and so the system is not square
                        Matrix<ComplexType, NVariables, NVariables + 1> J_next
                            = jacobian(root_next, t_next);
                        Matrix<ComplexType, NVariables + 1, 1> newton_update
                            = (-J_next).householderQr().solve(h_next);
                        root_next += newton_update; 

                        // Re-evaluate the homotopy at the updated root
                        h_next = this->eval(root_next, t_next);
                        correct_iter++;
                    }

                    // If, after the maximum number of corrections, the updated
                    // root is not within the desired tolerance, forgo the
                    // update and try again with a smaller stepsize (unless 
                    // the stepsize is already too small)
                    if (h_next.norm() > correct_tol && dt > min_dt)
                    {
                        for (int i = 0; i < 4; ++i)
                            success(i) = success(i + 1);
                        success(4) = false;
                        dt /= 2;
                    }
                    // Otherwise, update root and current time, and if the past
                    // five iterations were all successful (including the current
                    // iteration), then increase the stepsize (unless the stepsize
                    // is already too large)
                    else
                    {
                        for (int i = 0; i < 4; ++i)
                            success(i) = success(i + 1); 
                        success(4) = true;
                        t_curr = t_next;
                        RealType root_norm = root_next.norm();
                        root_curr = root_next / root_norm;
                        if (success.all() && dt < max_dt)
                            dt *= 2;
                    }
                }

                roots_end_hom.row(i) = root_curr.transpose();
            }

            // Dehomogenize the roots
            Matrix<ComplexType, Dynamic, NVariables> roots_end(nroots, NVariables);
            for (int i = 0; i < nroots; ++i)
            {
                for (int j = 0; j < NVariables; ++j)
                {
                    roots_end(i, j) = roots_end_hom(i, j) / roots_end_hom(i, NVariables);
                }
            }

            return roots_end;
        }

};

#endif
