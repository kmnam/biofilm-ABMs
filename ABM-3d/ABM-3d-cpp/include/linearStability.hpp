/**
 * Functions for linear stability analysis and simulating post-instability
 * dynamics.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     9/17/2026
 */

#ifndef LINEAR_STABILITY_ANALYSIS_HPP
#define LINEAR_STABILITY_ANALYSIS_HPP

#include <iostream>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "duals.hpp"
#include "distances.hpp"
#include "integrals.hpp"
#include "rootFinding.hpp"

using namespace Eigen;

using std::sin;
using std::sqrt; 
using std::pow;
using std::min;
using std::max; 
using std::clamp; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

using Dual::DualNumber; 
using Dual::DualVector; 
using Dual::DualMatrix;

enum class JKRCellBodyContactMode
{
    DisallowContacts, 
    IgnoreCellBody,
    AllowRepulsiveContacts
};

/**
 * Calculate the 6x6 dissipation matrix specifying the viscosity forces
 * on the given cell.  
 *
 * @param r Cell position. 
 * @param n Cell orientation. 
 * @param length Cell length. 
 * @param R Cell radius. 
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient. 
 * @returns The 6x6 dissipation matrix specifying the viscosity forces. 
 */
template <typename T>
Matrix<T, 6, 6> getDissipationMatrix(const Ref<const Matrix<T, 3, 1> >& r, 
                                     const Ref<const Matrix<T, 3, 1> >& n,
                                     const T length, const T R, const T eta0,
                                     const T eta1) 
{
    Matrix<T, 6, 6> A = Matrix<T, 6, 6>::Zero();  

    // If the cell is horizontal ...
    if (n(2) == 0)
    {
        // Get the cell-surface overlap 
        T overlap = (r(2) < R ? R - r(2) : 0.0);

        // Calculate the cell-surface contact area density 
        T area_density = (overlap > 0 ? sqrt(R) * sqrt(overlap) : 0.0);

        // Get the generalized friction coefficients
        T l3 = length * length * length;  
        T K = (eta0 + (eta1 / R) * area_density) * length;
        T L = (eta0 + (eta1 / R) * area_density) * l3 / 12;  
        A(0, 0) = K;  
        A(1, 1) = K; 
        A(2, 2) = eta0 * length;
        A(3, 3) = L;
        A(4, 4) = L; 
        A(5, 5) = eta0 * l3 / 12;
    }
    else                            // If the cell is tilted ... 
    {
        // Get the cell-surface overlap at the two endpoints 
        const T half_l = length / 2;
        const T l3 = length * length * length; 
        T overlap = (r(2) < R ? R - r(2) : 0.0); 
        T delta_p = R - (r(2) - half_l * n(2)); 
        T delta_q = R - (r(2) + half_l * n(2));

        // If the cell fully contacts the surface ...
        T int1 = 0; 
        T int2 = 0; 
        T int3 = 0;  
        if (delta_p > 0 && delta_q > 0)
        {
            // If the cell's vertical tilt is small ... 
            const T nz_min = length * abs(n(2)) / (2 * overlap);
            if (nz_min < 0.05)
            { 
                // Calculate the cell-surface contact area integrals using a small-nz
                // Taylor expansion
                T area_density_horiz = sqrt(R) * sqrt(overlap);
                const T l5 = l3 * length * length;  
                int1 = length * area_density_horiz;    // Horizontal term (does not depend on nz)
                int1 += n(2) * n(2) * (                  // nz^2 term
                    -length * area_density_horiz
                    - l3 * area_density_horiz / (96 * overlap * overlap)
                    + boost::math::constants::pi<T>() * R * length  
                );
                int2 = -area_density_horiz * l3 / (24 * overlap) * n(2);   // Linear term 
                int3 = l3 * area_density_horiz / 12;   // Horizontal term (does not depend on nz)
                int3 += n(2) * n(2) * (                  // nz^2 term 
                    -l3 * area_density_horiz / 12
                    - l5 * area_density_horiz / (640 * overlap * overlap)
                    + boost::math::constants::pi<T>() * R * l3 / 12 
                );
            }
            else     // Otherwise, use the full surface energy formula 
            {
                // Calculate the cell-surface contact area integrals 
                T ss = (R - r(2)) / n(2); 
                auto result = areaIntegrals<T>(r(2), n(2), R, length / 2, ss);
                int1 = std::get<0>(result); 
                int2 = std::get<1>(result); 
                int3 = std::get<2>(result);
            }
        }
        else         // Otherwise, use the full surface energy formula 
        {
            // Calculate the cell-surface contact area integrals 
            T ss = (R - r(2)) / n(2); 
            auto result = areaIntegrals<T>(r(2), n(2), R, length / 2, ss);
            int1 = std::get<0>(result); 
            int2 = std::get<1>(result); 
            int3 = std::get<2>(result);
        }

        // Populate the matrix 
        A(0, 0) = eta0 * length + (eta1 / R) * int1; 
        A(0, 3) = (eta1 / R) * int2; 
        A(1, 1) = eta0 * length + (eta1 / R) * int1; 
        A(1, 4) = (eta1 / R) * int2; 
        A(2, 2) = eta0 * length; 
        A(3, 0) = (eta1 / R) * int2; 
        A(3, 3) = (eta0 * l3 / 12) + (eta1 / R) * int3; 
        A(4, 1) = (eta1 / R) * int2; 
        A(4, 4) = (eta0 * l3 / 12) + (eta1 / R) * int3;  
        A(5, 5) = eta0 * l3 / 12; 
    }

    return A; 
}

/**
 * Calculate the Hertzian cell-cell contact force on the cell, given the 
 * neighbor configuration.
 *
 * @param distances Array of cell-cell distance vectors, specified in duals. 
 *                  Each row specifies the two centerline coordinates and the
 *                  distance vector coordinates. 
 * @param R Cell radius (including the EPS).  
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @returns Dual vector specifying the generalized translational and orientational
 *          forces on the cell with the given neighbor configuration.   
 */
template <typename T>
DualVector<T> getHertzianCellCellForce(const Ref<const Matrix<DualNumber<T>, Dynamic, 5> >& distances, 
                                       const T R, const T Rcell, const T E0,
                                       const T Ecell)
{
    DualVector<T> b = DualVector<T>::Zero(6);

    // For each neighboring cell ... 
    for (int i = 0; i < distances.rows(); ++i)
    {
        // Get the distance between the two cells
        DualVector<T> dvec = distances(i, Eigen::seqN(2, 3)); 
        DualNumber<T> dist = dvec.norm(); 

        // Calculate the Hertzian force based on the distance
        DualVector<T> dnorm = dvec / dist;
        if (dist < 2 * R && dist >= 2 * Rcell)
        { 
            DualNumber<T> overlap = 2 * R - dist;
            DualVector<T> force = (4. / 3.) * E0 * sqrt(R / 2) * pow(overlap, 1.5) * dnorm;
            b(Eigen::seqN(0, 3)) += force; 
            b(Eigen::seqN(3, 3)) += distances(i, 0) * force; 
        }
        else if (dist < 2 * Rcell)
        {
            DualNumber<T> overlap = 2 * Rcell - dist; 
            DualNumber<T> term1 = (4. / 3.) * E0 * sqrt(R / 2) * pow(2 * R - 2 * Rcell, 1.5); 
            DualNumber<T> term2 = (4. / 3.) * Ecell * sqrt(Rcell / 2) * pow(overlap, 1.5);
            DualVector<T> force = (term1 + term2) * dnorm; 
            b(Eigen::seqN(0, 3)) += force; 
            b(Eigen::seqN(3, 3)) += distances(i, 0) * force;
        }
    }
    
    return -b; 
}

/**
 * Calculate the Hertzian cell-cell contact energy due to a contact between 
 * one neighboring pair of cells.
 *
 * @param dist Centerline distance from cell 1 to cell 2. 
 * @param R Cell radius (including the EPS).  
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @returns Dual number specifying the cell-cell contact energy. 
 */
template <typename T>
DualNumber<T> getHertzianCellCellEnergy(const DualNumber<T> dist, const T R,
                                        const T Rcell, const T E0,
                                        const T Ecell)
{
    // If the cell-cell overlap lies within the EPS layer ... 
    if (dist.getValue() < 2 * R && dist.getValue() >= 2 * Rcell)
    {
        DualNumber<T> overlap = 2 * R - dist; 
        return (8. / 15.) * E0 * sqrt(R / 2) * pow(overlap, 2.5); 
    }
    // If the cell-cell overlap encroaches into the cell bodies ... 
    else if (dist.getValue() < 2 * Rcell)
    {
        DualNumber<T> overlap = 2 * Rcell - dist;
        T d0 = (4 * R + 6 * Rcell) / 5.0; 
        DualNumber<T> energy = (
            (4. / 3.) * E0 * sqrt(R / 2) * pow(2 * R - 2 * Rcell, 1.5) * (d0 - dist)
        ); 
        energy += (8. / 15.) * Ecell * sqrt(Rcell / 2) * pow(overlap, 2.5);
        return energy;
    }
    else    // Otherwise, the cell-cell contact energy is zero
    {
        return DualNumber<T>(0.0, 0.0); 
    }
}

/**
 * Calculate the Hertzian cell-cell contact energy due to a contact between 
 * one neighboring pair of cells, together with the gradient of this contact
 * energy with the cell coordinates. 
 *
 * @param r1 Cell 1 center.
 * @param n1 Cell 1 orientation. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation.  
 * @param s Pair of centerline coordinates specifying the contact point.  
 * @param R Cell radius (including the EPS).  
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @returns The cell-cell contact energy, together with a matrix specifying 
 *          the gradient. 
 */
template <typename T>
std::pair<T, Matrix<T, 4, 3> > getHertzianCellCellEnergyGradient(const Ref<const Matrix<T, 3, 1> >& r1,  
                                                                 const Ref<const Matrix<T, 3, 1> >& n1,
                                                                 const Ref<const Matrix<T, 3, 1> >& r2, 
                                                                 const Ref<const Matrix<T, 3, 1> >& n2, 
                                                                 const Ref<const Matrix<T, 2, 1> >& s, 
                                                                 const T R,
                                                                 const T Rcell, 
                                                                 const T E0, 
                                                                 const T Ecell)
{
    // Set up the gradient matrix
    T energy = 0;   
    Matrix<T, 4, 3> grad = Matrix<T, 4, 3>::Zero(); 

    // For each center coordinate ... 
    for (int i = 0; i < 3; ++i)
    {
        // Get the gradient w.r.t the position of cell 1 ... 
        //
        // Calculate the distance vector using dual vectors 
        DualVector<T> r1_(3), r2_(3), n1_(3), n2_(3);
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                r1_(j) = DualNumber<T>(r1(j), 1.0); 
            else
                r1_(j) = DualNumber<T>(r1(j), 0.0); 
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        DualVector<T> p(3), q(3); 
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        DualVector<T> dvec = q - p; 

        // Calculate the Hertzian contact energy and extract the partial
        // derivative  
        DualNumber<T> energy_ = getHertzianCellCellEnergy<T>(dvec.norm(), R, Rcell, E0, Ecell); 
        if (i == 0)
            energy = energy_.getValue();  
        grad(0, i) = energy_.getDerivative();

        // Get the gradient w.r.t the position of cell 2 ... 
        //
        // Calculate the distance vector using dual vectors 
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                r2_(j) = DualNumber<T>(r2(j), 1.0); 
            else
                r2_(j) = DualNumber<T>(r2(j), 0.0); 
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        dvec = q - p;

        // Calculate the Hertzian contact energy and extract the partial
        // derivative  
        energy_ = getHertzianCellCellEnergy<T>(dvec.norm(), R, Rcell, E0, Ecell); 
        grad(1, i) = energy_.getDerivative();
    }

    // For each orientation coordinate ... 
    for (int i = 0; i < 3; ++i)
    {
        // Get the gradient w.r.t the orientation of cell 1 ... 
        //
        // Calculate the distance vector using dual vectors 
        DualVector<T> r1_(3), r2_(3), n1_(3), n2_(3);
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                n1_(j) = DualNumber<T>(n1(j), 1.0); 
            else
                n1_(j) = DualNumber<T>(n1(j), 0.0);
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        DualVector<T> p(3), q(3); 
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        DualVector<T> dvec = q - p; 

        // Calculate the Hertzian contact energy and extract the partial
        // derivative  
        DualNumber<T> energy_ = getHertzianCellCellEnergy<T>(dvec.norm(), R, Rcell, E0, Ecell); 
        grad(2, i) = energy_.getDerivative();

        // Get the gradient w.r.t the position of cell 2 ... 
        //
        // Calculate the distance vector using dual vectors 
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                n2_(j) = DualNumber<T>(n2(j), 1.0); 
            else
                n2_(j) = DualNumber<T>(n2(j), 0.0); 
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
        }
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        dvec = q - p;

        // Calculate the Hertzian contact energy and extract the partial
        // derivative  
        energy_ = getHertzianCellCellEnergy<T>(dvec.norm(), R, Rcell, E0, Ecell); 
        grad(3, i) = energy_.getDerivative();
    }

    return std::make_pair(energy, grad); 
}

/**
 * Calculate the JKR cell-cell contact force on the cell, given the neighbor
 * configuration.
 *
 * This calculation assumes that each pair of neighboring cells are further
 * than 2 * Rcell apart.  
 *
 * @param distances Array of cell-cell distance vectors, specified in duals. 
 *                  Each row specifies the two centerline coordinates and the
 *                  distance vector coordinates. 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).  
 * @param E0 Elastic modulus of the EPS.
 * @param gamma Cell-cell adhesion energy density.  
 * @returns Dual vector specifying the generalized translational and orientational
 *          forces on the cell with the given neighbor configuration.   
 */
template <typename T>
DualVector<T> getJKRCellCellForce(const Ref<const Matrix<DualNumber<T>, Dynamic, 5> >& distances, 
                                  const T R, const T Rcell, const T E0,
                                  const T gamma)
{
    DualVector<T> b = DualVector<T>::Zero(6);
    const T Req = R / 2;  

    // For each neighboring cell ... 
    for (int i = 0; i < distances.rows(); ++i)
    {
        // Get the distance between the two cells
        DualVector<T> dvec = distances(i, Eigen::seqN(2, 3)); 
        DualNumber<T> dist = dvec.norm();

        // If the overlap is nonzero ...
        if (dist.getValue() < 2 * R)
        {
            // Raise an exception if the cells are within 2 * Rcell apart 
            if (dist.getValue() < 2 * Rcell)
                throw std::runtime_error(
                    "Encountered disallowed JKR cell body contact"
                );

            // Calculate the contact radius
            //
            // Here, we use the analytical formula derived by Parteli et al.
            // based on the quartic formula
            DualNumber<T> overlap = 2 * R - dist;
            DualVector<T> dnorm = dvec / dist;
            DualNumber<T> c2 = -2 * Req * overlap;
            DualNumber<T> c1 = -4 * boost::math::constants::pi<T>() * gamma * Req * Req / E0;
            DualNumber<T> c0 = Req * Req * overlap * overlap; 
            DualNumber<T> P = -c2 * c2 / 12.0 - c0; 
            DualNumber<T> Q = -pow(c2, 3.0) / 108.0 + c0 * c2 / 3.0 - c1 * c1 / 8.0; 
            DualNumber<T> half_Q = Q / 2.0; 
            DualNumber<T> third_P = P / 3.0; 
            DualNumber<T> U = pow(
                -half_Q + sqrt(half_Q * half_Q + pow(third_P, 3.0)), 1.0 / 3.0
            ); 
            DualNumber<T> s = -5 * c2 / 6.0; 
            if (P.getValue() != 0)
                s += (U - third_P / U); 
            else 
                s -= pow(Q, 1.0 / 3.0); 
            DualNumber<T> w = sqrt(c2 + 2 * s);
            DualNumber<T> lambda = c1 / (2 * w); 
            DualNumber<T> radius = 0.5 * (w + sqrt(w * w - 4 * (c2 + s + lambda)));

            // Calculate the JKR force magnitude 
            DualNumber<T> force = (4. / 3.) * E0 * pow(radius, 3) / Req;
            force -= (4 * sqrt(boost::math::constants::pi<T>() * pow(radius, 3) * gamma * E0));  
            b(Eigen::seqN(0, 3)) += force * dnorm; 
            b(Eigen::seqN(3, 3)) += distances(i, 0) * force * dnorm;  
        }
    }
    
    return -b; 
}

/**
 * Calculate the JKR cell-cell contact energy due to a contact between one
 * neighboring pair of cells.
 *
 * If the cells are less than 2 * Rcell apart, then this function can either
 * throw an exception or assume a repulsive contact (the adhesion is turned off). 
 *
 * @param dist Centerline distance from cell 1 to cell 2. 
 * @param R Cell radius (including the EPS).  
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS.
 * @param Ecell Elastic modulus of the cell body.  
 * @param gamma Cell-cell adhesion energy density. 
 * @param mode JKR cell body contact mode.  
 * @returns Dual number specifying the cell-cell contact energy. 
 */
template <typename T>
DualNumber<T> getJKRCellCellEnergy(const DualNumber<T> dist, const T R,
                                   const T Rcell, const T E0, const T Ecell, 
                                   const T gamma, 
                                   const JKRCellBodyContactMode mode = JKRCellBodyContactMode::DisallowContacts)
{
    const T Req = R / 2;
    DualNumber<T> energy(0.0, 0.0);  

    // If the overlap is nonzero ... 
    if (dist < 2 * R)
    {
        // If the two cell bodies are contacting and such contacts are disallowed,
        // then raise an exception 
        if (mode == JKRCellBodyContactMode::DisallowContacts && dist < 2 * Rcell)
            throw std::runtime_error("Encountered disallowed JKR cell body contact");

        // Otherwise, calculate the JKR energy
        //
        // If the cell bodies are to be ignored completely, calculate the 
        // JKR energy corresponding to the full overlap  
        //
        // Otherwise, if the two cell bodies are contacting, calculate the
        // JKR energy corresponding to an overlap of 2 * R - 2 * Rcell
        DualNumber<T> dist_; 
        if (mode == JKRCellBodyContactMode::IgnoreCellBody)
        {
            dist_ = dist;
        } 
        else if (dist.getValue() < 2 * Rcell)
        {
            dist_.setValue(2 * Rcell); 
            dist_.setDerivative(0); 
        }
        else 
        { 
            dist_ = dist;
        }

        // Calculate the contact radius
        //
        // Here, we use the analytical formula derived by Parteli et al.
        // based on the quartic formula
        DualNumber<T> overlap = 2 * R - dist_;
        DualNumber<T> c2 = -2 * Req * overlap;
        DualNumber<T> c1 = -4 * boost::math::constants::pi<T>() * gamma * Req * Req / E0;
        DualNumber<T> c0 = Req * Req * overlap * overlap; 
        DualNumber<T> P = -c2 * c2 / 12.0 - c0; 
        DualNumber<T> Q = -pow(c2, 3.0) / 108.0 + c0 * c2 / 3.0 - c1 * c1 / 8.0; 
        DualNumber<T> half_Q = Q / 2.0; 
        DualNumber<T> third_P = P / 3.0; 
        DualNumber<T> U = pow(
            -half_Q + sqrt(half_Q * half_Q + pow(third_P, 3.0)), 1.0 / 3.0
        ); 
        DualNumber<T> s = -5 * c2 / 6.0; 
        if (P.getValue() != 0)
            s += (U - third_P / U); 
        else 
            s -= pow(Q, 1.0 / 3.0); 
        DualNumber<T> w = sqrt(c2 + 2 * s);
        DualNumber<T> lambda = c1 / (2 * w); 
        DualNumber<T> radius = 0.5 * (w + sqrt(w * w - 4 * (c2 + s + lambda)));

        // Calculate the JKR contact energy ...
        T a0 = pow(2 * boost::math::constants::two_pi<T>() * gamma * Req * Req / E0, 1. / 3.); 
        energy = (8. / 15.) * E0 * pow(radius, 5) / (Req * Req);
        energy -= (8. / 3.) * sqrt(boost::math::constants::pi<T>() * gamma * E0) * pow(radius, 3.5) / Req;
        energy += boost::math::constants::two_pi<T>() * pow(radius, 2) * gamma;
        energy += boost::math::constants::two_pi<T>() * gamma * a0 * a0 / 5;

        // If the two cell bodies are contacting and this contact should be 
        // counted as repulsive, add an extra Hertzian term 
        if (mode == JKRCellBodyContactMode::AllowRepulsiveContacts && dist < 2 * Rcell)
        {
            DualNumber<T> overlap_body = 2 * Rcell - dist; 
            energy += (8. / 15.) * Ecell * sqrt(Rcell / 2) * pow(overlap_body, 2.5);
        }
    }
    
    return energy;  
}

/**
 * Calculate the JKR cell-cell contact energy due to a contact between one
 * neighboring pair of cells, together with the gradient of this contact
 * energy with the cell coordinates. 
 *
 * @param r1 Cell 1 center.
 * @param n1 Cell 1 orientation. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation.  
 * @param s Pair of centerline coordinates specifying the contact point.  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).  
 * @param E0 Elastic modulus of the EPS.
 * @param Ecell Elastic modulus of the cell body. 
 * @param gamma Cell-cell adhesion energy density. 
 * @param mode JKR cell body contact mode.  
 * @returns The cell-cell contact energy, together with a matrix specifying 
 *          the gradient. 
 */
template <typename T>
std::pair<T, Matrix<T, 4, 3> > getJKRCellCellEnergyGradient(const Ref<const Matrix<T, 3, 1> >& r1,
                                                            const Ref<const Matrix<T, 3, 1> >& n1,
                                                            const Ref<const Matrix<T, 3, 1> >& r2, 
                                                            const Ref<const Matrix<T, 3, 1> >& n2, 
                                                            const Ref<const Matrix<T, 2, 1> >& s, 
                                                            const T R,
                                                            const T Rcell,
                                                            const T E0, 
                                                            const T Ecell, 
                                                            const T gamma,
                                                            const JKRCellBodyContactMode mode = JKRCellBodyContactMode::DisallowContacts)
{
    // Set up the gradient matrix
    T energy = 0;   
    Matrix<T, 4, 3> grad = Matrix<T, 4, 3>::Zero(); 

    // For each center coordinate ... 
    for (int i = 0; i < 3; ++i)
    {
        // Get the gradient w.r.t the position of cell 1 ... 
        //
        // Calculate the distance vector using dual vectors 
        DualVector<T> r1_(3), r2_(3), n1_(3), n2_(3);
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                r1_(j) = DualNumber<T>(r1(j), 1.0); 
            else
                r1_(j) = DualNumber<T>(r1(j), 0.0); 
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        DualVector<T> p(3), q(3); 
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        DualVector<T> dvec = q - p; 

        // Calculate the JKR contact energy and extract the partial
        // derivative  
        DualNumber<T> energy_ = getJKRCellCellEnergy<T>(
            dvec.norm(), R, Rcell, E0, Ecell, gamma, mode
        );
        if (i == 0)
            energy = energy_.getValue();  
        grad(0, i) = energy_.getDerivative();

        // Get the gradient w.r.t the position of cell 2 ... 
        //
        // Calculate the distance vector using dual vectors 
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                r2_(j) = DualNumber<T>(r2(j), 1.0); 
            else
                r2_(j) = DualNumber<T>(r2(j), 0.0); 
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        dvec = q - p;

        // Calculate the JKR contact energy and extract the partial
        // derivative  
        energy_ = getJKRCellCellEnergy<T>(
            dvec.norm(), R, Rcell, E0, Ecell, gamma, mode
        ); 
        grad(1, i) = energy_.getDerivative();
    }

    // For each orientation coordinate ... 
    for (int i = 0; i < 3; ++i)
    {
        // Get the gradient w.r.t the orientation of cell 1 ... 
        //
        // Calculate the distance vector using dual vectors 
        DualVector<T> r1_(3), r2_(3), n1_(3), n2_(3);
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                n1_(j) = DualNumber<T>(n1(j), 1.0); 
            else
                n1_(j) = DualNumber<T>(n1(j), 0.0);
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n2_(j) = DualNumber<T>(n2(j), 0.0);  
        }
        DualVector<T> p(3), q(3); 
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        DualVector<T> dvec = q - p; 

        // Calculate the JKR contact energy and extract the partial
        // derivative  
        DualNumber<T> energy_ = getJKRCellCellEnergy<T>(
            dvec.norm(), R, Rcell, E0, Ecell, gamma, mode
        );
        grad(2, i) = energy_.getDerivative();

        // Get the gradient w.r.t the position of cell 2 ... 
        //
        // Calculate the distance vector using dual vectors 
        for (int j = 0; j < 3; ++j)
        {
            if (j == i)
                n2_(j) = DualNumber<T>(n2(j), 1.0); 
            else
                n2_(j) = DualNumber<T>(n2(j), 0.0); 
            r1_(j) = DualNumber<T>(r1(j), 0.0);
            r2_(j) = DualNumber<T>(r2(j), 0.0);
            n1_(j) = DualNumber<T>(n1(j), 0.0); 
        }
        p << r1_(0) + s(0) * n1_(0),
             r1_(1) + s(0) * n1_(1),
             r1_(2) + s(0) * n1_(2);
        q << r2_(0) + s(1) * n2_(0),
             r2_(1) + s(1) * n2_(1), 
             r2_(2) + s(1) * n2_(2);  
        dvec = q - p;

        // Calculate the JKR contact energy and extract the partial
        // derivative  
        energy_ = getJKRCellCellEnergy<T>(
            dvec.norm(), R, Rcell, E0, Ecell, gamma, mode
        ); 
        grad(3, i) = energy_.getDerivative();
    }

    return std::make_pair(energy, grad); 
}

/**
 * Calculate the cell-surface repulsion force on the given cell. 
 *
 * @param rz z-coordinate of cell position, specified as a dual.  
 * @param n Cell orientation, specified in duals.
 * @param length Cell length. 
 * @param R Cell radius. 
 * @param E0 Elastic modulus of the EPS. 
 * @returns Dual vector specifying the generalized translational and orientational
 *          forces on the cell due to cell-surface repulsion. 
 */
template <typename T>
DualVector<T> getCellSurfaceRepulsionForce(const DualNumber<T> rz, 
                                           const Ref<const DualVector<T> >& n,
                                           const T length, const T R, const T E0)
{
    DualVector<T> b = DualVector<T>::Zero(6);

    // Calculate the cell-surface repulsion force ... 
    DualNumber<T> nz = n(2);
    DualNumber<T> rz_int1, rz_int2, nz_int1, nz_int2, nz_int3, nz_int4;

    if (nz.getValue() == 0)    // If the cell is horizontal ...  
    {
        DualNumber<T> delta = R - rz; 
        rz_int1 = delta * length;  
        rz_int2 = sqrt(delta) * length;  
        nz_int1 = delta * delta * length;
        nz_int2 = -nz * length * length * length / 12;
        nz_int3 = pow(delta, 1.5) * length; 
        nz_int4.setValue(0); 
        nz_int4.setDerivative(0);  
    }
    else
    {
        throw std::runtime_error("Not implemented yet"); 
    }
    DualNumber<T> rz_term1 = -2 * E0 * (1 - nz * nz) * rz_int1; 
    DualNumber<T> rz_term2 = -2 * E0 * sqrt(R) * nz * nz * rz_int2;  
    DualNumber<T> nz_term1 = -2 * E0 * nz * nz_int1; 
    DualNumber<T> nz_term2 = -2 * E0 * (1 - nz * nz) * nz_int2; 
    DualNumber<T> nz_term3 = (8. / 3.) * E0 * sqrt(R) * nz * nz_int3; 
    DualNumber<T> nz_term4 = -2 * E0 * sqrt(R) * nz * nz * nz_int4;
    b(2) = rz_term1 + rz_term2; 
    b(5) = nz_term1 + nz_term2 + nz_term3 + nz_term4;  
    
    return -b; 
}  

/**
 * Calculate the cell-surface repulsion energy on the given cell. 
 *
 * @param rz z-coordinate of cell position, specified as a dual.  
 * @param n Cell orientation, specified in duals.
 * @param length Cell length. 
 * @param R Cell radius. 
 * @param E0 Elastic modulus of the EPS. 
 * @returns Dual number specifying the cell-surface repulsion energy. 
 */
template <typename T>
DualNumber<T> getCellSurfaceRepulsionEnergy(const DualNumber<T> rz, 
                                            const Ref<const DualVector<T> >& n,
                                            const T length, const T R,
                                            const T E0)
{
    // Calculate the penetration depth integral 
    DualNumber<T> nz = n(2);
    DualNumber<T> delta = (R - rz > 0 ? R - rz : 0);
    DualNumber<T> depth; 
    if (nz.getValue() == 0)           // If the cell is horizontal ...
    {
        if (delta > 0)
        {
            depth = pow(R, -0.5) * (1 - nz * nz) * delta * delta * length;
            depth += (4. / 3.) * nz * nz * pow(delta, 1.5) * length;
        }
        else 
        {
            depth.setValue(0); 
            depth.setDerivative(0); 
        } 
    }
    else                              // If the cell is tilted ... 
    {
        // Get the cell-surface overlap at the two endpoints 
        const T half_l = length / 2; 
        DualNumber<T> delta_p = R - (rz - half_l * nz); 
        DualNumber<T> delta_q = R - (rz + half_l * nz);

        // If the cell fully contacts the surface ... 
        if (delta_p > 0 && delta_q > 0)
        {
            // If the cell's vertical tilt is small ... 
            const T nz_min = length * abs(nz.getValue()) / (2 * delta.getValue()); 
            if (nz_min < 0.05)
            {
                // Calculate the cell-surface contact area integrals using a small-nz
                // Taylor expansion
                //
                // Get the integral of \delta_i^2(s)
                const T l3 = length * length * length; 
                DualNumber<T> int1 = length * delta * delta + l3 * nz * nz / 12;

                // Get the integral of \delta_i^{3/2}(s)
                DualNumber<T> int2 = length * pow(delta, 1.5) + l3 * nz * nz / (32 * sqrt(delta));

                // Get the penetration depth 
                depth = pow(R, -0.5) * (1 - nz * nz) * int1 + (4. / 3.) * nz * nz * int2;
            }
            else    // Otherwise, use the full surface energy formula 
            {
                // Get the integral of \delta_i^2(s)
                DualNumber<T> ss = (R - rz) / nz;  
                DualNumber<T> int1(0, 0);
                if (ss > half_l)
                {
                    DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                    DualNumber<T> phi2 = R - rz - half_l * nz;
                    DualNumber<T> overlap1 = pow(phi1, 3);    // 2 + 1 
                    DualNumber<T> overlap2 = pow(phi2, 3);
                    int1 = (overlap1 - overlap2) / (3 * nz); 
                }
                else if (ss > -half_l)    // -half_l < s <= half_l
                {
                    DualNumber<T> phi = R - rz - (-half_l) * nz; 
                    DualNumber<T> overlap = pow(phi, 3);      // 2 + 1
                    int1 = overlap / (3 * nz); 
                }

                // Get the integral of \delta_i^{3/2}(s)
                DualNumber<T> int2(0, 0); 
                if (ss > half_l)
                {
                    DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                    DualNumber<T> phi2 = R - rz - half_l * nz;
                    DualNumber<T> overlap1 = pow(phi1, 2.5);    // 3/2 + 1
                    DualNumber<T> overlap2 = pow(phi2, 2.5);
                    int2 = (overlap1 - overlap2) / (2.5 * nz); 
                }
                else if (ss > -half_l)    // -half_l < s <= half_l
                {
                    DualNumber<T> phi = R - rz - (-half_l) * nz; 
                    DualNumber<T> overlap = pow(phi, 2.5);      // 3/2 + 1
                    int2 = overlap / (2.5 * nz); 
                }
                depth = pow(R, -0.5) * (1 - nz * nz) * int1 + (4. / 3.) * nz * nz * int2; 
            }
        }
        else        // Otherwise, use the full surface energy formula  
        {
            // Get the integral of \delta_i^2(s)
            DualNumber<T> ss = (R - rz) / nz;  
            DualNumber<T> int1(0, 0);
            if (ss > half_l)
            {
                DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                DualNumber<T> phi2 = R - rz - half_l * nz;
                DualNumber<T> overlap1 = pow(phi1, 3);    // 2 + 1 
                DualNumber<T> overlap2 = pow(phi2, 3);
                int1 = (overlap1 - overlap2) / (3 * nz); 
            }
            else if (ss > -half_l)    // -half_l < s <= half_l
            {
                DualNumber<T> phi = R - rz - (-half_l) * nz; 
                DualNumber<T> overlap = pow(phi, 3);      // 2 + 1
                int1 = overlap / (3 * nz); 
            }

            // Get the integral of \delta_i^{3/2}(s)
            DualNumber<T> int2(0, 0); 
            if (ss > half_l)
            {
                DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                DualNumber<T> phi2 = R - rz - half_l * nz;
                DualNumber<T> overlap1 = pow(phi1, 2.5);    // 3/2 + 1
                DualNumber<T> overlap2 = pow(phi2, 2.5);
                int2 = (overlap1 - overlap2) / (2.5 * nz); 
            }
            else if (ss > -half_l)    // -half_l < s <= half_l
            {
                DualNumber<T> phi = R - rz - (-half_l) * nz; 
                DualNumber<T> overlap = pow(phi, 2.5);      // 3/2 + 1
                int2 = overlap / (2.5 * nz); 
            }
            depth = pow(R, -0.5) * (1 - nz * nz) * int1 + (4. / 3.) * nz * nz * int2; 
        }
    }
    
    return E0 * sqrt(R) * depth; 
}  

/**
 * Calculate the cell-surface adhesion force on the given cell. 
 *
 * @param rz z-coordinate of cell position, specified as a dual.  
 * @param n Cell orientation, specified in duals.
 * @param length Cell length. 
 * @param R Cell radius. 
 * @param sigma0 Cell-surface adhesion energy density.
 * @returns Dual vector specifying the generalized translational and orientational
 *          forces on the cell due to cell-surface adhesion. 
 */
template <typename T>
DualVector<T> getCellSurfaceAdhesionForce(const DualNumber<T> rz,
                                          const Ref<const DualVector<T> >& n,
                                          const T length, const T R,
                                          const T sigma0)
{
    DualVector<T> b = DualVector<T>::Zero(6);

    // Calculate the cell-surface adhesion force 
    DualNumber<T> nz = n(2);
    DualNumber<T> rz_int1, nz_int1, nz_int2, nz_int3;  
    
    if (nz.getValue() == 0)    // If the cell is horizontal ...  
    {
        DualNumber<T> delta = R - rz; 
        rz_int1 = pow(delta, -0.5) * length;  
        nz_int1 = sqrt(delta) * length;
        nz_int2 = nz * pow(length, 3) / (24 * pow(delta, 1.5)); 
        nz_int3 = length; 
    }
    else 
    {
        throw std::runtime_error("Not implemented yet"); 
    }
    DualNumber<T> rz_term1 = 0.5 * sigma0 * sqrt(R) * (1 - nz * nz) * rz_int1;
    DualNumber<T> nz_term1 = 2 * sigma0 * sqrt(R) * nz * nz_int1; 
    DualNumber<T> nz_term2 = 0.5 * sigma0 * sqrt(R) * (1 - nz * nz) * nz_int2; 
    DualNumber<T> nz_term3 = -sigma0 * boost::math::constants::two_pi<T>() * R * nz * nz_int3;
    DualNumber<T> rz_term2, nz_term4; 
    if (nz.getValue() == 0)
    {
        rz_term2.setValue(0);
        rz_term2.setDerivative(0);   
        nz_term4.setValue(0);
        nz_term4.setDerivative(0);  
    }
    else 
    {
        throw std::runtime_error("Not implemented yet"); 
    } 
    b(2) = rz_term1 + rz_term2; 
    b(5) = nz_term1 + nz_term2 + nz_term3 + nz_term4;
    
    return -b; 
}

/**
 * Calculate the cell-surface adhesion energy on the given cell. 
 *
 * @param rz z-coordinate of cell position, specified as a dual.  
 * @param n Cell orientation, specified in duals.
 * @param length Cell length. 
 * @param R Cell radius. 
 * @param E0 Elastic modulus of the EPS. 
 * @returns Dual number specifying the cell-surface adhesion energy. 
 */
template <typename T>
DualNumber<T> getCellSurfaceAdhesionEnergy(const DualNumber<T> rz, 
                                           const Ref<const DualVector<T> >& n,
                                           const T length, const T R,
                                           const T sigma0)
{
    // Calculate the cell-surface contact area ...  
    DualNumber<T> nz = n(2);
    DualNumber<T> delta = (R - rz > 0 ? R - rz : 0);
    DualNumber<T> area; 
    if (nz.getValue() == 0)           // If the cell is horizontal ...
    {
        if (delta > 0)
        {
            area = sqrt(R) * (1 - nz * nz) * sqrt(delta) * length; 
            area += boost::math::constants::pi<T>() * R * nz * nz * length;
        }
        else 
        {
            area.setValue(0);
            area.setDerivative(0);  
        } 
    }
    else                              // If the cell is tilted ... 
    {
        // Get the cell-surface overlap at the two endpoints
        const T half_l = length / 2;  
        DualNumber<T> delta_p = R - (rz - half_l * nz);
        DualNumber<T> delta_q = R - (rz + half_l * nz);

        // If the cell fully contacts the surface ... 
        if (delta_p > 0 && delta_q > 0)
        {
            // If the cell's vertical tilt is small ...
            const T nz_min = length * abs(nz.getValue()) / (2 * delta.getValue());  
            if (nz_min < 0.05)
            {
                // Calculate the cell-surface contact area integrals using a small-nz
                // Taylor expansion
                //
                // Get the integral of \delta_i^{1/2}(s)
                const T l3 = length * length * length;
                DualNumber<T> int1 = length * sqrt(delta) - l3 * nz * nz / (96 * pow(delta, 1.5));

                // Get the integral of \Theta(\delta_i(s))
                DualNumber<T> int2 = length; 

                // Get the contact area  
                area = sqrt(R) * (1 - nz * nz) * int1;
                area += boost::math::constants::pi<T>() * R * nz * nz * int2;
            }
            else    // Otherwise, use the full surface energy formula
            {
                // Get the integral of \delta_i^{1/2}(s)
                DualNumber<T> ss = (R - rz) / nz;  
                DualNumber<T> int1(0, 0);
                if (ss > half_l)
                {
                    DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                    DualNumber<T> phi2 = R - rz - half_l * nz;
                    DualNumber<T> overlap1 = pow(phi1, 1.5);    // 1/2 + 1 
                    DualNumber<T> overlap2 = pow(phi2, 1.5);
                    int1 = (overlap1 - overlap2) / (1.5 * nz); 
                }
                else if (ss > -half_l)    // -half_l < s <= half_l
                {
                    DualNumber<T> phi = R - rz - (-half_l) * nz; 
                    DualNumber<T> overlap = pow(phi, 1.5);      // 1/2 + 1
                    int1 = overlap / (1.5 * nz); 
                }

                // Get the integral of \Theta(\delta_i(s))
                DualNumber<T> int2(0, 0);
                if (ss > half_l)
                    int2 = DualNumber<T>(2 * half_l, 0); 
                else if (ss > -half_l)    // -half_l < s <= half_l
                    int2 = ss + half_l; 

                // Get the contact area  
                area = sqrt(R) * (1 - nz * nz) * int1;
                area += boost::math::constants::pi<T>() * R * nz * nz * int2;
            } 
        }
        else        // Otherwise, use the full surface energy formula  
        {
            // Get the integral of \delta_i^{1/2}(s)
            DualNumber<T> ss = (R - rz) / nz;  
            DualNumber<T> int1(0, 0);
            if (ss > half_l)
            {
                DualNumber<T> phi1 = R - rz - (-half_l) * nz;
                DualNumber<T> phi2 = R - rz - half_l * nz;
                DualNumber<T> overlap1 = pow(phi1, 1.5);    // 1/2 + 1 
                DualNumber<T> overlap2 = pow(phi2, 1.5);
                int1 = (overlap1 - overlap2) / (1.5 * nz); 
            }
            else if (ss > -half_l)    // -half_l < s <= half_l
            {
                DualNumber<T> phi = R - rz - (-half_l) * nz; 
                DualNumber<T> overlap = pow(phi, 1.5);      // 1/2 + 1
                int1 = overlap / (1.5 * nz); 
            }

            // Get the integral of \Theta(\delta_i(s))
            DualNumber<T> int2(0, 0);
            if (ss > half_l)
                int2 = DualNumber<T>(2 * half_l, 0); 
            else if (ss > -half_l)    // -half_l < s <= half_l
                int2 = ss + half_l; 

            // Get the contact area  
            area = sqrt(R) * (1 - nz * nz) * int1;
            area += boost::math::constants::pi<T>() * R * nz * nz * int2;
        }
    }
    
    return -sigma0 * area; 
}  

/**
 * Calculate the configurational energy for the given configuration of cells, 
 * together with its gradient w.r.t the cell coordinates. 
 *
 * For simplicity, all interactions between neighboring cells are neglected
 * from this calculation by default.
 *
 * @param r Array of cell centers. 
 * @param n Array of cell orientations. 
 * @param length Cell length (assumed to be the same for all cells). 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density. 
 * @param ignore_neighbor_interactions If true, ignore interactions between
 *                                     neighboring cells in the energy 
 *                                     calculation.  
 * @returns The configurational energy and its gradient.  
 */
template <typename T>
std::pair<T, Matrix<T, Dynamic, 6> > getConfigurationalEnergy(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                                                              const Ref<const Matrix<T, Dynamic, 3> >& n, 
                                                              const T length,
                                                              const T R,
                                                              const T Rcell,
                                                              const T E0,
                                                              const T Ecell,
                                                              const T sigma0,
                                                              const T eta0,
                                                              const T eta1,
                                                              const T gamma,
                                                              const JKRCellBodyContactMode mode = JKRCellBodyContactMode::AllowRepulsiveContacts, 
                                                              const bool ignore_neighbor_interactions = true)
{
    // Calculate the configurational energy and its gradient w.r.t all
    // position/orientation coordinates
    K kernel;  
    const int n_cells = r.rows();
    const T half_l = length / 2;  
    T energy = 0; 
    Matrix<T, Dynamic, 6> grad = Matrix<T, Dynamic, 6>::Zero(n_cells, 6);

    // Run over all contacting pairs of cells ... 
    for (int i = 0; i < n_cells; ++i)
    {
        for (int j = i + 1; j < n_cells; ++j)
        {
            // Ignore interactions between neighboring cells, if desired 
            if (ignore_neighbor_interactions && i != 0)
                continue;

            // Determine if cells i and j are contacting 
            Segment_3 seg1 = generateSegment<double>(
                r.row(i).template cast<double>(),
                n.row(i).template cast<double>(),
                static_cast<double>(half_l)
            ); 
            Segment_3 seg2 = generateSegment<double>(
                r.row(j).template cast<double>(),
                n.row(j).template cast<double>(),
                static_cast<double>(half_l)
            ); 
            auto result = distBetweenCells<double>(
                seg1, seg2, 0,
                r.row(i).template cast<double>(),
                n.row(i).template cast<double>(),
                static_cast<double>(half_l), 1,
                r.row(j).template cast<double>(),  
                n.row(j).template cast<double>(),
                static_cast<double>(half_l), kernel
            );
            Matrix<T, 3, 1> dvec = std::get<0>(result).template cast<T>();
            T dist = dvec.norm(); 
            if (dist < 2 * R)
            {
                // If so, get the centerline coordinates 
                T s1 = static_cast<T>(std::get<1>(result));  
                T s2 = static_cast<T>(std::get<2>(result));
                Matrix<T, 2, 1> s; 
                s << s1, s2; 

                // Calculate the Hertzian or JKR cell-cell contact energy
                // and its gradient
                T energy_ij; 
                Matrix<T, 4, 3> grad_ij;
                if (gamma == 0)
                { 
                    auto result2 = getHertzianCellCellEnergyGradient<T>(
                        r.row(i), n.row(i), r.row(j), n.row(j), s, R, Rcell,
                        E0, Ecell
                    );
                    energy_ij = result2.first; 
                    grad_ij = result2.second;
                }
                else 
                {
                    auto result2 = getJKRCellCellEnergyGradient<T>(
                        r.row(i), n.row(i), r.row(j), n.row(j), s, R, Rcell, 
                        E0, Ecell, gamma, mode
                    );
                    energy_ij = result2.first; 
                    grad_ij = result2.second;
                } 
                energy += energy_ij;
                grad(i, Eigen::seqN(0, 3)) += grad_ij.row(0); 
                grad(j, Eigen::seqN(0, 3)) += grad_ij.row(1);
                grad(i, Eigen::seqN(3, 3)) += grad_ij.row(2); 
                grad(j, Eigen::seqN(3, 3)) += grad_ij.row(3); 
            } 
        }
    }

    // Calculate the cell-surface repulsion energy for each cell 
    for (int i = 0; i < n_cells; ++i)
    {
        // The energy depends only on the z-coordinates, so the partials 
        // w.r.t the x- and y-coordinates must be zero 
        //
        // First get the partial w.r.t the z-position
        DualNumber<T> rzi(r(i, 2), 1); 
        DualVector<T> ni(3);  
        ni << DualNumber<T>(n(i, 0), 0),
              DualNumber<T>(n(i, 1), 0),
              DualNumber<T>(n(i, 2), 0); 
        DualNumber<T> cell_surface_repulsion_energy = getCellSurfaceRepulsionEnergy<T>(
            rzi, ni, length, R, E0
        ); 
        energy += cell_surface_repulsion_energy.getValue(); 
        grad(i, 2) += cell_surface_repulsion_energy.getDerivative();

        // Second, get the partial w.r.t the z-derivative 
        rzi.setDerivative(0); 
        ni << DualNumber<T>(n(i, 0), 0),
              DualNumber<T>(n(i, 1), 0),
              DualNumber<T>(n(i, 2), 1);
        cell_surface_repulsion_energy = getCellSurfaceRepulsionEnergy<T>(
            rzi, ni, length, R, E0
        ); 
        grad(i, 5) += cell_surface_repulsion_energy.getDerivative();  
    }

    // Calculate the cell-surface adhesion energy for each cell 
    for (int i = 0; i < n_cells; ++i)
    {
        // The energy depends only on the z-coordinates, so the partials 
        // w.r.t the x- and y-coordinates must be zero 
        //
        // First get the partial w.r.t the z-position 
        DualNumber<T> rzi(r(i, 2), 1); 
        DualVector<T> ni(3);  
        ni << DualNumber<T>(n(i, 0), 0),
              DualNumber<T>(n(i, 1), 0),
              DualNumber<T>(n(i, 2), 0); 
        DualNumber<T> cell_surface_adhesion_energy = getCellSurfaceAdhesionEnergy<T>(
            rzi, ni, length, R, sigma0
        ); 
        energy += cell_surface_adhesion_energy.getValue(); 
        grad(i, 2) += cell_surface_adhesion_energy.getDerivative();

        // Second, get the partial w.r.t the z-derivative 
        rzi.setDerivative(0); 
        ni << DualNumber<T>(n(i, 0), 0),
              DualNumber<T>(n(i, 1), 0),
              DualNumber<T>(n(i, 2), 1);
        cell_surface_adhesion_energy = getCellSurfaceAdhesionEnergy<T>(
            rzi, ni, length, R, sigma0
        ); 
        grad(i, 5) += cell_surface_adhesion_energy.getDerivative();  
    }

    return std::make_pair(energy, grad); 
}

/**
 * Calculate the update direction corresponding to minimizing the configurational
 * energy, given the current configuration and the energy gradient.
 *
 * @param r Array of cell centers. 
 * @param n Array of cell orientations.
 * @param constraints Constraint matrix. This constraint matrix does not need 
 *                    to include the orientation vector norm constraints, which
 *                    are added below.  
 * @param length Cell length (assumed to be the same for all cells).
 * @param grad Pre-computed configurational energy gradient.  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density. 
 * @param force_ext External forces on the neighboring cells. 
 * @param force_ext_s Centerline coordinates on which the external forces are
 *                    applied on the neighboring cells.  
 * @returns The update direction corresponding to minimizing the configurational
 *          energy.  
 */
template <typename T>
Matrix<T, Dynamic, 6> getEnergyGradientDescentDirection(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                                                        const Ref<const Matrix<T, Dynamic, 3> >& n,
                                                        const Ref<const Matrix<T, Dynamic, Dynamic> >& constraints,  
                                                        const T length,
                                                        const Ref<const Matrix<T, Dynamic, 6> >& grad,
                                                        const T R,
                                                        const T Rcell,
                                                        const T E0,
                                                        const T Ecell,
                                                        const T sigma0,
                                                        const T eta0,
                                                        const T eta1,
                                                        const T gamma,
                                                        const Ref<const Matrix<T, Dynamic, 3> >& force_ext,
                                                        const Ref<const Matrix<T, Dynamic, 1> >& force_ext_s)
{
    // Define one large coordinate vector 
    const int n_cells = r.rows();  
    Matrix<T, Dynamic, 1> q(6 * n_cells); 
    for (int i = 0; i < n_cells; ++i)
    {
        q(Eigen::seqN(6 * i, 3)) = r.row(i); 
        q(Eigen::seqN(6 * i + 3, 3)) = n.row(i); 
    } 
    
    // Get the dissipation matrix for each cell
    Matrix<T, Dynamic, Dynamic> diss_total = Matrix<T, Dynamic, Dynamic>::Zero(6 * n_cells, 6 * n_cells); 
    for (int i = 0; i < n_cells; ++i)
    {
        Matrix<T, 3, 1> ri = r.row(i); 
        Matrix<T, 3, 1> ni = n.row(i); 
        if (n(i, 2) < 0)
            ni *= -1;  
        Matrix<T, 6, 6> diss = getDissipationMatrix<T>(ri, ni, length, R, eta0, eta1);
        if (n(i, 2) < 0)
        {
            Matrix<T, 6, 6> trans = Matrix<T, 6, 6>::Identity(); 
            trans(Eigen::seqN(3, 3), Eigen::seqN(3, 3)) *= -1; 
            diss = trans * diss * trans; 
        }
        diss_total(Eigen::seqN(6 * i, 6), Eigen::seqN(6 * i, 6)) = diss; 
    }

    // Formulate the full constraint matrix ... 
    //
    // First, introduce the orientation vector norm constraints
    const int n_constraints = n_cells + constraints.rows();  
    Matrix<T, Dynamic, Dynamic> constraints_all = Matrix<T, Dynamic, Dynamic>::Zero(n_constraints, 6 * n_cells); 
    for (int i = 0; i < n_cells; ++i)
    {
        constraints_all(i, 6 * i + 3) = n(i, 0); 
        constraints_all(i, 6 * i + 4) = n(i, 1); 
        constraints_all(i, 6 * i + 5) = n(i, 2); 
    }

    // Then add in the remaining constraints
    if (constraints.rows() > 0)
        constraints_all(Eigen::seqN(n_cells, constraints.rows()), Eigen::all) = constraints;

    // Extract the linearly independent rows in the constraint matrix ... 
    //
    // Get the column-pivoted QR decomposition of the constraint matrix, get
    // the rank, then extract the first (rank) rows in the permuted constraint
    // matrix 
    auto constraints_qr = constraints_all.transpose().colPivHouseholderQr();
    const int rank = constraints_qr.rank(); 
    auto ind_idx = constraints_qr.colsPermutation().indices(); 
    Matrix<T, Dynamic, Dynamic> constraints_ind(rank, constraints_all.cols()); 
    for (int i = 0; i < rank; ++i)
        constraints_ind.row(i) = constraints_all.row(ind_idx(i)); 

    // Re-organize the gradient into one large vector 
    Matrix<T, Dynamic, 1> grad_(6 * n_cells);
    for (int i = 0; i < n_cells; ++i)
        grad_(Eigen::seqN(6 * i, 6)) = grad.row(i);

    // Re-organize the external forces into one large vector
    Matrix<T, Dynamic, 1> force_ext_(6 * n_cells);
    for (int i = 0; i < n_cells; ++i)
    {
        force_ext_(Eigen::seqN(6 * i, 3)) = force_ext.row(i);
        force_ext_(Eigen::seqN(6 * i + 3, 3)) = force_ext_s(i) * force_ext.row(i);  
    } 

    // First solve the linear equation:
    //
    // diss_total * w = -grad_ + force_ext_
    //
    // which yields the unconstrained velocity vector 
    auto decomp1 = diss_total.colPivHouseholderQr();
    Matrix<T, Dynamic, 1> w = decomp1.solve(-grad_ + force_ext_); 

    // Then solve the linear (matrix) equation: 
    //
    // diss_total * X = constraints_ind.transpose()
    Matrix<T, Dynamic, Dynamic> X = decomp1.solve(constraints_ind.transpose());

    // Then solve the linear equation:
    //
    // constraints_ind * X * u = constraints_ind * w
    auto decomp2 = (constraints_ind * X).colPivHouseholderQr(); 
    Matrix<T, Dynamic, 1> u = decomp2.solve(constraints_ind * w);

    // Calculate the constrained velocity vector and re-organize its entries 
    Matrix<T, Dynamic, 1> v = w - X * u;
    Matrix<T, Dynamic, 6> v2(n_cells, 6);
    for (int i = 0; i < n_cells; ++i)
        v2.row(i) = v(Eigen::seqN(6 * i, 6));  

    return v2;  
}

/**
 * Calculate the Dormand-Prince update for the given stepsize.
 *
 * @param r Array of cell centers. 
 * @param n Array of cell orientations.
 * @param constraints Constraint matrix. This constraint matrix does not need 
 *                    to include the orientation vector norm constraints, which
 *                    are added below.  
 * @param length Cell length (assumed to be the same for all cells).
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density.
 * @param force_ext External forces on the neighboring cells. 
 * @param force_ext_s Centerline coordinates on which the external forces are
 *                    applied on the neighboring cells.  
 * @param stepsize Trial stepsize. 
 * @param ignore_neighbor_interactions If true, ignore interactions between
 *                                     neighboring cells in the energy 
 *                                     calculation.  
 * @returns The Dormand-Prince update and the associated fifth-order error. 
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 6>, Matrix<T, Dynamic, 6> > getDormandPrinceUpdate(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                                                                                const Ref<const Matrix<T, Dynamic, 3> >& n, 
                                                                                const Ref<const Matrix<T, Dynamic, Dynamic> >& constraints, 
                                                                                const T length,
                                                                                const T R,
                                                                                const T Rcell,
                                                                                const T E0,
                                                                                const T Ecell,
                                                                                const T sigma0,
                                                                                const T eta0,
                                                                                const T eta1,
                                                                                const T gamma,
                                                                                const Ref<const Matrix<T, Dynamic, 3> >& force_ext,
                                                                                const Ref<const Matrix<T, Dynamic, 1> >& force_ext_s,  
                                                                                const T stepsize,
                                                                                const JKRCellBodyContactMode mode = JKRCellBodyContactMode::AllowRepulsiveContacts, 
                                                                                const bool ignore_neighbor_interactions = true)
{
    // Define the Dormand-Prince tableau
    Matrix<T, Dynamic, Dynamic> A(7, 7);  
    A <<              0,               0,              0,            0,               0,         0, 0,
                1. / 5.,               0,              0,            0,               0,         0, 0,
               3. / 40.,        9. / 40.,              0,            0,               0,         0, 0,
              44. / 45.,      -56. / 15.,       32. / 9.,            0,               0,         0, 0,
         19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729.,               0,         0, 0,
          9017. / 3168.,     -355. / 33., 46732. / 5247.,   49. / 176., -5103. / 18656.,         0, 0,
             35. / 384.,               0,   500. / 1113.,  125. / 192.,  -2187. / 6784., 11. / 84., 0;
    Matrix<T, Dynamic, 1> b(7), bs(7), c(7); 
    b << 35. / 384., 0, 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0;
    bs <<    5179. / 57600.,
                          0,
             7571. / 16695.,
                393. / 640.,
          -92097. / 339200.,
               187. / 2100.,
                   1. / 40.;
    c << 0, 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1, 1; 

    // Calculate each step of the Runge-Kutta integration ...
    //
    // Pass in a transformed set of coordinates in which all z-orientations
    // are positive 
    const int n_cells = r.rows(); 
    std::vector<Matrix<T, Dynamic, 6> > steps;
    Matrix<T, Dynamic, 3> n_(n); 
    for (int j = 0; j < n_cells; ++j)
    {
        if (n(j, 2) < 0)
            n_.row(j) *= -1; 
    }
    auto result = getConfigurationalEnergy<T>(
        r, n_, length, R, Rcell, E0, Ecell, sigma0, eta0, eta1, gamma, mode,
        ignore_neighbor_interactions
    );

    // Get the energy gradient and transform back the z-orientations so that
    // they match the input coordinates  
    Matrix<T, Dynamic, 6> grad = result.second;
    for (int j = 0; j < n_cells; ++j)
    {
        if (n(j, 2) < 0)
            grad(j, Eigen::seqN(3, 3)) *= -1; 
    }

    // Get the velocities
    Matrix<T, Dynamic, 6> v = getEnergyGradientDescentDirection<T>(
        r, n, constraints, length, grad, R, Rcell, E0, Ecell, sigma0, eta0,
        eta1, gamma, force_ext, force_ext_s
    );
    steps.push_back(v);   
    for (int i = 1; i < A.rows(); ++i)
    {
        // Calculate the intermediate cell coordinates for the i-th step
        Matrix<T, Dynamic, 3> r_delta = Matrix<T, Dynamic, 3>::Zero(n_cells, 3);
        Matrix<T, Dynamic, 3> n_delta = Matrix<T, Dynamic, 3>::Zero(n_cells, 3);
        for (int j = 0; j < i; ++j)
        {
            Matrix<T, Dynamic, 3> r_step_j = steps[j](Eigen::all, Eigen::seqN(0, 3));
            Matrix<T, Dynamic, 3> n_step_j = steps[j](Eigen::all, Eigen::seqN(3, 3));  
            r_delta += A(i, j) * r_step_j;
            n_delta += A(i, j) * n_step_j; 
        }
        Matrix<T, Dynamic, 3> ri = r + stepsize * r_delta;
        Matrix<T, Dynamic, 3> ni = n + stepsize * n_delta;

        // Re-normalize the cell orientations 
        for (int j = 0; j < n_cells; ++j)
        {
            T norm = ni.row(j).norm();
            ni.row(j) /= norm; 
        }

        // Calculate the corresponding energy gradient
        Matrix<T, Dynamic, 3> ni_(ni); 
        for (int j = 0; j < n_cells; ++j)    // Make all z-orientations positive for energy calculations
        {
            if (ni(j, 2) < 0)
                ni_.row(j) *= -1; 
        }  
        result = getConfigurationalEnergy<T>(
            ri, ni_, length, R, Rcell, E0, Ecell, sigma0, eta0, eta1, gamma,
            mode, ignore_neighbor_interactions
        ); 
        Matrix<T, Dynamic, 6> grad_i = result.second;
        for (int j = 0; j < n_cells; ++j)    // Switch z-orientations back to original signs 
        {
            if (ni(j, 2) < 0)
                grad_i(j, Eigen::seqN(3, 3)) *= -1; 
        }

        // Calculate the corresponding velocity vector 
        Matrix<T, Dynamic, 6> vi = getEnergyGradientDescentDirection<T>(
            ri, ni, constraints, length, grad_i, R, Rcell, E0, Ecell, sigma0,
            eta0, eta1, gamma, force_ext, force_ext_s
        );
        steps.push_back(vi);  
    }

    // Combine the steps together to get the final update 
    Matrix<T, Dynamic, 3> vr_final = Matrix<T, Dynamic, 3>::Zero(n_cells, 3);
    Matrix<T, Dynamic, 3> vn_final = Matrix<T, Dynamic, 3>::Zero(n_cells, 3); 
    for (int i = 0; i < b.size(); ++i)
    {
        Matrix<T, Dynamic, 6> step_i = b(i) * steps[i]; 
        vr_final += step_i(Eigen::all, Eigen::seqN(0, 3)); 
        vn_final += step_i(Eigen::all, Eigen::seqN(3, 3)); 
    }
    Matrix<T, Dynamic, 6> update_final = Matrix<T, Dynamic, 6>::Zero(n_cells, 6); 
    update_final(Eigen::all, Eigen::seqN(0, 3)) = stepsize * vr_final;
    update_final(Eigen::all, Eigen::seqN(3, 3)) = stepsize * vn_final; 

    // Get the Runge-Kutta error estimate 
    Matrix<T, Dynamic, 3> vr_embed = Matrix<T, Dynamic, 3>::Zero(n_cells, 3);
    Matrix<T, Dynamic, 3> vn_embed = Matrix<T, Dynamic, 3>::Zero(n_cells, 3); 
    for (int i = 0; i < bs.size(); ++i)
    {
        Matrix<T, Dynamic, 6> step_i = bs(i) * steps[i]; 
        vr_embed += step_i(Eigen::all, Eigen::seqN(0, 3)); 
        vn_embed += step_i(Eigen::all, Eigen::seqN(3, 3)); 
    }
    Matrix<T, Dynamic, 6> update_embed = Matrix<T, Dynamic, 6>::Zero(n_cells, 6);  
    update_embed(Eigen::all, Eigen::seqN(0, 3)) = stepsize * vr_embed;
    update_embed(Eigen::all, Eigen::seqN(3, 3)) = stepsize * vn_embed; 
    
    return std::make_pair(update_final, update_final - update_embed); 
}

/**
 * Apply a protocol for adaptive stepsize control, using the fifth-order 
 * Dormand-Prince method.  
 *
 * @param r Array of cell centers. 
 * @param n Array of cell orientations.
 * @param constraints Constraint matrix. This constraint matrix does not need 
 *                    to include the orientation vector norm constraints, which
 *                    are added below.  
 * @param length Cell length (assumed to be the same for all cells).
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density.
 * @param force_ext External forces on the neighboring cells. 
 * @param force_ext_s Centerline coordinates on which the external forces are
 *                    applied on the neighboring cells.  
 * @param curr_stepsize Initial trial stepsize. 
 * @param r_tol Absolute tolerance for position coordinates. 
 * @param n_tol Absolute tolerance for orientation coordinates. 
 * @param min_stepsize Minimum Dormand-Prince stepsize. 
 * @param max_stepsize Maximum Dormand-Prince stepsize. 
 * @param max_tries Maximum number of stepsize control iterations.
 * @param ignore_neighbor_interactions If true, ignore interactions between
 *                                     neighboring cells in the energy 
 *                                     calculation.  
 * @returns The Dormand-Prince update, the stepsize corresponding to the update,
 *          and the initial trial stepsize to try for the next iteration. 
 */
template <typename T>
std::tuple<Matrix<T, Dynamic, 6>, T, T> getDormandPrinceUpdateWithAdaptedStepsize(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                                                                                  const Ref<const Matrix<T, Dynamic, 3> >& n, 
                                                                                  const Ref<const Matrix<T, Dynamic, Dynamic> >& constraints, 
                                                                                  const T length,
                                                                                  const T R,
                                                                                  const T Rcell,
                                                                                  const T E0,
                                                                                  const T Ecell,
                                                                                  const T sigma0,
                                                                                  const T eta0,
                                                                                  const T eta1,
                                                                                  const T gamma,
                                                                                  const Ref<const Matrix<T, Dynamic, 3> >& force_ext,
                                                                                  const Ref<const Matrix<T, Dynamic, 1> >& force_ext_s,  
                                                                                  const T curr_stepsize,
                                                                                  const T r_tol, 
                                                                                  const T n_tol,
                                                                                  const T min_stepsize, 
                                                                                  const T max_stepsize, 
                                                                                  const int max_tries,
                                                                                  const JKRCellBodyContactMode mode = JKRCellBodyContactMode::AllowRepulsiveContacts, 
                                                                                  const bool ignore_neighbor_interactions = true)
{
    // Generate an initial Dormand-Prince update 
    auto result = getDormandPrinceUpdate<T>(
        r, n, constraints, length, R, Rcell, E0, Ecell, sigma0, eta0, eta1, 
        gamma, force_ext, force_ext_s, curr_stepsize, mode,
        ignore_neighbor_interactions
    );
    Matrix<T, Dynamic, 6> update_final = result.first; 
    Matrix<T, Dynamic, 6> update_error = result.second;
    Matrix<T, Dynamic, 6> update_embed = update_final - update_error; 

    // Get the updated orientation vectors 
    Matrix<T, Dynamic, 3> n_updated_final = n + update_final(Eigen::all, Eigen::seqN(3, 3)); 
    Matrix<T, Dynamic, 3> n_updated_embed = n + update_embed(Eigen::all, Eigen::seqN(3, 3));

    // Normalize the updated orientation vectors
    Matrix<T, Dynamic, 1> n_updated_norms_final = n_updated_final.rowwise().norm(); 
    Matrix<T, Dynamic, 1> n_updated_norms_embed = n_updated_embed.rowwise().norm();
    const int n_cells = n.rows(); 
    for (int i = 0; i < n_cells; ++i)
    {
        for (int j = 0; j < 3; ++j) 
        {
            n_updated_final(i, j) /= n_updated_norms_final(i);  
            n_updated_embed(i, j) /= n_updated_norms_embed(i);
        }
    }

    // Update the orientation vector increment
    update_final(Eigen::all, Eigen::seqN(3, 3)) = n_updated_final - n; 
    update_error(Eigen::all, Eigen::seqN(3, 3)) = n_updated_final - n_updated_embed;  

    // Define an error scale for each coordinate 
    Matrix<T, Dynamic, 6> scale = Matrix<T, Dynamic, 6>::Zero(n_cells, 6); 
    for (int i = 0; i < n_cells; ++i)
    {
        for (int j = 0; j < 3; ++j)
            scale(i, j) = r_tol;
        for (int j = 3; j < 6; ++j)
            scale(i, j) = n_tol;  
    } 

    // Calculate the maximum normalized error 
    //
    // Here, we use the maximum (L_infinity) instead of the root-mean-square
    // (L2) due to possible sparsity in the normalized error matrix in a
    // highly symmetry-constrained system
    T max_error = 0; 
    for (int i = 0; i < n_cells; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            T scaled_error = abs(update_error(i, j) / scale(i, j));
            max_error = max(max_error, scaled_error); 
        }
    }

    // While the maximum error is greater than 1 ... 
    int n_tries = 0;
    T stepsize_factor = 1.0;
    while (n_tries < max_tries && max_error > 1)
    {
        // Reduce the stepsize
        stepsize_factor *= max(0.9 * pow(max_error, -0.2), 0.2); 
        T next_stepsize = stepsize_factor * curr_stepsize;
        
        // Check if the stepsize is less than the prescribed minimum
        if (next_stepsize < min_stepsize)
        {
            stepsize_factor = min_stepsize / curr_stepsize; 
            next_stepsize = min_stepsize;
        } 

        // Re-try the Dormand-Prince step
        result = getDormandPrinceUpdate<T>(
            r, n, constraints, length, R, Rcell, E0, Ecell, sigma0, eta0, eta1, 
            gamma, force_ext, force_ext_s, next_stepsize, mode,
            ignore_neighbor_interactions
        );
        update_final = result.first; 
        update_error = result.second;
        update_embed = update_final - update_error; 

        // Get the updated orientation vectors 
        n_updated_final = n + update_final(Eigen::all, Eigen::seqN(3, 3)); 
        n_updated_embed = n + update_embed(Eigen::all, Eigen::seqN(3, 3));

        // Normalize the updated orientation vectors
        n_updated_norms_final = n_updated_final.rowwise().norm(); 
        n_updated_norms_embed = n_updated_embed.rowwise().norm();
        for (int i = 0; i < n_cells; ++i)
        { 
            for (int j = 0; j < 3; ++j)
            {
                n_updated_final(i, j) /= n_updated_norms_final(i); 
                n_updated_embed(i, j) /= n_updated_norms_embed(i);
            }
        }

        // Update the orientation vector increment
        update_final(Eigen::all, Eigen::seqN(3, 3)) = n_updated_final - n; 
        update_error(Eigen::all, Eigen::seqN(3, 3)) = n_updated_final - n_updated_embed;  

        // Calculate the maximum normalized error 
        max_error = 0; 
        for (int i = 0; i < n_cells; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                T scaled_error = abs(update_error(i, j) / scale(i, j));
                max_error = max(max_error, scaled_error);  
            }
        }

        // Increment the number of tries 
        n_tries++;
        
        // If we have already reached the minimum stepsize, break 
        if (next_stepsize == min_stepsize)
            break;  
    }

    // Get the updated stepsize 
    T updated_stepsize = stepsize_factor * curr_stepsize;  
    
    // Test if the initial stepsize for the next iteration can be increased
    T s = clamp(0.9 * pow(max_error, -0.2), 0.2, 10.0);
    T new_stepsize; 
    if (n_tries > 0)    // If there were rejected steps ... 
        new_stepsize = (s > 1 ? updated_stepsize : s * updated_stepsize);
    else                // Otherwise ... 
        new_stepsize = s * updated_stepsize; 

    // Clamp the stepsize between the minimum and maximum 
    new_stepsize = clamp(new_stepsize, min_stepsize, max_stepsize); 

    return std::make_tuple(update_final, updated_stepsize, new_stepsize); 
}

/**
 * Calculate the stability of the z-orientation of a horizontal, surface-
 * attached cell with the given configuration of neighbors.  
 *
 * @param neighbors_rxy Array of neighbor cell positions in x and y. All cells
 *                      are assumed to have the same z-position. 
 * @param neighbors_nxy Array of neighbor cell orientations in x and y.
 *                      All cells are assumed to be horizontal (nz = 0).
 * @param length Cell length (assumed to be the same for all cells). 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density.  
 * @returns The stability eigenvalue corresponding to the given configuration. 
 */
template <typename T>
T getOrientationalStability(const Ref<const Matrix<T, Dynamic, 2> >& neighbors_rxy, 
                            const Ref<const Matrix<T, Dynamic, 2> >& neighbors_nxy, 
                            const T length, const T R, const T Rcell,
                            const T E0, const T Ecell, const T sigma0,
                            const T eta0, const T eta1, const T gamma)
{
    K kernel;

    // Define the coordinates for all cells in the configuration 
    //
    // The central cell is assumed to be positioned at (0, 0, rz) with an 
    // orientation of (1, 0, 0)
    const int n_cells = neighbors_rxy.rows() + 1;
    const T rz = R - pow(sigma0 * sqrt(R) / (4 * E0), 2. / 3.);
    Matrix<T, Dynamic, 3> r(n_cells, 3), n(n_cells, 3);
    r(0, 0) = 0; 
    r(0, 1) = 0; 
    r(0, 2) = rz; 
    n(0, 0) = 1; 
    n(0, 1) = 0; 
    n(0, 2) = 0; 
    for (int i = 1; i < n_cells; ++i)
    {
        r(i, Eigen::seqN(0, 2)) = neighbors_rxy.row(i - 1);
        r(i, 2) = rz;  
        n(i, Eigen::seqN(0, 2)) = neighbors_nxy.row(i - 1);
        n(i, 2) = 0;  
    } 

    // Define a vector of dual coordinates for the central cell
    //
    // We are only interested in the derivative w.r.t the z-orientation
    DualVector<T> q(6);
    q << DualNumber<T>(r(0, 0), 0), 
         DualNumber<T>(r(0, 1), 0), 
         DualNumber<T>(r(0, 2), 0), 
         DualNumber<T>(n(0, 0), 0), 
         DualNumber<T>(n(0, 1), 0), 
         DualNumber<T>(n(0, 2), 1);
    DualVector<T> r0 = q(Eigen::seqN(0, 3)); 
    DualVector<T> n0 = q(Eigen::seqN(3, 3));

    // First calculate the distance vectors from each neighbor to the central 
    // cell, and the corresponding centerline coordinates
    const T half_l = length / 2;  
    Matrix<DualNumber<T>, Dynamic, 5> distances(n_cells - 1, 5); 
    for (int i = 1; i < n_cells; ++i)
    {
        Segment_3 seg1 = generateSegment<double>(
            r.row(0).template cast<double>(), n.row(0).template cast<double>(),
            static_cast<double>(half_l)
        ); 
        Segment_3 seg2 = generateSegment<double>(
            r.row(i).template cast<double>(), n.row(i).template cast<double>(),
            static_cast<double>(half_l)
        ); 
        auto result = distBetweenCells<double>(
            seg1, seg2, 0,
            r.row(0).template cast<double>(),
            n.row(0).template cast<double>(),
            static_cast<double>(half_l), 1,
            r.row(i).template cast<double>(),  
            n.row(i).template cast<double>(),
            static_cast<double>(half_l), kernel
        );

        // The centerline coordinates do not vary with the perturbation
        // coordinate u
        T s1 = static_cast<T>(std::get<1>(result));  
        T s2 = static_cast<T>(std::get<2>(result));
        distances(i - 1, 0) = DualNumber<T>(s1, 0); 
        distances(i - 1, 1) = DualNumber<T>(s2, 0);

        // The distance vectors do vary with the perturbation coordinate u 
        Matrix<T, 3, 1> dvec = std::get<0>(result).template cast<T>();
        distances(i - 1, 2) = DualNumber<T>(dvec(0), 0);
        distances(i - 1, 3) = DualNumber<T>(dvec(1), 0); 
        distances(i - 1, 4) = DualNumber<T>(dvec(2), -s1); 
    }

    // Calculate the corresponding Hertzian forces on the central cell
    DualVector<T> cell_cell_forces; 
    if (gamma == 0)
    { 
        cell_cell_forces = getHertzianCellCellForce<T>(
            distances, R, Rcell, E0, Ecell
        );
    }
    else 
    {
        cell_cell_forces = getJKRCellCellForce<T>(
            distances, R, Rcell, E0, gamma
        ); 
    }
    DualVector<T> forces = cell_cell_forces(Eigen::seqN(3, 3)); 

    // Calculate the cell-surface repulsion force on the central cell  
    DualVector<T> cell_surface_repulsion_forces = getCellSurfaceRepulsionForce<T>(
        r0(2), n0, length, R, E0
    );
    forces += cell_surface_repulsion_forces(Eigen::seqN(3, 3)); 

    // Calculate the cell-surface adhesion force on the central cell 
    DualVector<T> cell_surface_adhesion_forces = getCellSurfaceAdhesionForce<T>(
        r0(2), n0, length, R, sigma0
    );
    forces += cell_surface_adhesion_forces(Eigen::seqN(3, 3));

    // Calculate the derivative of the orientation vector w.r.t the
    // perturbation coordinate u
    DualNumber<T> u(0, 1);
    DualVector<T> x(3), z(3);    // Unit vectors in the x- and z-directions  
    x << DualNumber<T>(1, 0),
         DualNumber<T>(0, 0), 
         DualNumber<T>(0, 0); 
    z << DualNumber<T>(0, 0), 
         DualNumber<T>(0, 0), 
         DualNumber<T>(1, 0);
    DualVector<T> dndu = (-u / sqrt(1 - u * u)) * x + z;

    // Calculate the left-hand dissipation matrix
    Matrix<T, 3, 1> r0_, n0_; 
    r0_ << r0(0).getValue(), r0(1).getValue(), r0(2).getValue(); 
    n0_ << n0(0).getValue(), n0(1).getValue(), n0(2).getValue(); 
    Matrix<T, 6, 6> diss = getDissipationMatrix<T>(r0_, n0_, length, R, eta0, eta1);

    // Calculate the stability eigenvalue
    Matrix<T, 3, 1> z_; 
    z_ << 0, 0, 1; 
    T denom = z_.transpose() * diss(Eigen::seqN(3, 3), Eigen::seqN(3, 3)) * z_;
    return dndu.dot(forces).getDerivative() / denom;  
}

/**
 * Calculate the stability of the z-position and z-orientation of a horizontal,
 * surface-attached cell with the given configuration of neighbors. 
 *
 * @param neighbors_rxy Array of neighbor cell positions in x and y. All cells
 *                      are assumed to have the same z-position. 
 * @param neighbors_nxy Array of neighbor cell orientations in x and y.
 *                      All cells are assumed to be horizontal (nz = 0).
 * @param length Cell length (assumed to be the same for all cells). 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS). 
 * @param E0 Elastic modulus of the EPS. 
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient.
 * @param gamma Cell-cell adhesion energy density.  
 * @returns The stability eigenvalues corresponding to the given configuration. 
 */
template <typename T>
Matrix<T, 2, 1> get2DStability(const Ref<const Matrix<T, Dynamic, 2> >& neighbors_rxy, 
                               const Ref<const Matrix<T, Dynamic, 2> >& neighbors_nxy,
                               const T length, const T R, const T Rcell,
                               const T E0, const T Ecell, const T sigma0,
                               const T eta0, const T eta1, const T gamma)
{
    K kernel;

    // Define the coordinates for all cells in the configuration 
    //
    // The central cell is assumed to be positioned at (0, 0, rz) with an 
    // orientation of (1, 0, 0)
    const int n_cells = neighbors_rxy.rows() + 1;
    const T rz = R - pow(sigma0 * sqrt(R) / (4 * E0), 2. / 3.);
    Matrix<T, Dynamic, 3> r(n_cells, 3), n(n_cells, 3);
    r(0, 0) = 0; 
    r(0, 1) = 0; 
    r(0, 2) = rz; 
    n(0, 0) = 1; 
    n(0, 1) = 0; 
    n(0, 2) = 0; 
    for (int i = 1; i < n_cells; ++i)
    {
        r(i, Eigen::seqN(0, 2)) = neighbors_rxy.row(i - 1);
        r(i, 2) = rz;  
        n(i, Eigen::seqN(0, 2)) = neighbors_nxy.row(i - 1);
        n(i, 2) = 0;  
    }

    // First calculate the distance vectors from each neighbor to the central 
    // cell, and the corresponding centerline coordinates
    //
    // We are only interested in the derivatives w.r.t the z-position and 
    // z-orientation
    const T half_l = length / 2;  
    Matrix<DualNumber<T>, Dynamic, 5> distances_du(n_cells - 1, 5), 
                                      distances_dw(n_cells - 1, 5);
    for (int i = 1; i < n_cells; ++i)
    {
        Segment_3 seg1 = generateSegment<double>(
            r.row(0).template cast<double>(), n.row(0).template cast<double>(),
            static_cast<double>(half_l)
        ); 
        Segment_3 seg2 = generateSegment<double>(
            r.row(i).template cast<double>(), n.row(i).template cast<double>(),
            static_cast<double>(half_l)
        ); 
        auto result = distBetweenCells<double>(
            seg1, seg2, 0,
            r.row(0).template cast<double>(),
            n.row(0).template cast<double>(),
            static_cast<double>(half_l), 1,
            r.row(i).template cast<double>(),  
            n.row(i).template cast<double>(),
            static_cast<double>(half_l), kernel
        );

        // The centerline coordinates do not vary with either perturbation 
        // coordinate (u for nz, w for rz)
        T s1 = static_cast<T>(std::get<1>(result));  
        T s2 = static_cast<T>(std::get<2>(result));
        distances_dw(i - 1, 0) = DualNumber<T>(s1, 0); 
        distances_dw(i - 1, 1) = DualNumber<T>(s2, 0);
        distances_du(i - 1, 0) = DualNumber<T>(s1, 0); 
        distances_du(i - 1, 1) = DualNumber<T>(s2, 0);

        // The distance vectors do vary with both perturbation coordinates 
        //
        // These distance vectors are from the central cell to the neighbors
        Matrix<T, 3, 1> dvec = std::get<0>(result); 
        DualVector<T> dvec_dw(3), dvec_du(3);
        dvec_dw << DualNumber<T>(dvec(0), 0),    // Contact with cell to the left
                   DualNumber<T>(dvec(1), 0), 
                   DualNumber<T>(dvec(2), -1); 
        dvec_du << DualNumber<T>(dvec(0), 0),
                   DualNumber<T>(dvec(1), 0), 
                   DualNumber<T>(dvec(2), -s1);
        distances_dw(i - 1, Eigen::seqN(2, 3)) = dvec_dw; 
        distances_du(i - 1, Eigen::seqN(2, 3)) = dvec_du;  
    } 
    
    // Calculate the corresponding Hertzian/JKR forces on the central cell
    DualVector<T> forces_dw, forces_du;
    if (gamma == 0)
    {
        forces_dw = getHertzianCellCellForce<T>(distances_dw, R, Rcell, E0, Ecell);
        forces_du = getHertzianCellCellForce<T>(distances_du, R, Rcell, E0, Ecell);
    }
    else 
    {
        forces_dw = getJKRCellCellForce<T>(distances_dw, R, Rcell, E0, gamma);
        forces_du = getJKRCellCellForce<T>(distances_du, R, Rcell, E0, gamma);
    } 

    // Calculate the cell-surface repulsion force on the central cell
    DualNumber<T> rz1(rz, 1), rz0(rz, 0);  
    DualVector<T> n0_z1(3), n0_z0(3);
    n0_z1 << DualNumber<T>(1, 0), DualNumber<T>(0, 0), DualNumber<T>(0, 1);
    n0_z0 << DualNumber<T>(1, 0), DualNumber<T>(0, 0), DualNumber<T>(0, 0);
    DualVector<T> cell_surface_repulsion_forces = getCellSurfaceRepulsionForce<T>(
        rz1, n0_z0, length, R, E0
    );
    forces_dw += cell_surface_repulsion_forces;
    cell_surface_repulsion_forces = getCellSurfaceRepulsionForce<T>(
        rz0, n0_z1, length, R, E0
    );
    forces_du += cell_surface_repulsion_forces;  

    // Calculate the cell-surface adhesion force on the central cell 
    DualVector<T> cell_surface_adhesion_forces = getCellSurfaceAdhesionForce<T>(
        rz1, n0_z0, length, R, sigma0
    );
    forces_dw += cell_surface_adhesion_forces;
    cell_surface_adhesion_forces = getCellSurfaceAdhesionForce<T>(
        rz0, n0_z1, length, R, sigma0
    );
    forces_du += cell_surface_adhesion_forces; 

    // Calculate the Jacobian of the position and orientation vectors w.r.t 
    // the two perturbation coordinates, w and u
    //
    // To store the derivatives of the Jacobian entries w.r.t w and u, we need
    // two matrices 
    DualMatrix<T> jac_dw = DualMatrix<T>::Zero(6, 2);   // Store derivatives w.r.t w
    DualMatrix<T> jac_du = DualMatrix<T>::Zero(6, 2);   // Store derivatives w.r.t u
    
    // Get the Jacobian with derivatives w.r.t w
    DualNumber<T> w(0, 1), u(0, 0);  
    jac_dw(2, 0).setValue(1);  
    jac_dw(3, 1) = -u / sqrt(1 - u * u);
    jac_dw(5, 1).setValue(1);

    // Get the Jacobian with derivatives w.r.t u
    DualNumber<T> w_(0, 0), u_(0, 1); 
    jac_du(2, 0).setValue(1); 
    jac_du(3, 1) = -u_ / sqrt(1 - u_ * u_);
    jac_du(5, 1).setValue(1);

    // Extract the values of the Jacobian 
    Matrix<T, 6, 2> jac; 
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            jac(i, j) = jac_dw(i, j).getValue(); 
        }
    }

    // Extract the derivatives of the Jacobian w.r.t w and u
    Matrix<T, 6, 2> djac_dw, djac_du;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            djac_dw(i, j) = jac_dw(i, j).getDerivative();
            djac_du(i, j) = jac_du(i, j).getDerivative(); 
        }
    }

    // Calculate the left-hand dissipation matrix
    Matrix<T, 3, 1> r0_, n0_; 
    r0_ << 0, 0, rz; 
    n0_ << 1, 0, 0; 
    Matrix<T, 6, 6> diss = getDissipationMatrix<T>(r0_, n0_, length, R, eta0, eta1);
    
    // Extract the force values 
    Matrix<T, 6, 1> forces = Matrix<T, 6, 1>::Zero();
    for (int i = 0; i < 6; ++i)
        forces(i) = forces_dw(i).getValue();

    // Extract the derivatives of the forces w.r.t w and u
    Matrix<T, 6, 1> dforces_dw, dforces_du; 
    for (int i = 0; i < 6; ++i)
    {
        dforces_dw(i) = forces_dw(i).getDerivative();
        dforces_du(i) = forces_du(i).getDerivative(); 
    }

    // Project the force vector onto the two perturbation coordinates
    Matrix<T, 2, 2> K = Matrix<T, 2, 2>::Zero();
    K.col(0) = djac_dw.transpose() * forces + jac.transpose() * dforces_dw;
    K.col(1) = djac_du.transpose() * forces + jac.transpose() * dforces_du;

    // Calculate the reduced dissipation matrix 
    Matrix<T, 2, 2> diss_reduced = jac.transpose() * diss * jac;

    // Solve the generalized eigenvalue problem 
    LLT<Matrix<T, 2, 2> > llt(diss_reduced);  
    if (llt.info() != Eigen::Success)
    {
        throw std::runtime_error(
            "Reduced dissipation matrix is not positive definite"
        ); 
    }
    GeneralizedSelfAdjointEigenSolver<Matrix<T, 2, 2> > solver(K, diss_reduced);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Generalized eigendecomposition failed"); 
    }

    return solver.eigenvalues();
}

/**
 * Generate the 3-cell configuration with the given overlap and given angle 
 * between the central cell and the surface.
 *
 * The z-position is set to the equilibrium position for a horizontal cell, 
 * given the elastic modulus of the EPS and the cell-surface adhesion energy
 * density. 
 *
 * @param delta Cell-cell overlap. 
 * @param theta Angle between the central cell and the surface.
 * @param length Cell length (assumed to be the same for all cells).  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param E0 Elastic modulus of the EPS. 
 * @param sigma0 Cell-surface adhesion energy density.
 * @returns Array of cell position and orientation coordinates. 
 */
template <typename T>
Matrix<T, Dynamic, 6> getTwoNeighborConfiguration(const T delta, const T theta, 
                                                  const T length, const T R,
                                                  const T Rcell, const T E0,
                                                  const T sigma0)
{
    // Get the equilibrium z-position 
    const T rz = R - pow(sigma0 * sqrt(R) / (4 * E0), 2. / 3.);

    // Get the z-orientation of the central cell 
    const T nz = sin(theta); 

    // Specify the cell coordinates
    Matrix<T, Dynamic, 6> coords(3, 6); 
    coords << 0,                         0, rz, sqrt(1 - nz * nz), 0, nz,
              -(length + 2 * R) + delta, 0, rz, 1,                 0, 0, 
              length + 2 * R - delta,    0, rz, 1,                 0, 0;

    return coords;  
}

/**
 * Generate constraints for the 3-cell configuration. 
 *
 * The three cells must, at a minimum, all remain horizontal.
 *
 * @returns Constraint matrix for the 3-cell configuration. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getTwoNeighborConfigurationConstraints()
{
    // The y-positions and y-orientations of all three cells must remain zero
    const int n_cells = 3; 
    const int n_vars = 6 * n_cells;  
    Matrix<T, Dynamic, Dynamic> constraints = Matrix<T, Dynamic, Dynamic>::Zero(6, n_vars);
    int constraint_idx = 0; 
    for (int i = 0; i < 3; ++i)
    {
        const int ry_idx = 6 * i + 1;
        const int ny_idx = 6 * i + 4; 
        constraints(constraint_idx, ry_idx) = 1;
        constraint_idx++; 
        constraints(constraint_idx, ny_idx) = 1; 
        constraint_idx++;
    }

    return constraints; 
}

/**
 * Generate external inward forces of the given magnitude on the 3-cell
 * configuration.
 *
 * @param magnitude Force magnitude. 
 * @returns Array of force vectors and normalized centerline coordinates at 
 *          which the forces are applied. 
 */
template <typename T>
Matrix<T, Dynamic, 4> getTwoNeighborConfigurationInwardForces(const T magnitude)
{
    Matrix<T, Dynamic, 4> forces(3, 4);
    forces <<          0, 0, 0,  0,
               magnitude, 0, 0, -1,
              -magnitude, 0, 0,  1;  

    return forces; 
}

/**
 * Generate the 9-cell configuration with the given overlap and given angle 
 * between the central cell and the surface. 
 *
 * The z-position is set to the equilibrium position for a horizontal cell, 
 * given the elastic modulus of the EPS and the cell-surface adhesion energy
 * density. 
 *
 * @param delta Cell-cell overlap. 
 * @param theta Angle between the central cell and the surface.
 * @param length Cell length (assumed to be the same for all cells).  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param E0 Elastic modulus of the EPS. 
 * @param sigma0 Cell-surface adhesion energy density.
 * @returns Array of cell position and orientation coordinates. 
 */
template <typename T>
Matrix<T, Dynamic, 6> getEightNeighborConfiguration(const T delta, const T theta, 
                                                    const T length, const T R,
                                                    const T Rcell, const T E0,
                                                    const T sigma0)
{
    // Get the equilibrium z-position 
    const T rz = R - pow(sigma0 * sqrt(R) / (4 * E0), 2. / 3.);

    // Get the z-orientation of the central cell 
    const T nz = sin(theta); 

    // Specify the cell coordinates
    const T phi = boost::math::constants::quarter_pi<T>();
    const T a = (2 * R + length / 2 - delta) * cos(phi); 
    const T b = (2 * R + length / 2 - delta) * sin(phi); 
    const T c = 2 * R + length / 2 - delta; 
    Matrix<T, Dynamic, 6> coords(9, 6); 
    coords << 0,                         0,  rz, sqrt(1 - nz * nz), 0,         nz,   // Center
              -(length + 2 * R) + delta, 0,  rz, 1,                 0,         0,    // West
              -length / 2 - a,           b,  rz, -cos(phi),         sin(phi),  0,    // Northwest
              0,                         c,  rz, 0,                 1,         0,    // North
              length / 2 + a,            b,  rz, cos(phi),          sin(phi),  0,    // Northeast
              length + 2 * R - delta,    0,  rz, 1,                 0,         0,    // East
              length / 2 + a,            -b, rz, cos(phi),          -sin(phi), 0,    // Southeast
              0,                         -c, rz, 0,                 -1,        0,    // South
              -length / 2 - a,           -b, rz, -cos(phi),         -sin(phi), 0;    // Southwest 

    return coords;  
}

/**
 * Generate constraints for the 9-cell configuration. 
 *
 * These constraints enforce the following:
 * - The central cell retains its y-position and y-orientation.
 * - The east/west cells retain their y-positions and y-orientations.
 * - The west cell also retains its x- and z-orientations.  
 * - The north/south cells are symmetric. 
 * - The northeast and southeast cells are symmetric.
 * - The northwest and southwest cells are symmetric. 
 * - The north cell retains its x-position and x-orientation.
 * - The northeast cell retains its orientation when projected onto the xy-plane.
 * - The northwest cell retains its full orientation. 
 *
 * @returns Constraint matrix for the 9-cell configuration. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getEightNeighborConfigurationConstraints()
{
    // Set up the constraint matrix 
    const int n_cells = 9; 
    const int n_vars = 6 * n_cells;  
    Matrix<T, Dynamic, Dynamic> constraints = Matrix<T, Dynamic, Dynamic>::Zero(1, n_vars);
    const int idx_central = 0; 
    const int idx_west = 1; 
    const int idx_northwest = 2; 
    const int idx_north = 3; 
    const int idx_northeast = 4; 
    const int idx_east = 5; 
    const int idx_southeast = 6;
    const int idx_south = 7;
    const int idx_southwest = 8; 

    // Fix the y-position and y-orientation of the central cell
    int idx = 0;
    constraints.conservativeResize(idx + 2, n_vars); 
    constraints(Eigen::seqN(idx, 2), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(2, n_vars); 
    constraints(idx, 6 * idx_central + 1) = 1;
    idx++;
    constraints(idx, 6 * idx_central + 4) = 1; 
    idx++;
    
    // Fix the y-positions and y-orientations of the east/west cells
    constraints.conservativeResize(idx + 4, n_vars);
    constraints(Eigen::seqN(idx, 4), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(4, n_vars); 
    constraints(idx, 6 * idx_west + 1) = 1;
    idx++; 
    constraints(idx, 6 * idx_west + 4) = 1;
    idx++; 
    constraints(idx, 6 * idx_east + 1) = 1;
    idx++; 
    constraints(idx, 6 * idx_east + 4) = 1;
    idx++;

    // Fix the x- and z-orientations of the west cell 
    constraints.conservativeResize(idx + 2, n_vars); 
    constraints(Eigen::seqN(idx, 2), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(2, n_vars); 
    constraints(idx, 6 * idx_west + 3) = 1; 
    idx++; 
    constraints(idx, 6 * idx_west + 5) = 1;
    idx++;  

    // Introduce symmetry between the north and south cells
    constraints.conservativeResize(idx + 6, n_vars);
    constraints(Eigen::seqN(idx, 6), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(6, n_vars);  
    constraints(idx, 6 * idx_north) = 1; 
    constraints(idx, 6 * idx_south) = -1;       // x-positions are the same
    idx++; 
    constraints(idx, 6 * idx_north + 1) = 1;
    constraints(idx, 6 * idx_south + 1) = 1;    // y-positions are mirrored 
    idx++;
    constraints(idx, 6 * idx_north + 2) = 1; 
    constraints(idx, 6 * idx_south + 2) = -1;   // z-positions are the same
    idx++;
    constraints(idx, 6 * idx_north + 3) = 1;
    constraints(idx, 6 * idx_south + 3) = -1;   // x-orientations are the same (both zero)
    idx++; 
    constraints(idx, 6 * idx_north + 4) = 1;
    constraints(idx, 6 * idx_south + 4) = 1;    // y-orientations are mirrored
    idx++;
    constraints(idx, 6 * idx_north + 5) = 1;
    constraints(idx, 6 * idx_south + 5) = -1;   // z-orientations are the same
    idx++;

    // Introduce symmetry between the northeast and southeast cells
    constraints.conservativeResize(idx + 6, n_vars);
    constraints(Eigen::seqN(idx, 6), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(6, n_vars);  
    constraints(idx, 6 * idx_northeast) = 1; 
    constraints(idx, 6 * idx_southeast) = -1;       // x-positions are the same
    idx++; 
    constraints(idx, 6 * idx_northeast + 1) = 1;
    constraints(idx, 6 * idx_southeast + 1) = 1;    // y-positions are mirrored 
    idx++; 
    constraints(idx, 6 * idx_northeast + 2) = 1; 
    constraints(idx, 6 * idx_southeast + 2) = -1;   // z-positions are the same
    idx++; 
    constraints(idx, 6 * idx_northeast + 3) = 1;
    constraints(idx, 6 * idx_southeast + 3) = -1;   // x-orientations are the same
    idx++; 
    constraints(idx, 6 * idx_northeast + 4) = 1;
    constraints(idx, 6 * idx_southeast + 4) = 1;    // y-orientations are mirrored
    idx++; 
    constraints(idx, 6 * idx_northeast + 5) = 1;
    constraints(idx, 6 * idx_southeast + 5) = -1;   // z-orientations are the same
    idx++; 

    // Introduce symmetry between the northwest and southwest cells
    constraints.conservativeResize(idx + 6, n_vars);
    constraints(Eigen::seqN(idx, 6), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(6, n_vars);  
    constraints(idx, 6 * idx_northwest) = 1; 
    constraints(idx, 6 * idx_southwest) = -1;       // x-positions are the same
    idx++; 
    constraints(idx, 6 * idx_northwest + 1) = 1;
    constraints(idx, 6 * idx_southwest + 1) = 1;    // y-positions are mirrored 
    idx++; 
    constraints(idx, 6 * idx_northwest + 2) = 1; 
    constraints(idx, 6 * idx_southwest + 2) = -1;   // z-positions are the same
    idx++; 
    constraints(idx, 6 * idx_northwest + 3) = 1;
    constraints(idx, 6 * idx_southwest + 3) = -1;   // x-orientations are the same
    idx++; 
    constraints(idx, 6 * idx_northwest + 4) = 1;
    constraints(idx, 6 * idx_southwest + 4) = 1;    // y-orientations are mirrored
    idx++; 
    constraints(idx, 6 * idx_northwest + 5) = 1;
    constraints(idx, 6 * idx_southwest + 5) = -1;   // z-orientations are the same
    idx++; 

    // Fix the x-position and x-orientation of the north cell
    constraints.conservativeResize(idx + 2, n_vars);
    constraints(Eigen::seqN(idx, 2), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(2, n_vars);  
    constraints(idx, 6 * idx_north) = 1; 
    idx++;
    constraints(idx, 6 * idx_north + 3) = 1; 
    idx++;

    // Fix the xy-projection of the orientation of the northeast cell
    const T phi = boost::math::constants::quarter_pi<T>();  
    constraints.conservativeResize(idx + 1, n_vars); 
    constraints.row(idx) = Matrix<T, Dynamic, Dynamic>::Zero(1, n_vars);
    constraints(idx, 6 * idx_northeast + 3) = -sin(phi); 
    constraints(idx, 6 * idx_northeast + 4) = cos(phi);
    idx++;  

    // Fix the x-, y-, and z-orientations of the northwest cell 
    constraints.conservativeResize(idx + 3, n_vars); 
    constraints(Eigen::seqN(idx, 3), Eigen::all) = Matrix<T, Dynamic, Dynamic>::Zero(3, n_vars); 
    constraints(idx, 6 * idx_northwest + 3) = 1; 
    idx++;
    constraints(idx, 6 * idx_northwest + 4) = 1; 
    idx++;
    constraints(idx, 6 * idx_northwest + 5) = 1; 

    return constraints; 
}

/**
 * Generate external inward forces of the given magnitude on the 9-cell
 * configuration.
 *
 * @param magnitude Force magnitude. 
 * @returns Array of force vectors and normalized centerline coordinates at 
 *          which the forces are applied. 
 */
template <typename T>
Matrix<T, Dynamic, 4> getEightNeighborConfigurationInwardForces(const T magnitude)
{
    const T phi = boost::math::constants::quarter_pi<T>(); 
    Matrix<T, Dynamic, 4> forces(9, 4);
    forces <<                     0,                     0, 0,  0,
                          magnitude,                     0, 0, -1,      // West
               magnitude * cos(phi), -magnitude * sin(phi), 0,  1,      // Northwest 
                                  0,            -magnitude, 0,  1,      // North
              -magnitude * cos(phi), -magnitude * sin(phi), 0,  1,      // Northeast
                         -magnitude,                     0, 0,  1,      // East 
              -magnitude * cos(phi),  magnitude * sin(phi), 0,  1,      // Southeast
                                  0,             magnitude, 0,  1,      // South
               magnitude * cos(phi),  magnitude * sin(phi), 0,  1;      // Southwest

    return forces; 
}

/**
 * Find the critical cell-cell overlap in the 9-cell configuration, using
 * Brent's method over a range of admissible overlaps.  
 *
 * @param length Cell length (assumed to be the same for all cells). 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param E0 Elastic modulus of the EPS.
 * @param Ecell Elastic modulus of the cell body.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param eta0 Ambient viscosity. 
 * @param eta1 Cell-surface friction coefficient. 
 * @param gamma Cell-cell adhesion energy density.  
 * @param delta_min Minimum cell-cell overlap. 
 * @param delta_max Maximum cell-cell overlap. 
 * @param tol Tolerance for Brent's method.
 * @param max_iter Maximum number of iterations for Brent's method. 
 * @param verbose If true, print intermittent output to stdout.   
 * @returns Critical cell-cell overlap at which the system transitions from 
 *          stable to unstable.  
 */
template <typename T>
T findCriticalOverlapEightNeighbor(const T length, const T R, const T Rcell, 
                                   const T E0, const T Ecell, const T sigma0, 
                                   const T eta0, const T eta1, const T gamma,
                                   const T delta_min, const T delta_max, 
                                   const T tol = 1e-8, const int max_iter = 1000,
                                   const bool verbose = false)
{
    // Set up the function whose root is to be localized 
    std::function<T(const T)> func = [&length, &R, &Rcell, &E0, &Ecell, &sigma0, &eta0, &eta1, &gamma](const T delta)
    {
        Matrix<T, Dynamic, 6> coords = getEightNeighborConfiguration<T>(
            delta, 0.0, length, R, Rcell, E0, sigma0
        );
        Matrix<T, Dynamic, 2> neighbors_rxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(0, 2)); 
        Matrix<T, Dynamic, 2> neighbors_nxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(3, 2));
        Matrix<T, 2, 1> eigenvalues = get2DStability<T>(
            neighbors_rxy, neighbors_nxy, length, R, Rcell, E0, Ecell, sigma0, 
            eta0, eta1, gamma
        ); 
        return eigenvalues.maxCoeff(); 
    };

    // Use Brent's method to find the critical overlap 
    std::pair<T, T> bracket = brent<T>(
        func, delta_min, delta_max, tol, max_iter, verbose
    ); 
    return (bracket.first + bracket.second) / 2;  
}

/**
 * Generate the 11-cell configuration with the given overlap and given angle 
 * between the central cell and the surface. 
 *
 * The z-position is set to the equilibrium position for a horizontal cell, 
 * given the elastic modulus of the EPS and the cell-surface adhesion energy
 * density. 
 *
 * @param delta Cell-cell overlap. 
 * @param theta Angle between the central cell and the surface.
 * @param length Cell length (assumed to be the same for all cells).  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param E0 Elastic modulus of the EPS. 
 * @param sigma0 Cell-surface adhesion energy density.
 * @returns Array of cell position and orientation coordinates. 
 */
template <typename T>
Matrix<T, Dynamic, 6> getTenNeighborConfiguration(const T delta, const T theta, 
                                                  const T length, const T R,
                                                  const T Rcell, const T E0,
                                                  const T sigma0)
{
    // Get the equilibrium z-position 
    const T rz = R - pow(sigma0 * sqrt(R) / (4 * E0), 2. / 3.);

    // Get the z-orientation of the central cell 
    const T nz = sin(theta); 

    // Specify the cell coordinates
    const T phi = boost::math::constants::quarter_pi<T>();
    const T a = (2 * R + length / 2 - delta) * cos(phi); 
    const T b = (2 * R + length / 2 - delta) * sin(phi); 
    const T c = (2. / 3.) * (length / 2); 
    const T d = 2 * R + length / 2 - delta; 
    Matrix<T, Dynamic, 6> coords(11, 6); 
    coords << 0,                         0,  rz, sqrt(1 - nz * nz), 0,         nz,   // Center
              -(length + 2 * R) + delta, 0,  rz, 1,                 0,         0,    // West
              -length / 2 - a,           b,  rz, -cos(phi),         sin(phi),  0,    // Northwest
              -c,                        d,  rz, 0,                 1,         0,    // North, left
              c,                         d,  rz, 0,                 1,         0,    // North, right
              length / 2 + a,            b,  rz, cos(phi),          sin(phi),  0,    // Northeast
              length + 2 * R - delta,    0,  rz, 1,                 0,         0,    // East
              length / 2 + a,            -b, rz, cos(phi),          -sin(phi), 0,    // Southeast
              c,                         -d, rz, 0,                 -1,        0,    // South, right
              -c,                        -d, rz, 0,                 -1,        0,    // South, left
              -length / 2 - a,           -b, rz, -cos(phi),         -sin(phi), 0;    // Southwest 

    return coords;  
}

#endif
