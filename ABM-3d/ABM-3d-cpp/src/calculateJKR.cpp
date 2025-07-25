/**
 * Calculate JKR contact forces over a collection of cell-cell configurations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/18/2025
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <utility>
#include <chrono>
#include <Eigen/Dense>
#include <boost/json/src.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <CGAL/Kd_tree.h>
#include "../include/utils.hpp"
#include "../include/distances.hpp"
#include "../include/adhesion.hpp"
#include "../include/jkr.hpp"

using boost::multiprecision::abs;
using std::sqrt; 
using boost::multiprecision::sqrt;
using boost::multiprecision::log10;
using std::pow; 
using boost::multiprecision::pow; 

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > PreciseType;

/**
 * Calculate the Hertz-JKR equilibrium cell-cell distance for the given
 * surface energy density.
 *
 * @param R Cell radius (including the EPS). 
 * @param E Elastic modulus.
 * @param gamma Surface energy density. 
 * @param min_overlap Minimum overlap distance. 
 * @param max_overlap Maximum overlap distance.
 * @param rng Random number generator.  
 * @param d_overlap Increment for finite differences approximation.
 * @param tol Tolerance for Newton's method.  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @returns Equilibrium cell-cell distance.  
 */
template <typename T>
T jkrEquilibriumDistance(const T R, const T E, const T gamma, const T min_overlap,
                         const T max_overlap, boost::random::mt19937& rng,
                         const T d_overlap = 1e-8, const T tol = 1e-8, 
                         const T imag_tol = 1e-8, const T aberth_tol = 1e-20)
{
    // Get an initial cell-cell distance
    T dmin = 2 * R - max_overlap; 
    T dmax = 2 * R - min_overlap;
    boost::random::uniform_01<> uniform_dist;  
    T dist = dmin + (dmax - dmin) * uniform_dist(rng);
    T overlap = 2 * R - dist;  
    T update = std::numeric_limits<T>::infinity();

    // While the update between consecutive overlaps is larger than the 
    // given tolerance ... 
    while (abs(update) > tol)
    {
        // Compute the contact radius for each overlap in the mesh
        T radius = jkrContactRadius<T>(overlap, R, E, gamma, imag_tol, aberth_tol).second;

        // Compute the corresponding force
        T prefactor1 = static_cast<T>(4) / static_cast<T>(3) * E / R;
        T prefactor2 = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E); 
        T f_hertz = prefactor1 * radius * radius * radius; 
        T f_jkr = prefactor2 * pow(radius, 1.5); 
        T force = f_hertz - f_jkr;

        // Estimate the derivative of this force w.r.t the overlap
        T radius_plus = jkrContactRadius<T>(
            overlap + d_overlap, R, E, gamma, imag_tol, aberth_tol
        ).second;
        T radius_minus = jkrContactRadius<T>(
            overlap - d_overlap, R, E, gamma, imag_tol, aberth_tol
        ).second;
        T f_hertz_plus = prefactor1 * radius_plus * radius_plus * radius_plus; 
        T f_jkr_plus = prefactor2 * pow(radius_plus, 1.5); 
        T force_plus = f_hertz_plus - f_jkr_plus; 
        T f_hertz_minus = prefactor1 * radius_minus * radius_minus * radius_minus; 
        T f_jkr_minus = prefactor2 * pow(radius_minus, 1.5); 
        T force_minus = f_hertz_minus - f_jkr_minus;
        T deriv = (force_plus - force_minus) / (2 * d_overlap);

        // Update the overlap according to Newton's method 
        update = -force / deriv;
        overlap += update;
    }

    return 2 * R - overlap; 
}

/**
 * Calculate the surface energy density for which the desired Hertz-JKR 
 * equilibrium cell-cell distance is achieved.
 *
 * @param R Cell radius (including the EPS). 
 * @param E Elastic modulus.
 * @param deq_target Target equilibrium cell-cell distance. 
 * @param min_gamma Minimum surface energy density.
 * @param max_gamma Maximum surface energy density.  
 * @param min_overlap Minimum overlap distance. 
 * @param max_overlap Maximum overlap distance.
 * @param rng Random number generator.
 * @param tol Tolerance for (steepest) gradient descent.
 * @param d_log_gamma Increment for finite differences approximation w.r.t.
 *                    log10(gamma) during gradient descent. 
 * @param max_learn_rate Maximum learning rate.
 * @param d_overlap Increment for finite differences approximation w.r.t. 
 *                  overlap in jkrEquilibriumDistance().
 * @param newton_tol Tolerance for Newton's method in jkrEquilibriumDistance().  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print iteration details to stdout.
 * @returns Optimal surface energy density for the desired equilibrium 
 *          cell-cell distance.  
 */
template <typename T>
T jkrOptimalSurfaceEnergyDensity(const T R, const T E, const T deq_target, 
                                 const T min_gamma, const T max_gamma,
                                 const T min_overlap, const T max_overlap,
                                 boost::random::mt19937& rng, const T tol = 1e-8,
                                 const T d_log_gamma = 1e-6,
                                 const T max_learn_rate = 1.0,  
                                 const T d_overlap = 1e-8, const T newton_tol = 1e-8,
                                 const T imag_tol = 1e-8, const T aberth_tol = 1e-20,
                                 const bool verbose = false)
{
    // Get an initial value for gamma
    boost::random::uniform_01<> dist;  
    T log_gamma = log10(min_gamma) + (log10(max_gamma) - log10(min_gamma)) * dist(rng);
    T gamma = pow(10.0, log_gamma); 
    T update = std::numeric_limits<T>::infinity();
    int iter = 0; 
    if (verbose)
        std::cout << std::setprecision(10);

    // Calculate the deviation of the current equilibrium distance from
    // the target equilibrium distance 
    T deq = jkrEquilibriumDistance<T>(
        R, E, gamma, min_overlap, max_overlap, rng, d_overlap, newton_tol,
        imag_tol, aberth_tol
    );
    T error = abs(deq - deq_target);

    // While the update between consecutive values of gamma is larger than
    // the given tolerance ... 
    while (abs(update) > tol)
    {
        // Estimate the derivative of this deviation w.r.t. gamma
        T gamma_plus = pow(10.0, log_gamma + d_log_gamma);
        T gamma_minus = pow(10.0, log_gamma - d_log_gamma);  
        T deq_plus = jkrEquilibriumDistance<T>(
            R, E, gamma_plus, min_overlap, max_overlap, rng, d_overlap,
            newton_tol, imag_tol, aberth_tol
        );
        T error_plus = abs(deq_plus - deq_target);  
        T deq_minus = jkrEquilibriumDistance<T>(
            R, E, gamma_minus, min_overlap, max_overlap, rng, d_overlap,
            newton_tol, imag_tol, aberth_tol
        );
        T error_minus = abs(deq_minus - deq_target);  
        T deriv = (error_plus - error_minus) / (2 * d_log_gamma);

        // Determine a learning rate that satisfies the Armijo condition
        // via backtracking line search (Nocedal and Wright, Algorithm 3.1)
        //
        // Start with the maximum learning rate 
        T learn_rate = max_learn_rate;
        update = -learn_rate * deriv;

        // Check if the proposed update exceeds the given bounds  
        if (pow(10.0, log_gamma + update) < min_gamma)
        {
            update = log10(min_gamma) - log_gamma;
            learn_rate = -update / deriv; 
        }
        else if (pow(10.0, log_gamma + update) > max_gamma)
        {
            update = log10(max_gamma) - log_gamma;
            learn_rate = -update / deriv; 
        }

        // Compute the new cell-cell equilibrium distance 
        T deq_new = jkrEquilibriumDistance<T>(
            R, E, pow(10.0, log_gamma + update), min_overlap, max_overlap, rng,
            d_overlap, newton_tol, imag_tol, aberth_tol
        );
        T error_new = abs(deq_new - deq_target);

        // If the Armijo condition is not satisfied (set c = 0.1) ...  
        while (error_new > error - 0.1 * learn_rate * deriv * deriv)
        {
            // Lower the learning rate and try again (set contraction factor
            // \rho = 0.5) 
            learn_rate *= 0.5;
            update = -learn_rate * deriv;

            // Check if the proposed update exceeds the given bounds  
            if (pow(10.0, log_gamma + update) < min_gamma)
            {
                update = log10(min_gamma) - log_gamma;
                learn_rate = -update / deriv; 
            }
            else if (pow(10.0, log_gamma + update) > max_gamma)
            {
                update = log10(max_gamma) - log_gamma;
                learn_rate = -update / deriv; 
            }
            
            // Compute the new cell-cell equilibrium distance 
            deq_new = jkrEquilibriumDistance<T>(
                R, E, pow(10.0, log_gamma + update), min_overlap, max_overlap,
                rng, d_overlap, newton_tol, imag_tol, aberth_tol
            );
            error_new = abs(deq_new - deq_target); 
        }
        
        // Print iteration details if desired 
        if (verbose)
            std::cout << "Iteration " << iter << ": gamma = " << gamma << ", "
                      << "deq = " << deq << ", log update = " << update << std::endl; 

        // Update gamma according to the given learning rate 
        log_gamma += update;
        gamma = pow(10.0, log_gamma + update); 
        deq = deq_new; 
        error = error_new; 
        iter++; 
    } 

    return gamma; 
}

/**
 * Calculate JKR forces for a given collection of cell-cell configurations.
 *
 * Each cell-cell configuration involves one cell at the origin, parallel to
 * the x-axis, with variable half-length, and another cell with variable
 * center, orientation, and half-length.
 *
 * The returned table maps the indices (i, j, k, l) to the corresponding 
 * JKR force, where:
 *
 * - half_l(i) is the half-length of cell 1.
 * - r2.row(j) is the center of cell 2.
 * - n2.row(k) is the orientation of cell 2.
 * - half_l(l) is the half-length of cell 2.
 *
 * @param half_l Mesh of cell half-lengths. 
 * @param r2 Lattice of cell 2 centers. 
 * @param n2 Mesh of cell 2 orientations. 
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus.
 * @param gamma Surface energy density.
 * @param dmin Minimum cell-cell distance at which to calculate JKR forces. 
 *             If the cell-cell distance is less than this value, this function
 *             returns nan's (encoded as a large float) for the forces. 
 * @param JKRMode JKR force mode (either isotropic or anisotropic). 
 * @param n_ellip_table Number of values to calculate for the elliptic integral
 *                      function used to calculate contact areas.
 * @param project_tol Tolerance for the ellipsoid projection method. 
 * @param project_max_iter Maximum number of iterations for the ellipsoid
 *                         projection method. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @returns Table of calculated JKR forces. 
 */
template <typename T, int Dim>
using ForceTable = std::unordered_map<std::tuple<int, int, int, int>,
                                      Array<T, 2, 2 * Dim>,
                                      boost::hash<std::tuple<int, int, int, int> > >;
template <typename T, int Dim>
ForceTable<double, Dim> calculateJKRForceTable(const Ref<const Matrix<T, Dynamic, 1> >& half_l,
                                               const Ref<const Matrix<T, Dynamic, Dim> >& r2, 
                                               const Ref<const Matrix<T, Dynamic, Dim> >& n2,
                                               const T R, const T E0, const T gamma, 
                                               const T dmin, const T dmax, 
                                               const JKRMode mode,
                                               const int n_ellip_table = 100,
                                               const T project_tol = 1e-6,
                                               const int project_max_iter = 100,
                                               const T imag_tol = 1e-8,
                                               const T aberth_tol = 1e-20)
{
    const double BIGNUM = 1e+8; 
    K kernel; 
    Matrix<T, Dim, 1> r1 = Matrix<T, Dim, 1>::Zero();
    Matrix<T, Dim, 1> n1 = Matrix<T, Dim, 1>::Zero(); 
    n1(0) = 1;
    Matrix<T, Dynamic, 4> ellip_table = getEllipticIntegralTable<T>(n_ellip_table);  
    ForceTable<double, Dim> forces;
    const int n_total = half_l.size() * half_l.size() * r2.rows() * n2.rows(); 
    std::cout << "- Half-length mesh size = " << half_l.size() << std::endl; 
    std::cout << "- Cell center lattice size = " << r2.rows() << std::endl; 
    std::cout << "- Cell orientation mesh size = " << n2.rows() << std::endl;
    std::cout << "- Maximum number of forces to calculate = " << n_total << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();  

    #pragma omp parallel for 
    int n_computed = 0; 
    int n_ignored = 0; 
    for (int i = 0; i < half_l.size(); ++i)
    {
        for (int j = 0; j < r2.rows(); ++j)
        {
            for (int k = 0; k < n2.rows(); ++k)
            {
                for (int m = i; m < half_l.size(); ++m)
                {
                    Matrix<double, 3, 1> r1_, n1_, r2_, n2_; 
                    if (Dim == 2)
                    {
                        r1_ << static_cast<double>(r1(0)),
                               static_cast<double>(r1(1)),
                               0; 
                        n1_ << static_cast<double>(n1(0)),
                               static_cast<double>(n1(1)),
                               0; 
                        r2_ << static_cast<double>(r2(j, 0)),
                               static_cast<double>(r2(j, 1)),
                               0;
                        n2_ << static_cast<double>(n2(k, 0)),
                               static_cast<double>(n2(k, 1)),
                               0;  
                    }
                    else 
                    {
                        r1_ << static_cast<double>(r1(0)),
                               static_cast<double>(r1(1)),
                               static_cast<double>(r1(2));  
                        n1_ << static_cast<double>(n1(0)),
                               static_cast<double>(n1(1)),
                               static_cast<double>(n1(2));
                        r2_ << static_cast<double>(r2(j, 0)),
                               static_cast<double>(r2(j, 1)),
                               static_cast<double>(r2(j, 2));
                        n2_ << static_cast<double>(n2(k, 0)),
                               static_cast<double>(n2(k, 1)),
                               static_cast<double>(n2(k, 2));
                    }
                    Segment_3 cell1 = generateSegment<double>(
                        r1_, n1_, static_cast<double>(half_l(i))
                    );  
                    Segment_3 cell2 = generateSegment<double>(
                        r2_, n2_, static_cast<double>(half_l(m))
                    ); 
                    auto result = distBetweenCells<double>(
                        cell1, cell2, 0, r1_, n1_, static_cast<double>(half_l(i)),
                        1, r2_, n2_, static_cast<double>(half_l(m)), kernel
                    );
                    Matrix<T, Dim, 1> d12;
                    if (Dim == 2)
                        d12 << static_cast<T>(std::get<0>(result)(0)), 
                               static_cast<T>(std::get<0>(result)(1)); 
                    else    // Dim == 3
                        d12 << static_cast<T>(std::get<0>(result)(0)), 
                               static_cast<T>(std::get<0>(result)(1)),
                               static_cast<T>(std::get<0>(result)(2));
                    T s = static_cast<T>(std::get<1>(result)); 
                    T t = static_cast<T>(std::get<2>(result));

                    // Check that the distance is nonzero
                    if (d12.norm() > dmin && d12.norm() < dmax)
                    {
                        Array<T, 2, 2 * Dim> forces_ijkm; 
                        if (mode == JKRMode::ISOTROPIC)
                        {
                            forces_ijkm = forcesIsotropicJKRLagrange<T, Dim>(
                                n1, n2.row(k), d12, R, E0, gamma, s, t, true,
                                imag_tol, aberth_tol
                            ); 
                        }
                        else    // mode == JKRMode::ANISOTROPIC
                        {
                            forces_ijkm = forcesAnisotropicJKRLagrange<T, Dim>(
                                r1, n1, half_l(i), r2.row(j), n2.row(k),
                                half_l(m), d12, R, E0, gamma, s, t, ellip_table,
                                true, project_tol, project_max_iter, imag_tol,
                                aberth_tol
                            ); 
                        }
                        forces[std::make_tuple(i, j, k, m)] = forces_ijkm.template cast<double>();
                        n_computed++; 
                    }
                    else    // Otherwise, add an array of large constants 
                    {
                        forces[std::make_tuple(i, j, k, m)] = BIGNUM * Array<double, 2, 2 * Dim>::Ones();
                        n_ignored++;  
                    }

                    // Print progress to stdout 
                    if ((n_computed + n_ignored) % 10000 == 0)
                    {
                        auto t_curr = std::chrono::high_resolution_clock::now();  
                        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                            t_curr - t_start
                        ); 
                        std::cout << "... Considered " << n_computed + n_ignored 
                                  << " input configurations ("
                                  << 100.0 * (n_computed + n_ignored) / n_total
                                  << "%)" << std::endl 
                                  << "    - Computed " << n_computed << " forces"
                                  << std::endl
                                  << "    - Ignored " << n_ignored << " configurations"
                                  << std::endl
                                  << "    - Time elapsed = " << duration.count()
                                  << " sec" << std::endl; 
                    } 
                }
            }
        }
    }

    return forces;
} 

int main(int argc, char** argv)
{
    // Parse input json file 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Define required input parameters
    const PreciseType R = static_cast<PreciseType>(json_data["R"].as_double());
    const PreciseType Rcell = static_cast<PreciseType>(json_data["Rcell"].as_double());
    const PreciseType E0 = static_cast<PreciseType>(json_data["E0"].as_double());
    const PreciseType lmin = static_cast<PreciseType>(json_data["lmin"].as_double());
    const PreciseType lmax = 2 * lmin + 2 * R;
    const PreciseType min_gamma = static_cast<PreciseType>(json_data["min_gamma"].as_double());
    const PreciseType max_gamma = static_cast<PreciseType>(json_data["max_gamma"].as_double());
    const PreciseType min_overlap = 0.0;
    const PreciseType max_overlap = 2 * R - 2 * Rcell;
    const PreciseType tol = static_cast<PreciseType>(json_data["opt_tol"].as_double()); 
    const PreciseType d_log_gamma = static_cast<PreciseType>(json_data["delta_log_gamma"].as_double()); 
    const PreciseType max_learn_rate = static_cast<PreciseType>(json_data["max_learn_rate"].as_double()); 
    const PreciseType d_overlap = static_cast<PreciseType>(json_data["delta_overlap"].as_double()); 
    const PreciseType newton_tol = static_cast<PreciseType>(json_data["eqdist_newton_tol"].as_double()); 
    const PreciseType imag_tol = static_cast<PreciseType>(json_data["imag_tol"].as_double()); 
    const PreciseType aberth_tol = static_cast<PreciseType>(json_data["aberth_tol"].as_double()); 
    const PreciseType project_tol = static_cast<PreciseType>(json_data["project_tol"].as_double());
    const int project_max_iter = json_data["project_max_iter"].as_int64();
    const bool verbose = true;
    boost::random::mt19937 rng(json_data["rng_seed"].as_int64()); 

    // Generate a uniform mesh of points on the unit sphere
    const int nmin_sphere_mesh = json_data["nmin_sphere_mesh"].as_int64();
    const bool restrict_zpos = true; 
    Matrix<PreciseType, Dynamic, 3> sphere_mesh = uniformMeshSphere<PreciseType>(
        nmin_sphere_mesh, restrict_zpos
    );

    // Generate a uniform mesh of points on the unit circle containing the 
    // square root of N points, where N is the size of the spherical mesh
    const int n_circle_mesh = json_data["n_circle_mesh"].as_int64(); 
    const bool restrict_ypos = true;  
    Matrix<PreciseType, Dynamic, 2> circle_mesh = uniformMeshCircle<PreciseType>(
        n_circle_mesh, restrict_ypos
    );

    // Generate a uniform lattice of cell centers in 2-D containing at least
    // N points 
    const int nmin_lattice_2d = sphere_mesh.rows();
    const PreciseType dmin = static_cast<PreciseType>(json_data["lattice_dmin"].as_double()); 
    const PreciseType dmax = static_cast<PreciseType>(json_data["lattice_dmax"].as_double()); 
    Matrix<PreciseType, Dynamic, 2> lattice_2d = uniformLattice<PreciseType, 2>(
        nmin_lattice_2d, dmin, dmax, lmax
    );

    // Generate a uniform lattice of cell centers in 3-D containing at least
    // N^(3/2) points
    const int nmin_lattice_3d = static_cast<int>(pow(static_cast<double>(sphere_mesh.rows()), 1.5)); 
    Matrix<PreciseType, Dynamic, 3> lattice_3d = uniformLattice<PreciseType, 3>(
        nmin_lattice_3d, dmin, dmax, lmax
    );

    // Generate a uniform mesh of cell half-lengths 
    const int n_half_l = n_circle_mesh; 
    Matrix<PreciseType, Dynamic, 1> half_l
        = Matrix<PreciseType, Dynamic, 1>::LinSpaced(n_half_l, 0.5 * lmin, 0.5 * lmax); 
   
    // Identify optimal values of gamma for achieving the below surface 
    // separations
    const PreciseType deq_target = static_cast<PreciseType>(json_data["eqdist_target"].as_double());
    const JKRMode jkr_mode = static_cast<JKRMode>(json_data["jkr_mode"].as_int64());  
    const PreciseType opt_gamma = jkrOptimalSurfaceEnergyDensity<PreciseType>(  
        R, E0, deq_target, min_gamma, max_gamma, min_overlap, max_overlap, rng,
        tol, d_log_gamma, max_learn_rate, d_overlap, newton_tol, imag_tol,
        aberth_tol, verbose
    );

    // Calculate 2D forces ... 
    std::cout << "Calculating 2D forces ...\n"; 
    auto forces_2d = calculateJKRForceTable<PreciseType, 2>(
        half_l, lattice_2d, circle_mesh, R, E0, opt_gamma, dmin, dmax, jkr_mode,
        100, project_tol, project_max_iter, imag_tol, aberth_tol
    );

    // Calculate 3D forces ... 
    std::cout << "Calculating 3D forces ...\n"; 
    auto forces_3d = calculateJKRForceTable<PreciseType, 3>(
        half_l, lattice_3d, sphere_mesh, R, E0, opt_gamma, dmin, dmax, jkr_mode,
        100, project_tol, project_max_iter, imag_tol, aberth_tol
    );

    // Output meshes and forces to file
    //
    // First parse the output file paths from the input JSON file 
    std::ofstream outfile_lattice2d(json_data["outfile_mesh_lattice2d"].as_string().c_str());
    std::ofstream outfile_lattice3d(json_data["outfile_mesh_lattice3d"].as_string().c_str()); 
    std::ofstream outfile_circle(json_data["outfile_mesh_orientations2d"].as_string().c_str()); 
    std::ofstream outfile_sphere(json_data["outfile_mesh_orientations3d"].as_string().c_str());
    std::ofstream outfile_half_l(json_data["outfile_mesh_half_l"].as_string().c_str());
    std::ofstream outfile_forces2d(json_data["outfile_forces_2d"].as_string().c_str()); 
    std::ofstream outfile_forces3d(json_data["outfile_forces_3d"].as_string().c_str()); 
    outfile_lattice2d << std::setprecision(10); 
    outfile_lattice3d << std::setprecision(10); 
    outfile_circle << std::setprecision(10); 
    outfile_sphere << std::setprecision(10); 
    outfile_half_l << std::setprecision(10); 
    outfile_forces2d << std::setprecision(10); 
    outfile_forces3d << std::setprecision(10);

    // Start with the meshes ... 
    for (int i = 0; i < n_half_l; ++i)
        outfile_half_l << half_l(i) << std::endl; 
    for (int j = 0; j < lattice_2d.rows(); ++j)
        outfile_lattice2d << lattice_2d(j, 0) << '\t'
                          << lattice_2d(j, 1) << std::endl;
    for (int j = 0; j < lattice_3d.rows(); ++j)
        outfile_lattice3d << lattice_3d(j, 0) << '\t'
                          << lattice_3d(j, 1) << '\t'
                          << lattice_3d(j, 2) << std::endl;
    for (int k = 0; k < circle_mesh.rows(); ++k)
        outfile_circle << circle_mesh(k, 0) << '\t'
                       << circle_mesh(k, 1) << std::endl; 
    for (int k = 0; k < sphere_mesh.rows(); ++k)
        outfile_sphere << sphere_mesh(k, 0) << '\t'
                       << sphere_mesh(k, 1) << '\t'
                       << sphere_mesh(k, 2) << std::endl;

    // ... then output the forces
    const double BIGNUM = 1e+8;
    Array<double, 2, 4> na_forces2d = BIGNUM * Array<double, 2, 4>::Ones(); 
    Array<double, 2, 6> na_forces3d = BIGNUM * Array<double, 2, 6>::Ones(); 
    for (int i = 0; i < n_half_l; ++i)
    {
        for (int j = 0; j < lattice_2d.rows(); ++j)
        {
            for (int k = 0; k < circle_mesh.rows(); ++k)
            {
                for (int m = i; m < n_half_l; ++m)
                {
                    Array<double, 2, 4> forces = forces_2d[std::make_tuple(i, j, k, m)];
                    if ((forces - na_forces2d).abs().sum() < 1e-8)
                        outfile_forces2d << i << '\t' << j << '\t' << k << '\t' << m << '\t'
                                         << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA" << std::endl; 
                    else  
                        outfile_forces2d << i << '\t' << j << '\t' << k << '\t' << m << '\t'
                                         << forces(0, 0) << '\t' << forces(0, 1) << '\t'
                                         << forces(0, 2) << '\t' << forces(0, 3) << '\t'
                                         << forces(1, 0) << '\t' << forces(1, 1) << '\t'
                                         << forces(1, 2) << '\t' << forces(1, 3) << std::endl; 
                }
            }
        }
    }
    for (int i = 0; i < n_half_l; ++i)
    {
        for (int j = 0; j < lattice_3d.rows(); ++j)
        {
            for (int k = 0; k < sphere_mesh.rows(); ++k)
            {
                for (int m = i; m < n_half_l; ++m)
                {
                    Array<double, 2, 6> forces = forces_3d[std::make_tuple(i, j, k, m)];
                    if ((forces - na_forces3d).abs().sum() < 1e-8)
                        outfile_forces3d << i << '\t' << j << '\t' << k << '\t' << m << '\t'
                                         << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA" << std::endl; 
                    else  
                        outfile_forces3d << i << '\t' << j << '\t' << k << '\t' << m << '\t'
                                         << forces(0, 0) << '\t' << forces(0, 1) << '\t'
                                         << forces(0, 2) << '\t' << forces(0, 3) << '\t'
                                         << forces(0, 4) << '\t' << forces(0, 5) << '\t'
                                         << forces(1, 0) << '\t' << forces(1, 1) << '\t'
                                         << forces(1, 2) << '\t' << forces(1, 3) << '\t'
                                         << forces(1, 4) << '\t' << forces(1, 5) << std::endl; 
                }
            }
        }
    }

    outfile_lattice2d.close(); 
    outfile_lattice3d.close(); 
    outfile_circle.close(); 
    outfile_sphere.close(); 
    outfile_half_l.close(); 
    outfile_forces2d.close(); 
    outfile_forces3d.close();

    return 0;
}
