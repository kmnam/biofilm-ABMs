/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/11/2025
 */

#include <Eigen/Dense>
#include "../include/simulation.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Define floating-point type to be used 
typedef double T;

int main(int argc, char** argv)
{
    // Parse input json file 
    std::string json_filename = argv[1];
    boost::json::object json_data = parseConfigFile(json_filename).as_object();

    // Define required input parameters
    const T R = static_cast<T>(json_data["R"].as_double());
    const T Rcell = static_cast<T>(json_data["Rcell"].as_double());
    const T L0 = static_cast<T>(json_data["L0"].as_double());
    const T Ldiv = 2 * L0 + 2 * R;
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    
    // Parse cell-cell adhesion parameters ... 
    T eqdist = static_cast<T>(json_data["adhesion_eqdist"].as_double());

    // Parse optional input parameters for both JKR adhesion modes 
    T imag_tol = 1e-20;  
    T aberth_tol = 1e-20; 
    int n_mesh_overlap = 100;
    int n_mesh_gamma = 100; 
    try
    {
        imag_tol = static_cast<T>(json_data["adhesion_jkr_imag_tol"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        aberth_tol = static_cast<T>(json_data["adhesion_jkr_aberth_tol"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_overlap = json_data["adhesion_n_mesh_overlap"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_gamma = json_data["adhesion_n_mesh_gamma"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }

    // Parse optional input parameters for anisotropic JKR adhesion
    int n_mesh_theta = 100; 
    int n_mesh_half_l = 100; 
    int n_mesh_centerline_coords = 100;
    int n_mesh_curvature_radii = 100;
    bool calibrate_endpoint_radii = true;
    T min_aspect_ratio = 0.01; 
    T project_tol = 1e-8; 
    int project_max_iter = 100; 
    T newton_tol = 1e-8; 
    int newton_max_iter = 1000;
    try
    {
        calibrate_endpoint_radii = json_data["adhesion_calibrate_endpoint_radii"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_theta = json_data["adhesion_n_mesh_theta"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_half_l = json_data["adhesion_n_mesh_half_l"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_centerline_coords = json_data["adhesion_n_mesh_centerline_coords"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_curvature_radii = json_data["adhesion_n_mesh_curvature_radii"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        project_tol = static_cast<T>(
            json_data["adhesion_ellipsoid_project_tol"].as_double()
        ); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        project_max_iter = json_data["adhesion_ellipsoid_project_max_iter"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        newton_tol = static_cast<T>(json_data["adhesion_newton_tol"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        newton_max_iter = json_data["adhesion_newton_max_iter"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }

    // Output file prefix
    std::string outprefix = argv[2];

    // Compute anisotropic JKR forces ... 
    JKRData<T> jkr_data;
    
    // Solve for the surface energy density corresponding to the desired 
    // equilibrium distance  
    jkr_data.max_gamma = jkrOptimalSurfaceEnergyDensity<T, 100>(
        R, Rcell, E0, eqdist, 1.0, 1000.0, 1e-6, 1e-8, 1e-8, 1e-8, 1000,
        1000, imag_tol, aberth_tol, true
    );

    // Calculate principal radii of curvature 
    jkr_data.theta = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_theta, 0.0, boost::math::constants::half_pi<T>()
    ); 
    jkr_data.half_l = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_half_l, 0.5 * L0, 0.5 * Ldiv
    );  
    jkr_data.centerline_coords = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_centerline_coords, 0.0, 1.0
    );
    jkr_data.curvature_radii = calculateCurvatureRadiiTable<T>(
        jkr_data.theta, jkr_data.half_l, jkr_data.centerline_coords, 
        R, calibrate_endpoint_radii, project_tol, project_max_iter
    );

    // Calculate JKR forces for a range of cell-cell configurations 
    T max_overlap = 2 * (R - Rcell);
    jkr_data.Rx = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_curvature_radii, 0.5 * R, R
    ); 
    jkr_data.Ry = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_curvature_radii, 0.5 * R, R
    );
    jkr_data.overlaps = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_overlap, 0, 2 * (R - Rcell)
    );
    jkr_data.gamma = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_gamma, 0, jkr_data.max_gamma
    );
    jkr_data.forces = calculateJKRForceTable<T>(
        jkr_data.Rx, jkr_data.Ry, jkr_data.overlaps, jkr_data.gamma, 
        E0, max_overlap, min_aspect_ratio, newton_tol,
        newton_max_iter, imag_tol, aberth_tol
    );
    
    // Write the principal radii of curvature and JKR forces to file
    std::stringstream ss; 
    ss << outprefix << "_curvature.txt"; 
    std::ofstream outfile1(ss.str());
    outfile1 << std::setprecision(10);  
    for (int i = 0; i < n_mesh_theta; ++i)
    {
        for (int j = 0; j < n_mesh_half_l; ++j)
        {
            for (int k = 0; k < n_mesh_centerline_coords; ++k)
            {
                auto tuple = std::make_tuple(i, j, k);
                auto result = jkr_data.curvature_radii[tuple]; 
                T Rx = result.first; 
                T Ry = result.second;  
                outfile1 << jkr_data.theta(i) << '\t'
                         << jkr_data.half_l(j) << '\t'
                         << jkr_data.centerline_coords(k) << '\t'
                         << Rx << '\t' << Ry << std::endl; 
            }
        }
    }
    ss.str(std::string()); 
    ss.clear(); 
    ss << outprefix << "_forces.txt"; 
    std::ofstream outfile2(ss.str());
    outfile2 << std::setprecision(10);  
    for (int i = 0; i < n_mesh_curvature_radii; ++i)
    {
        for (int j = 0; j < n_mesh_curvature_radii; ++j)
        {
            for (int k = 0; k < n_mesh_overlap; ++k)
            {
                for (int m = 0; m < n_mesh_gamma; ++m)
                {
                    auto tuple = std::make_tuple(i, j, k, m);
                    auto result = jkr_data.forces[tuple]; 
                    T force = result.first; 
                    T radius = result.second;  
                    outfile2 << jkr_data.Rx(i) << '\t'
                             << jkr_data.Ry(j) << '\t'
                             << jkr_data.overlaps(k) << '\t'
                             << jkr_data.gamma(m) << '\t'
                             << force << '\t' << radius << std::endl; 
                } 
            }
        }
    } 

    return 0; 
}
