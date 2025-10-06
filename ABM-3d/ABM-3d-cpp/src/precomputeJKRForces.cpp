/**
 * Pre-compute tables of values for the JKR force magnitude and contact
 * radius, according to the model proposed by Giudici et al. J. Phys. D
 * (2025), for different cell-cell configurations.   
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/6/2025
 */

#include <Eigen/Dense>
#include "../include/simulation.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Define floating-point type to be used 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T;

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
    
    // Parse desired equilibrium cell-cell distance 
    T eqdist = static_cast<T>(json_data["adhesion_eqdist"].as_double());

    // Parse optional input parameters for anisotropic JKR adhesion
    int n_mesh_theta = 100; 
    int n_mesh_half_l = 100; 
    int n_mesh_centerline_coords = 100;
    int n_mesh_max_curvature_radii = 100;
    int n_mesh_phi = 100; 
    int n_mesh_overlap = 100;
    bool calibrate_endpoint_radii = true;
    T min_aspect_ratio = 0.000001;
    T max_aspect_ratio = 0.999999;  
    T project_tol = 1e-8; 
    int project_max_iter = 100; 
    T brent_tol = 1e-8; 
    int brent_max_iter = 1000;
    T init_bracket_dx = 1e-3; 
    int n_tries_bracket = 5;
    T imag_tol = 1e-20;  
    T aberth_tol = 1e-20; 
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
        n_mesh_max_curvature_radii = json_data["adhesion_n_mesh_max_curvature_radii"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_phi = json_data["adhesion_n_mesh_phi"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_mesh_overlap = json_data["adhesion_n_mesh_overlap"].as_int64(); 
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
        brent_tol = static_cast<T>(json_data["adhesion_brent_tol"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        brent_max_iter = json_data["adhesion_brent_max_iter"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        init_bracket_dx = static_cast<T>(json_data["adhesion_init_bracket_dx"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        n_tries_bracket = json_data["adhesion_n_tries_bracket"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
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

    // Restrict cell-cell angle to specified range, if desired 
    T min_phi = 0; 
    T max_phi = boost::math::constants::half_pi<T>();
    Matrix<T, Dynamic, 1> phi;  
    try
    {
        min_phi = static_cast<T>(json_data["adhesion_min_phi"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        max_phi = static_cast<T>(json_data["adhesion_max_phi"].as_double()); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    if (abs(min_phi - max_phi) < 1e-8 && n_mesh_phi == 1)
    {
        phi.resize(1); 
        phi << min_phi; 
    }
    else if (abs(min_phi - max_phi) > 1e-8 && n_mesh_phi > 1) 
    {
        phi = Matrix<T, Dynamic, 1>::LinSpaced(n_mesh_phi, min_phi, max_phi); 
    }
    else
    {
        throw std::runtime_error(
            "Cannot generate multi-valued range for phi between specified "
            "minimum/maximum values"
        ); 
    }
    
    // Output file prefix
    std::string outfilename = argv[2]; 

    // Compute anisotropic JKR forces ...
    //
    // Solve for the surface energy density corresponding to the desired 
    // equilibrium distance  
    const T gamma = jkrOptimalSurfaceEnergyDensity<T, 100>(
        R, Rcell, E0, eqdist, 100.0, 1e-6, 1e-8, 1e-8, 1e-8, 1000, 1000,
        imag_tol, aberth_tol, true
    );

    // Calculate principal radii of curvature 
    Matrix<T, Dynamic, 1> theta = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_theta, 0.0, boost::math::constants::half_pi<T>()
    ); 
    Matrix<T, Dynamic, 1> half_l = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_half_l, 0.5 * L0, 0.5 * Ldiv
    );  
    Matrix<T, Dynamic, 1> centerline_coords = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_centerline_coords, 0.0, 1.0
    );
    R3ToR2Table<T> curvature_radii = calculateCurvatureRadiiTable<T>(
        theta, half_l, centerline_coords, R, calibrate_endpoint_radii,
        project_tol, project_max_iter
    );

    // Identify min/max values for principal radii of curvature
    T min_Rx = std::numeric_limits<T>::infinity();
    T max_Rx = 0;
    T min_Ry = std::numeric_limits<T>::infinity();
    T max_Ry = 0; 
    for (int i = 0; i < n_mesh_theta; ++i)
    {
        for (int j = 0; j < n_mesh_half_l; ++j)
        {
            for (int k = 0; k < n_mesh_centerline_coords; ++k)
            {
                auto tuple = std::make_tuple(i, j, k); 
                T Rx_ = curvature_radii[tuple].first;
                T Ry_ = curvature_radii[tuple].second; 
                if (Rx_ < min_Rx)
                    min_Rx = Rx_; 
                if (Rx_ > max_Rx)
                    max_Rx = Rx_; 
                if (Ry_ < min_Ry)
                    min_Ry = Ry_;  
                if (Ry_ > max_Ry)
                    max_Ry = Ry_;  
            }
        }
    }
    std::cout << "Maximum principal radii of curvature: [" << min_Rx << ", " << max_Rx << "]" << std::endl;
    std::cout << "Minimum principal radii of curvature: [" << min_Ry << ", " << max_Ry << "]" << std::endl; 

    // Calculate JKR forces for a range of cell-cell configurations 
    const T max_overlap = 2 * (R - Rcell);
    Matrix<T, Dynamic, 1> Rx = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_max_curvature_radii, min_Rx, max_Rx
    );
    Matrix<T, Dynamic, 1> overlaps = Matrix<T, Dynamic, 1>::LinSpaced(
        n_mesh_overlap, 0, max_overlap
    );
    R4ToR2Table<T> forces = calculateJKRForceTable<T>(
        Rx, phi, overlaps, gamma, R, E0, max_overlap, min_aspect_ratio,
        max_aspect_ratio, brent_tol, brent_max_iter, init_bracket_dx,
        n_tries_bracket, imag_tol, aberth_tol
    ); 

    // Write the JKR forces to file 
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(10);  
    for (int i = 0; i < n_mesh_max_curvature_radii; ++i)
    {
        for (int j = i; j < n_mesh_max_curvature_radii; ++j)
        {
            for (int k = 0; k < n_mesh_phi; ++k)
            {
                for (int m = 0; m < n_mesh_overlap; ++m)
                {
                    auto tuple = std::make_tuple(i, j, k, m);
                    auto result = forces[tuple]; 
                    T force = result.first; 
                    T radius = result.second;  
                    outfile << Rx(i) << '\t'
                            << Rx(j) << '\t'
                            << phi(k) << '\t'
                            << overlaps(m) << '\t'
                            << force << '\t' << radius << std::endl; 
                } 
            }
        }
    }
    outfile.close(); 

    return 0; 
}
