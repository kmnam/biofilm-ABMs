/**
 * Pre-compute a table of values for the principal radii of curvature along
 * along the spherocylindrical surface of a cell.   
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/8/2025
 */

#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
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
    const T L0 = static_cast<T>(json_data["L0"].as_double());
    const T Ldiv = 2 * L0 + 2 * R;
    
    // Parse cell-cell adhesion parameters ... 
    T eqdist = static_cast<T>(json_data["adhesion_eqdist"].as_double());

    // Parse optional input parameters for anisotropic JKR adhesion
    int n_mesh_theta = 100; 
    int n_mesh_half_l = 100; 
    int n_mesh_centerline_coords = 100;
    bool calibrate_endpoint_radii = true;
    T project_tol = 1e-8; 
    int project_max_iter = 100; 
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

    // Output file prefix
    std::string outfilename = argv[2]; 

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
    TupleToTupleTable<T, 3, 2> curvature_radii = calculateCurvatureRadiiTable<T>(
        theta, half_l, centerline_coords, R, calibrate_endpoint_radii,
        project_tol, project_max_iter
    );

    // Write the principal radii of curvature to file
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(10);  
    for (int i = 0; i < n_mesh_theta; ++i)
    {
        for (int j = 0; j < n_mesh_half_l; ++j)
        {
            for (int k = 0; k < n_mesh_centerline_coords; ++k)
            {
                std::array<int, 3> key = {i, j, k}; 
                auto result = curvature_radii[key];
                T Rx = result[0];
                T Ry = result[1];
                outfile << theta(i) << '\t' << half_l(j) << '\t'
                        << centerline_coords(k) << '\t'
                        << Rx << '\t' << Ry << std::endl; 
            }
        }
    }
    outfile.close(); 

    return 0; 
}
