/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     9/20/2026
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "../include/linearStability.hpp"
#include "../include/distances.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

typedef double T;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Segment_3 Segment_3;

using std::fmod; 

int main(int argc, char** argv)
{
    K kernel; 

    // Parse input json file 
    std::string json_filename = argv[1]; 
    boost::json::object json_data = parseConfigFile(json_filename).as_object(); 

    // Define input parameters
    const T R = static_cast<T>(json_data["R"].as_double()); 
    const T Rcell = static_cast<T>(json_data["Rcell"].as_double());
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T sigma0 = static_cast<T>(json_data["sigma0"].as_double()); 
    const T eta0 = static_cast<T>(json_data["eta_ambient"].as_double());
    const T eta1 = static_cast<T>(json_data["eta_surface"].as_double());
    const T length = static_cast<T>(json_data["length"].as_double()); 
    const T overlap = static_cast<T>(json_data["overlap"].as_double());
    const T gamma = static_cast<T>(json_data["gamma"].as_double()); 
    const T min_overlap = static_cast<T>(json_data["min_overlap"].as_double()); 
    const T max_overlap = static_cast<T>(json_data["max_overlap"].as_double()); 

    // Identify the critical overlap ... 
    //
    // First, get the stability eigenvalues at the maximum overlap
    const T brent_tol = 1e-8;
    const T brent_max_iter = 1000;
    Matrix<T, Dynamic, 6> coords = getEightNeighborConfiguration<T>(
        max_overlap, 0.0, length, R, Rcell, E0, sigma0
    );
    Matrix<T, Dynamic, 2> neighbors_rxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(0, 2)); 
    Matrix<T, Dynamic, 2> neighbors_nxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(3, 2));
    Matrix<T, 2, 1> max_overlap_eigenvalues = get2DStability<T>(
        neighbors_rxy, neighbors_nxy, length, R, Rcell, E0, Ecell, sigma0,
        eta0, eta1, gamma
    ); 
    
    // Is the maximum eigenvalue greater than zero?
    T overlap_crit; 
    if (max_overlap_eigenvalues.maxCoeff() > 0)
    {
        // Then calculate the critical overlap 
        overlap_crit = findCriticalOverlapEightNeighbor<T>(
            length, R, Rcell, E0, Ecell, sigma0, eta0, eta1, gamma, min_overlap,
            max_overlap, brent_tol, brent_max_iter, false
        );
    }
    else    // Otherwise, set to NaN
    {
        overlap_crit = std::numeric_limits<T>::quiet_NaN(); 
    }

    // Configure simulation parameters
    const T init_theta = static_cast<T>(json_data["init_theta_deg"].as_double()); 
    const T t_max = static_cast<T>(json_data["t_max"].as_double()); 
    const T t_write = static_cast<T>(json_data["t_write"].as_double()); 
    T t_next_write = t_write;  
    const T r_tol = static_cast<T>(json_data["dormand_prince_r_tol"].as_double()); 
    const T n_tol = static_cast<T>(json_data["dormand_prince_n_tol"].as_double()); 
    const T min_stepsize = static_cast<T>(json_data["min_stepsize"].as_double()); 
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double()); 
    const int max_tries = json_data["dormand_prince_adapt_stepsize_max_tries"].as_int64();
    JKRCellBodyContactMode mode; 
    try
    {
        mode = static_cast<JKRCellBodyContactMode>(
            json_data["jkr_cell_body_contact_mode"].as_int64()
        );
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        mode = JKRCellBodyContactMode::IgnoreCellBody; 
    } 
    bool ignore_neighbor_interactions;
    try
    {
        ignore_neighbor_interactions = json_data["ignore_neighbor_interactions"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        ignore_neighbor_interactions = true;
    } 

    // Initialize the eight-neighbor configuration 
    Matrix<T, Dynamic, 6> coords_tilted = getEightNeighborConfiguration<T>(
        overlap, init_theta * boost::math::constants::pi<T>() / 180, length, R,
        Rcell, E0, sigma0
    );
    Matrix<T, Dynamic, 3> r = coords_tilted(Eigen::all, Eigen::seqN(0, 3)); 
    Matrix<T, Dynamic, 3> n = coords_tilted(Eigen::all, Eigen::seqN(3, 3));

    // Initialize piston positions and orientations 
    const int n_cells = r.rows();  
    Matrix<T, Dynamic, 3> r_pistons(n_cells - 1, 3), n_pistons(n_cells - 1, 3);  
    for (int i = 0; i < n_cells - 1; ++i)
    {
        Matrix<T, 3, 1> p = r.row(i + 1) - (length / 2) * n.row(i + 1); 
        Matrix<T, 3, 1> q = r.row(i + 1) + (length / 2) * n.row(i + 1);
        if (p.norm() > q.norm())    // Cell orientation vector points inward
        { 
            r_pistons.row(i) = p.transpose() - R * n.row(i + 1);
            n_pistons.row(i) = n.row(i + 1); 
        } 
        else                        // Cell orientation vector points outward
        { 
            r_pistons.row(i) = q.transpose() + R * n.row(i + 1); 
            n_pistons.row(i) = -n.row(i + 1);
        } 
    }
    const T force_ext_prefactor = (4. / 3.) * E0 * sqrt(R);
    T piston_travel_dist;
    try
    {
        piston_travel_dist = json_data["piston_travel_dist"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        piston_travel_dist = 0.5 * R;
    } 
    const T piston_velocity = piston_travel_dist / t_max;

    // Specify constraints 
    Matrix<T, Dynamic, Dynamic> constraints = getEightNeighborConfigurationConstraints<T>();

    // Open output file and write header  
    std::ofstream outfile(argv[2]);
    outfile << std::setprecision(10);  
    outfile << "# length = " << length << std::endl
            << "# overlap = " << overlap << std::endl
            << "# R = " << R << std::endl
            << "# Rcell = " << Rcell << std::endl
            << "# E0 = " << E0 << std::endl
            << "# Ecell = " << Ecell << std::endl 
            << "# sigma0 = " << sigma0 << std::endl
            << "# eta_ambient = " << eta0 << std::endl
            << "# eta_surface = " << eta1 << std::endl
            << "# gamma = " << gamma << std::endl
            << "# min_overlap = " << min_overlap << std::endl
            << "# max_overlap = " << max_overlap << std::endl
            << "# brent_tol = " << brent_tol << std::endl
            << "# brent_max_iter = " << brent_max_iter << std::endl
            << "# overlap_crit = " << overlap_crit << std::endl 
            << "# init_theta_deg = " << init_theta << std::endl
            << "# piston_travel_dist = " << piston_travel_dist << std::endl
            << "# piston_velocity = " << piston_velocity << std::endl
            << "# t_max = " << t_max << std::endl
            << "# t_write = " << t_write << std::endl
            << "# dormand_prince_r_tol = " << r_tol << std::endl
            << "# dormand_prince_n_tol = " << n_tol << std::endl
            << "# min_stepsize = " << min_stepsize << std::endl
            << "# max_stepsize = " << max_stepsize << std::endl
            << "# dormand_prince_adapt_stepsize_max_tries = " << max_tries << std::endl
            << "# jkr_cell_body_contact_mode = " << static_cast<int>(mode) << std::endl
            << "# ignore_neighbor_interactions = " << ignore_neighbor_interactions << std::endl; 

    // Specify stepsize control parameters
    int iter = 0;
    T t_prev = 0; 
    T t_curr = 0;
    T stepsize = max_stepsize;
    while (t_curr < t_max)
    {
        // Take a Dormand-Prince step, update time, and update stepsize  
        auto result = getDormandPrinceUpdateWithAdaptedStepsize<T>(
            r, n, constraints, length, R, Rcell, E0, Ecell, sigma0, eta0, eta1,
            gamma, r_pistons, n_pistons, force_ext_prefactor, piston_velocity,
            stepsize, r_tol, n_tol, min_stepsize, max_stepsize, max_tries,
            mode, ignore_neighbor_interactions
        );
        Matrix<T, Dynamic, 6> update = std::get<0>(result);
        T curr_stepsize = std::get<1>(result);
        t_prev = t_curr; 
        t_curr += curr_stepsize; 
        stepsize = std::get<2>(result);
        iter++;  
        r += update(Eigen::all, Eigen::seqN(0, 3));
        n += update(Eigen::all, Eigen::seqN(3, 3));

        // Re-normalize all orientation vectors 
        for (int k = 0; k < n.rows(); ++k)
        {
            T norm = n.row(k).norm(); 
            n.row(k) /= norm; 
        }

        // Move the pistons inward
        T dr_piston = piston_velocity * curr_stepsize;  
        for (int i = 0; i < n_cells - 1; ++i)
            r_pistons.row(i) += dr_piston * n_pistons.row(i);

        // Intermittently write cell coordinates to output file 
        if (t_prev < t_next_write && t_curr >= t_next_write)
        {
            outfile << t_curr << '\t';
            for (int k = 0; k < r.rows(); ++k)
            {
                for (int m = 0; m < 3; ++m)
                    outfile << r(k, m) << '\t';
                for (int m = 0; m < 3; ++m)
                    outfile << n(k, m) << '\t'; 
            }
            outfile.seekp(-1, std::ios_base::cur); 
            outfile << std::endl;
            t_next_write += t_write;

            // Also print a couple of lines to stdout 
            std::cout << "Iteration " << iter << ", t = " << t_curr << ", "
                      << "current stepsize = " << curr_stepsize << ", "
                      << "next trial stepsize = " << stepsize << std::endl;
            std::cout << "- Central cell coordinates = ("
                      << r(0, 0) << ", " << r(0, 1) << ", " << r(0, 2) << "; "
                      << n(0, 0) << ", " << n(0, 1) << ", " << n(0, 2) << "), "
                      << "minimum z-coordinate = " 
                      << (n(0, 2) > 0 ? r(0, 2) - (length / 2) * n(0, 2) : r(0, 2) + (length / 2) * n(0, 2))
                      << std::endl; 

            // Look for any neighboring cells that have detached from the 
            // central cell
            int n_detached = 0;  
            for (int i = 1; i < n_cells; ++i)
            {
                Segment_3 seg1 = generateSegment<T>(r.row(0), n.row(0), length / 2); 
                Segment_3 seg2 = generateSegment<T>(r.row(i), n.row(i), length / 2);
                auto result2 = distBetweenCells<T>(
                    seg1, seg2, 0, r.row(0), n.row(0), length / 2,
                    1, r.row(i), n.row(i), length / 2, kernel
                );
                T dist = std::get<0>(result2).norm(); 
                if (dist > 2 * R)
                {
                    if (n_detached == 0)
                        std::cout << "- Neighbors detached from central cell:\n";
                    std::cout << "  - Cell " << i << ": (" << r(i, 0) << ", "
                              << r(i, 1) << ", " << r(i, 2) << "; " 
                              << n(i, 0) << ", " << n(i, 1) << ", "
                              << n(i, 2) << "), dist = " << dist << std::endl;
                    n_detached++; 
                } 
            }
        }
    }
    outfile.close(); 

    return 0;
}
