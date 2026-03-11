/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/11/2026
 */

#include <iostream>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/distances.hpp"
#include "../include/utils.hpp"
#include "../include/fields.hpp"
#include "../include/topology.hpp"

using std::abs;
using std::modf; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

enum class InputMode
{
    FILENAME_MODE,
    DIRECTORY_MODE,
    TIMEPOINT_MODE,
    SIZE_MODE
};

int main(int argc, char** argv)
{
    K kernel;

    // Parse input arguments
    std::vector<std::string> cells_filenames, complex_filenames;
    std::string outdir; 

    InputMode mode = InputMode::FILENAME_MODE;
    if (std::string(argv[1]) == "-d")
        mode = InputMode::DIRECTORY_MODE; 
    else if (std::string(argv[1]) == "-t")
        mode = InputMode::TIMEPOINT_MODE;
    else if (std::string(argv[1]) == "-s")
        mode = InputMode::SIZE_MODE;  
    int nframes = 20;  
    double tmin = 2.0; 
    double tmax = std::numeric_limits<double>::max(); 
    bool skip_minimize = false;
    bool verbose = false;

    // Get the homology generators for frames across a simulation  
    if (mode == InputMode::DIRECTORY_MODE)
    {
        if (argc > 5)   // Parse additional arguments 
        {
            std::vector<std::string> args; 
            for (int i = 5; i < argc; ++i)
                args.push_back(argv[i]);

            // --skip-min: Skip cycle minimization 
            if (std::find(args.begin(), args.end(), "--skip-min") != args.end())
                skip_minimize = true;

            // -n: Number of frames 
            auto it = std::find(args.begin(), args.end(), "-n"); 
            if (it != args.end())
            {
                auto next = std::next(it);
                if (next == args.end())
                    throw std::runtime_error("Invalid format for input option -n"); 
                nframes = std::stoi(*next); 
            }

            // --tmin: Minimum time 
            it = std::find(args.begin(), args.end(), "--tmin"); 
            if (it != args.end())
            {
                auto next = std::next(it); 
                if (next == args.end())
                    throw std::runtime_error("Invalid format for input option --tmin"); 
                tmin = std::stod(*next); 
            } 

            // --tmax: Maximum time 
            it = std::find(args.begin(), args.end(), "--tmax"); 
            if (it != args.end())
            {
                auto next = std::next(it); 
                if (next == args.end())
                    throw std::runtime_error("Invalid format for input option --tmax"); 
                tmax = std::stod(*next); 
            }

            // --verbose: Verbosity 
            if (std::find(args.begin(), args.end(), "--verbose") != args.end())
                verbose = true;  
        }

        // Gather the filenames in the given directories 
        std::string cells_dir = argv[2]; 
        std::string complex_dir = argv[3];
        outdir = argv[4]; 
        cells_filenames = parseDir(cells_dir, nframes, tmin, tmax).first;
        for (const std::string& filename : cells_filenames)
        {
            std::string basename = std::filesystem::path(filename).stem();  
            std::stringstream ss; 
            ss << basename << "_graph.txt";
            std::string complex_filename = std::filesystem::path(complex_dir) / ss.str(); 
            complex_filenames.push_back(complex_filename);  
        }
    }
    // Or get the homology generators for a single simulation frame that is
    // closest to the given timepoint 
    else if (mode == InputMode::TIMEPOINT_MODE)
    {
        if (argc > 6)   // Parse additional arguments
        {
            std::vector<std::string> args; 
            for (int i = 6; i < argc; ++i)
                args.push_back(argv[i]);

            // --skip-min: Skip cycle minimization 
            if (std::find(args.begin(), args.end(), "--skip-min") != args.end())
                skip_minimize = true;

            // --verbose: Verbosity 
            if (std::find(args.begin(), args.end(), "--verbose") != args.end())
                verbose = true;  
        }

        // Get the simulation frame closest to the given timepoint
        std::string cells_dir = argv[2]; 
        std::string complex_dir = argv[3];
        outdir = argv[4];  
        double t_query = std::stod(argv[5]);
        std::string cells_filename = findFileNearestToTimepoint(cells_dir, t_query);
        cells_filenames.push_back(cells_filename); 

        // Construct the corresponding complex filename  
        std::string basename = std::filesystem::path(cells_filename).stem(); 
        std::stringstream ss; 
        ss << basename << "_graph.txt"; 
        std::string complex_filename = std::filesystem::path(complex_dir) / ss.str(); 
        complex_filenames.push_back(complex_filename); 
    }
    // Or get the homology generators for a single simulation frame that is
    // closest to the given population size 
    else if (mode == InputMode::SIZE_MODE)
    {
        if (argc > 6)   // Parse additional arguments
        {
            std::vector<std::string> args; 
            for (int i = 6; i < argc; ++i)
                args.push_back(argv[i]);

            // --skip-min: Skip cycle minimization 
            if (std::find(args.begin(), args.end(), "--skip-min") != args.end())
                skip_minimize = true;

            // --verbose: Verbosity 
            if (std::find(args.begin(), args.end(), "--verbose") != args.end())
                verbose = true;  
        }

        // Get the simulation frame closest to the given timepoint
        std::string cells_dir = argv[2]; 
        std::string complex_dir = argv[3]; 
        outdir = argv[4];  
        int size = std::stoi(argv[5]);
        std::string cells_filename = findFileNearestToSize(cells_dir, size);  
        cells_filenames.push_back(cells_filename); 

        // Construct the corresponding complex filename  
        std::string basename = std::filesystem::path(cells_filename).stem(); 
        std::stringstream ss; 
        ss << basename << "_graph.txt"; 
        std::string complex_filename = std::filesystem::path(complex_dir) / ss.str(); 
        complex_filenames.push_back(complex_filename); 
    }
    else    // Or get the homology generators for the specified simulation frame 
    {
        if (argc > 4)   // Parse additional arguments
        {
            std::vector<std::string> args; 
            for (int i = 4; i < argc; ++i)
                args.push_back(argv[i]);

            // --skip-min: Skip cycle minimization 
            if (std::find(args.begin(), args.end(), "--skip-min") != args.end())
                skip_minimize = true;

            // --verbose: Verbosity 
            if (std::find(args.begin(), args.end(), "--verbose") != args.end())
                verbose = true;  
        }
        cells_filenames.push_back(argv[1]); 
        complex_filenames.push_back(argv[2]);
        outdir = argv[3]; 
    } 

    // Parse the cells and the simplicial complex at each frame ... 
    nframes = cells_filenames.size(); 
    for (int i = 0 ; i < nframes; ++i)
    {
        std::string cells_filename = cells_filenames[i]; 
        std::string complex_filename = complex_filenames[i];
        std::cout << "Parsing frame #" << i << ":\n"
                  << "- Frame filename: " << cells_filename << std::endl
                  << "- Complex filename: " << complex_filename << std::endl;

        // Determine output filename 
        std::string basename = std::filesystem::path(complex_filename).stem();
        std::stringstream ss;
        ss << basename << "_cycles.txt";  
        std::string outfilename = std::filesystem::path(outdir) / ss.str();
        std::cout << "- Output filename: " << outfilename << std::endl; 

        // Parse simulation frame and simplicial complex 
        auto result = readCells<double>(cells_filename); 
        Array<double, Dynamic, Dynamic> cells = result.first;
        SimplicialComplex3D<double> cplex;
        Array<double, Dynamic, 3> coords(cells.rows(), 3);
        coords.col(0) = cells.col(__colidx_rx); 
        coords.col(1) = cells.col(__colidx_ry); 
        coords.col(2) = cells.col(__colidx_rz);
        cplex.read(complex_filename, coords);

        // Get subcomplex induced by the group 1 cells 
        std::vector<int> in_group1; 
        for (int j = 0; j < cells.rows(); ++j)
        {
            if (cells(j, __colidx_group) == 1)
                in_group1.push_back(j); 
        }
        SimplicialComplex3D<double> subcomplex = cplex.getSubcomplex(in_group1);

        // Get a minimal first homology basis for the subcomplex 
        Matrix<Z2, Dynamic, Dynamic> min_basis = subcomplex.getMinimalFirstHomology(verbose);

        // Compute cell-cell distances for each edge in the subcomplex 
        Matrix<int, Dynamic, 2> edges = subcomplex.getSimplices<1>();
        Matrix<double, Dynamic, 1> distances(edges.rows());  
        for (int j = 0; j < edges.rows(); ++j)
        {
            int u = in_group1[edges(j, 0)]; 
            int v = in_group1[edges(j, 1)];
            Matrix<double, 3, 1> r1 = cells(u, __colseq_r); 
            Matrix<double, 3, 1> n1 = cells(u, __colseq_n); 
            Matrix<double, 3, 1> r2 = cells(v, __colseq_r); 
            Matrix<double, 3, 1> n2 = cells(v, __colseq_n);
            double half_l1 = cells(u, __colidx_half_l); 
            double half_l2 = cells(v, __colidx_half_l); 
            Segment_3 seg1 = generateSegment<double>(r1, n1, half_l1);
            Segment_3 seg2 = generateSegment<double>(r2, n2, half_l2);  
            auto result = distBetweenCells<double>(
                seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel
            );
            distances(j) = std::get<0>(result).norm();  
        }

        // If desired, minimize each cycle in the basis
        const int ne = min_basis.rows(); 
        const int ncycles = min_basis.cols(); 
        Matrix<Z2, Dynamic, Dynamic> final_cycles(ne, ncycles); 
        if (!skip_minimize)
        {
            Matrix<double, Dynamic, Dynamic> opt_basis = subcomplex.minimizeCycles(
                min_basis, 1, CycleMinimizeMode::MINIMIZE_CYCLE_SIZE, distances,
                verbose
            );

            // For each minimized cycle, store the edges for which the coefficient
            // is odd 
            const double tol = 1e-5;
            for (int j = 0; j < ncycles; ++j)
            {
                Matrix<Z2, Dynamic, 1> min_cycle = min_basis.col(j); 
                Matrix<Z2, Dynamic, 1> opt_cycle = Matrix<Z2, Dynamic, 1>::Zero(ne); 
                for (int k = 0; k < ne; ++k)
                {
                    // Check if the coefficient is close to an integer 
                    double intpart; 
                    double frac = abs(modf(opt_basis(k, j), &intpart));
                    if (frac > tol || frac < 1 - tol) 
                    {
                        // Check if the coefficient is odd
                        if (static_cast<int>(abs(intpart)) % 2 == 1)
                        {
                            // If so, this edge is part of the cycle 
                            opt_cycle(k) = 1; 
                        }
                    }
                }

                // Store the minimized cycle only if its weight is less than 
                // the initial cycle (this may not be the case, since the 
                // minimization was done over real coefficients)
                double weight1 = 0;
                double weight2 = 0; 
                for (int k = 0; k < ne; ++k)
                {
                    if (opt_cycle(k) == 1)
                        weight1 += distances(k); 
                    if (min_cycle(k) == 1)
                        weight2 += distances(k);  
                }
                if (weight1 < weight2)
                    final_cycles.col(j) = opt_cycle; 
                else 
                    final_cycles.col(j) = min_cycle;
            }
        }
        else    // Otherwise, simply use the minimal homology basis 
        {
            final_cycles = min_basis; 
        }

        // Write the cycles to file
        std::ofstream outfile(outfilename);
        outfile << "# cells_filename = " << cells_filename << std::endl; 
        outfile << "# complex_filename = " << complex_filename << std::endl;
        Matrix<double, Dynamic, 3> points = subcomplex.getPoints().matrix();  
        for (int j = 0; j < ncycles; ++j)
        {
            std::stringstream ss_line; 
            for (int k = 0; k < ne; ++k)
            {
                if (final_cycles(k, j) == 1)
                {
                    // Output each edge in terms of the cell IDs
                    int u = edges(k, 0); 
                    int v = edges(k, 1);
                    Matrix<double, 3, 1> point_u = points.row(u); 
                    Matrix<double, 3, 1> point_v = points.row(v);  

                    // Since the subcomplex of group 1 cells was extracted
                    // prior to computing the cycles, we must identify the 
                    // cell corresponding to each point in the subcomplex
                    Index u_idx, v_idx;
                    (coords.matrix().rowwise() - point_u.transpose()).rowwise().squaredNorm().minCoeff(&u_idx);
                    (coords.matrix().rowwise() - point_v.transpose()).rowwise().squaredNorm().minCoeff(&v_idx);  
                    int u_id = static_cast<int>(cells(u_idx, __colidx_id)); 
                    int v_id = static_cast<int>(cells(v_idx, __colidx_id));  
                    ss_line << u_id << "," << v_id << ";"; 
                } 
            }
            std::string line = ss_line.str(); 
            line.pop_back(); 
            outfile << line << std::endl; 
        }
    } 

    return 0;
}
