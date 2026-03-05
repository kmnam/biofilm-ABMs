/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/5/2026
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

int main(int argc, char** argv)
{
    K kernel;

    // Parse input arguments
    std::vector<std::string> cells_filenames, complex_filenames;
    int nframes = 0;       // Impose no maximum number of frames by default
    double tmin = 2.0; 
    double tmax = std::numeric_limits<double>::max(); 
    bool verbose = false;

    if (argc > 4)   // Parse additional arguments 
    {
        std::vector<std::string> args; 
        for (int i = 4; i < argc; ++i)
            args.push_back(argv[i]);

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
    std::string cells_dir = argv[1]; 
    std::string complex_dir = argv[2];
    cells_filenames = parseDir(cells_dir, nframes, tmin, tmax).first;
    for (const std::string& filename : cells_filenames)
    {
        std::string basename = std::filesystem::path(filename).stem();  
        std::stringstream ss; 
        ss << basename << "_graph.txt";
        std::string complex_filename = std::filesystem::path(complex_dir) / ss.str(); 
        complex_filenames.push_back(complex_filename);  
    }

    // Parse the cells and the simplicial complex at each frame ... 
    nframes = cells_filenames.size();
    Array<int, Dynamic, Dynamic> betti = Array<int, Dynamic, Dynamic>::Zero(nframes, 3);
    Array<double, Dynamic, 1> times = Array<double, Dynamic, 1>::Zero(nframes);  
    for (int i = 0 ; i < nframes; ++i)
    {
        std::string cells_filename = cells_filenames[i]; 
        std::string complex_filename = complex_filenames[i];
        std::cout << "Parsing frame #" << i << ":\n"
                  << "- Frame filename: " << cells_filename << std::endl
                  << "- Complex filename: " << complex_filename << std::endl;

        // Parse simulation frame and simplicial complex 
        auto result = readCells<double>(cells_filename);
        Array<double, Dynamic, Dynamic> cells = result.first;
        std::map<std::string, std::string> params = result.second; 
        times(i) = std::stod(params["t_curr"]); 
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

        // Get the Betti numbers of the subcomplex
        std::cout << subcomplex.getZ2BettiNumbers().transpose() << std::endl; 
        betti.row(i) = subcomplex.getZ2BettiNumbers().head(3);  
    }

    // Write Betti numbers to file 
    std::string outfilename = argv[3];
    std::ofstream outfile(outfilename);
    outfile << std::setprecision(10);  
    outfile << "# tmin = " << tmin << std::endl; 
    outfile << "# tmax = " << tmax << std::endl;
    for (int i = 0; i < nframes; ++i)
    {
        outfile << times(i) << '\t'
                << betti(i, 0) << '\t'
                << betti(i, 1) << '\t'
                << betti(i, 2) << std::endl; 
    } 
    outfile.close(); 
    
    return 0;
}
