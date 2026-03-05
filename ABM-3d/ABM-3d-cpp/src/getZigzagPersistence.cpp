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

int main(int argc, char** argv)
{
    // Parse input arguments
    std::vector<std::string> cells_filenames, complex_filenames;
    int nframes = 0;        // Impose no maximum number of frames by default
    double tmin = 2.0; 
    double tmax = std::numeric_limits<double>::max(); 
    bool verbose = false; 

    // Parse additional arguments
    if (argc > 4)
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

    // Compute zigzag persistence
    std::vector<Bar> zigzag = computeZigzagPersistence<double>(
        cells_filenames, complex_filenames, verbose
    );  

    return 0;  
}
