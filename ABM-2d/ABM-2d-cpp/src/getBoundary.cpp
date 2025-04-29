/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/28/2025
 */

#include <iostream>
#include <string> 
#include <filesystem>
#include <algorithm>
#include <Eigen/Dense>
#include "../include/confinement.hpp"
#include "../include/utils.hpp"

typedef double T;

int main(int argc, char** argv)
{
    std::string dir = argv[1];
    std::string shape_dir = std::filesystem::path(dir) / "boundaries";

    // For each file in the input directory ... 
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        std::string filename = entry.path();
        if (filename.size() >= 4 && filename.compare(filename.size() - 4, filename.size(), ".txt") == 0)
        {
            // Skip over the lineage file 
            if (filename.compare(filename.size() - 12, filename.size(), "_lineage.txt") == 0)
                continue; 

            // Define the output filename
            std::string basename = std::filesystem::path(filename).stem();  
            std::stringstream ss;
            ss << basename << "_boundary.txt";
            std::string outfilename = std::filesystem::path(shape_dir) / ss.str();  

            // Parse the data file 
            auto result = readCells<T>(filename); 
            Array<T, Dynamic, Dynamic> cells = result.first;
            std::map<std::string, std::string> params = result.second;
            const T R = static_cast<T>(std::stod(params["R"]));

            // Get the boundary cells
            if (cells.rows() >= 50)
            { 
                std::vector<int> boundary = getBoundary<T>(cells, R, 50); 

                // Output the boundary cell indices
                std::ofstream outfile(outfilename); 
                for (int i = 0; i < boundary.size(); ++i)
                    outfile << boundary[i] << std::endl; 
            } 
        }
    }

    return 0; 
}
