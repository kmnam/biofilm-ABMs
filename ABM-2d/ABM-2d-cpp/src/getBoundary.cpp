/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/28/2025
 */

#include <iostream>
#include <string>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <Eigen/Dense>
#include "../include/confinement.hpp"
#include "../include/boundaries.hpp"
#include "../include/indices.hpp"
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
            if (cells.rows() >= 20)
            {
                std::cout << "Computing boundary: " << filename << std::endl;  
                auto boundary = getBoundaryFromOutlines<T>(cells, R, 50);
                AlphaShape2DProperties shape = boundary.first; 
                std::vector<int> idx = boundary.second;            
                std::cout << "... found simply connected boundary: "
                          << shape.is_simple_cycle << std::endl;

                // Get the boundary cell indices
                std::unordered_set<int> boundary_cells; 
                for (const int& v : shape.vertices)
                    boundary_cells.insert(idx[v]);
                std::vector<int> boundary_cells_sorted(boundary_cells.begin(), boundary_cells.end()); 
                std::sort(boundary_cells_sorted.begin(), boundary_cells_sorted.end()); 

                // Output the boundary cell indices
                std::ofstream outfile(outfilename);
                outfile << std::setprecision(10);  
                for (const int& i : boundary_cells_sorted)
                    outfile << "BOUNDARY_CELL\t" << i << '\t'
                            << cells(i, __colidx_id) << '\t'
                            << cells(i, __colidx_rx) << '\t'
                            << cells(i, __colidx_ry) << '\t'
                            << cells(i, __colidx_nx) << '\t'
                            << cells(i, __colidx_ny) << '\t'
                            << cells(i, __colidx_l) << std::endl; 
                
                // Output each boundary point and edge 
                for (const int& v : shape.vertices)
                    outfile << "BOUNDARY_VERTEX\t"
                            << v << '\t' << shape.x[v] << '\t' << shape.y[v] << std::endl;
                for (const std::pair<int, int>& e : shape.edges)
                    outfile << "BOUNDARY_EDGE\t"
                            << e.first << '\t' << e.second << std::endl;  
            } 
        }
    }

    return 0; 
}
