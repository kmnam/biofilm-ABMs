/**
 * Given a directory of simulation frames containing cell coordinates, parse 
 * the cell coordinates in each frame and calculate the corresponding
 * simplicial complex.
 *
 * Two cells are deemed to be in contact if their centerlines are within 
 * distance R, which is given in the header of each file.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/23/2025
 */

#include <iostream>
#include <string> 
#include <filesystem>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include "../include/utils.hpp"
#include "../include/graphs.hpp"

typedef double T;

int main(int argc, char** argv)
{
    std::string dir = argv[1];

    // Parse additional arguments 
    std::string graph_dir = std::filesystem::path(dir) / "graphs";
    bool get_triangles = false; 
    bool get_tetrahedra = false; 
    if (argc > 2)
    {
        std::vector<std::string> args; 
        for (int i = 2; i < argc; ++i)
            args.push_back(argv[i]);

        if (args[0] != "-3" && args[0] != "-4")
        {
            graph_dir = args[0]; 
        }        
        if (std::find(args.begin(), args.end(), "-3") != args.end())
        {
            get_triangles = true; 
        }
        if (std::find(args.begin(), args.end(), "-4") != args.end())
        {
            get_triangles = true; 
            get_tetrahedra = true; 
        }
    }

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
            ss << basename << "_graph.txt";
            std::string outfilename = std::filesystem::path(graph_dir) / ss.str();  

            // Parse the data file 
            auto result = readCells<T>(filename); 
            Array<T, Dynamic, Dynamic> cells = result.first;
            std::map<std::string, std::string> params = result.second;
            const T R = static_cast<T>(std::stod(params["R"]));
            const T Ldiv = static_cast<T>(std::stod(params["Ldiv"])); 

            // Define the cell-cell neighbor graph and get its degree distribution
            // and connected components 
            Graph graph = getNeighborGraph<T>(cells, R, Ldiv);
            std::vector<int> components = getConnectedComponents(graph);
            Array<int, Dynamic, 1> degrees = getDegrees(graph);

            // Get the clustering coefficients 
            Array<T, Dynamic, 1> cluster_coefs = getLocalClusteringCoefficients<T>(graph);  

            // Get the triangles and/or tetrahedra in the graph, if desired
            Array<int, Dynamic, 3> triangles(0, 3);
            Array<int, Dynamic, 4> tetrahedra(0, 4);  
            if (get_triangles)
                triangles = getTriangles(graph);
            if (get_tetrahedra)
                tetrahedra = getTetrahedra(graph); 

            // Output the graph 
            writeGraph<T>(
                graph, components, degrees, outfilename, true, cluster_coefs,
                get_triangles, triangles, get_tetrahedra, tetrahedra
            ); 
        }
    }

    return 0; 
}
