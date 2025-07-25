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
 *     7/21/2025
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
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS> Graph; 

int main(int argc, char** argv)
{
    std::string dir = argv[1];
    std::string graph_dir = (argc > 2 ? argv[2] : std::filesystem::path(dir) / "graphs");

    // Check if triangles and tetrahedra are to be identified 
    bool get_triangles = false; 
    bool get_tetrahedra = false; 
    if (argc > 2)
    {
        std::string arg = argv[2];
        if (arg.compare(0, arg.size(), "-3") == 0)
        {
            get_triangles = true; 
        }
        else if (arg.compare(0, arg.size(), "-4") == 0) 
        {
            get_triangles = true; 
            get_tetrahedra = true; 
        }
        else 
        {
            throw std::invalid_argument("Invalid input argument specified"); 
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
            std::vector<int> components = getConnectedComponents<T>(graph);
            Array<int, Dynamic, 1> degrees = getDegrees<T>(graph);

            // Get the clustering coefficients 
            Array<T, Dynamic, 1> cluster_coefs = getLocalClusteringCoefficients<T>(graph);  

            // Get the triangles and/or tetrahedra in the graph, if desired
            Array<int, Dynamic, 3> triangles(0, 3);
            Array<int, Dynamic, 4> tetrahedra(0, 4);  
            if (get_triangles)
                triangles = getTriangles<T>(graph);
            if (get_tetrahedra)
                tetrahedra = getTetrahedra<T>(graph); 

            // Output the graph 
            writeGraph<T>(
                graph, components, degrees, outfilename, true, cluster_coefs,
                get_triangles, triangles, get_tetrahedra, tetrahedra
            ); 
        }
    }

    return 0; 
}
