/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/14/2025
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
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph; 

int main(int argc, char** argv)
{
    std::string dir = argv[1];
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        std::string filename = entry.path();
        if (filename.size() >= 4 && filename.compare(filename.size() - 4, filename.size(), ".txt") == 0)
        {
	    // Define the output filename 
	    std::string outfilename; 
	    std::stringstream ss;
	    ss << filename.substr(0, filename.size() - 4) << "_graph.txt";
	    outfilename = ss.str();  

            // Parse the data file 
            auto result = readCells<T>(filename); 
            Array<T, Dynamic, Dynamic> cells = result.first;
            std::map<std::string, std::string> params = result.second;
            const T R = static_cast<T>(std::stod(params["R"]));
            const T Ldiv = static_cast<T>(std::stod(params["Ldiv"])); 

            // Define the cell-cell neighbor graph and get its connected
            // components 
            Graph graph = getNeighborGraph<T>(cells, R, Ldiv);
            std::vector<int> components = getConnectedComponents<T>(graph); 

            // Output the graph 
	    writeGraph<T>(graph, components, outfilename); 
        }
    }

    return 0; 
}
