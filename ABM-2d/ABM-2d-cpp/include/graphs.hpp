/**
 * A set of functions for instantiating and analyzing a graph describing
 * cell-cell contacts.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/14/2025
 */

#ifndef CELL_CELL_NEIGHBOR_GRAPH_HPP
#define CELL_CELL_NEIGHBOR_GRAPH_HPP 

#include <iostream>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include "distances.hpp"
#include "mechanics.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph; 

/**
 * Define and return a graph in which each cell is a vertex, and two vertices
 * are connected by an edge if (1) both cells are in group 1 and (2) the two
 * cells are within 2 * R distance of each other.
 *
 * @param cells 
 * @param R
 * @param Ldiv
 * @returns Cell-cell neighbor graph. 
 */
template <typename T>
Graph getNeighborGraph(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                       const T R, const T Ldiv)
{
    // Get the cell-cell distance between each pair of neighboring cells in
    // the population
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, 2 * R, R, Ldiv);

    // Define a graph and add a vertex for each cell  
    Graph graph(cells.rows());

    // Add an edge between each pair of group 1 cells that are within 2 * R
    // distance 
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = neighbors(k, 0); 
        int j = neighbors(k, 1); 
        T dij = neighbors(k, Eigen::seq(2, 3)).matrix().norm();
        if (dij < 2 * R && cells(i, __colidx_group) == 1 && cells(j, __colidx_group) == 1)
            boost::add_edge(i, j, graph);
    }

    return graph; 
}

/**
 * Get the connected components in the given graph.
 *
 * The i-th entry in the returned std::vector is the index of the connected
 * component containing vertex i.
 *
 * @param graph
 * @returns Indices of connected components containing each vertex.  
 */
template <typename T>
std::vector<int> getConnectedComponents(const Graph& graph)
{
    std::vector<int> components(boost::num_vertices(graph));
    boost::connected_components(graph, &components[0]);

    return components;  
}

/**
 * Write the given graph and its connectivity information to file. 
 *
 * @param graph
 * @param components
 * @param filename
 */
template <typename T>
void writeGraph(const Graph& graph, std::vector<int>& components,
                const std::string filename)
{
    // Get the component sizes 
    int num_components = *std::max_element(components.begin(), components.end()) + 1;
    Array<int, Dynamic, 1> component_sizes = Array<int, Dynamic, 1>::Zero(num_components); 
    for (int i = 0; i < components.size(); ++i)
        component_sizes(components[i]) += 1;

    // Write the vertices to the output file 
    std::ofstream outfile(filename);
    outfile << "NUM_VERTICES\t" << boost::num_vertices(graph) << std::endl; 
    outfile << "NUM_EDGES\t" << boost::num_edges(graph) << std::endl; 
    outfile << "NUM_COMPONENTS\t" << num_components << std::endl; 
    for (int i = 0; i < boost::num_vertices(graph); ++i)
        outfile << "VERTEX\t" << i << '\t' << components[i] << std::endl;

    // Write the edges to the output file
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it; 
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        outfile << "EDGE\t" << i << '\t' << j << std::endl; 
    }

    // Write the component sizes to the output file 
    for (int i = 0; i < num_components; ++i)
        outfile << "COMPONENT\t" << i << '\t' << component_sizes(i) << std::endl;

    // Close output file 
    outfile.close();  
}

#endif 
