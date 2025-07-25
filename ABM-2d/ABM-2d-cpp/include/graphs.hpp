/**
 * A set of functions for instantiating and analyzing a graph describing
 * cell-cell contacts.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/19/2025
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

typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS> Graph; 

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
 * Get the degrees of all vertices in the given graph. 
 *
 * The i-th entry in the returned array is the number of edges incident on 
 * the vertex i. 
 *
 * @param graph
 * @returns Degrees of vertices in the graph. 
 */
template <typename T>
Array<int, Dynamic, 1> getDegrees(const Graph& graph)
{
    Array<int, Dynamic, 1> degrees = Array<int, Dynamic, 1>::Zero(boost::num_vertices(graph));
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it; 
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        degrees(i) += 1; 
        degrees(j) += 1; 
    }

    return degrees; 
}

/**
 * Get the local clustering coefficients for all vertices in the given graph. 
 *
 * @param graph
 * @returns Local clustering coefficients for all vertices in the graph. 
 */
template <typename T>
Array<T, Dynamic, 1> getLocalClusteringCoefficients(const Graph& graph)
{
    Array<T, Dynamic, 1> coefs = Array<T, Dynamic, 1>::Zero(boost::num_vertices(graph));
    std::pair<boost::graph_traits<Graph>::out_edge_iterator, 
              boost::graph_traits<Graph>::out_edge_iterator> it;
    for (int i = 0; i < boost::num_vertices(graph); ++i)
    {
        // First identify the neighbors of i
        std::vector<int> neighbors;  
        for (it = boost::out_edges(i, graph); it.first != it.second; ++it.first)
        {
            // Rigorously check which vertex is the neighbor and which is i
            // (the former should always be the target) 
            boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
            int j = boost::source(edge, graph); 
            int k = boost::target(edge, graph);
            if (i == j) 
                neighbors.push_back(k); 
            else 
                neighbors.push_back(j); 
        }
        int n_neighbors = neighbors.size();

        // If degree = 0 or 1, then the local clustering coefficient is undefined
        if (n_neighbors == 0 || n_neighbors == 1)
        {
            coefs(i) = -1;
        }
        // Otherwise, check, for each pair of neighbors, if they are connected
        // by a third edge
        else
        {
            T n_cluster = 0.0; 
            for (int j = 0; j < neighbors.size(); ++j)
            {
                for (int k = j + 1; k < neighbors.size(); ++k)
                {
                    if (boost::edge(neighbors[j], neighbors[k], graph).second)
                        n_cluster += 1; 
                }
            } 
            coefs(i) = 2 * n_cluster / (n_neighbors * (n_neighbors - 1));
        }    
    }

    return coefs;
}

/**
 * Get all triangles (3-cliques) in the given graph. 
 *
 * Each row in the returned array is a combination of three vertices that 
 * form a triangle. 
 *
 * @param graph 
 * @returns Array of triangle-forming vertices. 
 */
template <typename T>
Array<int, Dynamic, 3> getTriangles(const Graph& graph)
{
    const int n = boost::num_vertices(graph);
    int nt = 0;  
    Array<int, Dynamic, 3> triangles(nt, 3);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (boost::edge(i, j, graph).second)
            {
                for (int k = j + 1; k < n; ++k)
                {
                    if (boost::edge(j, k, graph).second && boost::edge(i, k, graph).second)
                    {
                        nt++; 
                        triangles.conservativeResize(nt, 3); 
                        triangles(nt - 1, 0) = i; 
                        triangles(nt - 1, 1) = j; 
                        triangles(nt - 1, 2) = k; 
                    }
                }
            }
        }
    }

    return triangles; 
}

/**
 * Get all tetrahedra (4-cliques) in the given graph. 
 *
 * Each row in the returned array is a combination of four vertices that 
 * form a tetrahedron.  
 *
 * @param graph 
 * @returns Array of tetrahedron-forming vertices. 
 */
template <typename T>
Array<int, Dynamic, 4> getTetrahedra(const Graph& graph)
{
    const int n = boost::num_vertices(graph);
    int nt = 0;  
    Array<int, Dynamic, 4> tetrahedra(nt, 4);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (boost::edge(i, j, graph).second)    // Check that i-j
            {
                for (int k = j + 1; k < n; ++k)
                {
                    // Check that j-k and i-k
                    if (boost::edge(j, k, graph).second && boost::edge(i, k, graph).second)
                    {
                        for (int m = k + 1; m < n; ++m)
                        {
                            // Check that i-m, j-m, and k-m
                            if (boost::edge(i, m, graph).second && 
                                boost::edge(j, m, graph).second &&
                                boost::edge(k, m, graph).second)
                            {
                                nt++; 
                                tetrahedra.conservativeResize(nt, 4); 
                                tetrahedra(nt - 1, 0) = i; 
                                tetrahedra(nt - 1, 1) = j; 
                                tetrahedra(nt - 1, 2) = k;
                                tetrahedra(nt - 1, 3) = m;
                            }
                        } 
                    }
                }
            }
        }
    }

    return tetrahedra; 
}

/**
 * Write the given graph and its connectivity information to file. 
 *
 * @param graph
 * @param components
 * @param degrees
 * @param filename
 * @param write_cluster_coefs
 * @param cluster_coefs 
 * @param write_triangles
 * @param triangles
 * @param write_tetrahedra
 * @param tetrahedra
 */
template <typename T>
void writeGraph(const Graph& graph, std::vector<int>& components,
                const Ref<const Array<int, Dynamic, 1> >& degrees,
                const std::string filename, const bool write_cluster_coefs,  
                const Ref<const Array<T, Dynamic, 1> >& cluster_coefs,         
                const bool write_triangles, 
                const Ref<const Array<int, Dynamic, 3> >& triangles,
                const bool write_tetrahedra,
                const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
{
    // Get the component sizes 
    int num_components = *std::max_element(components.begin(), components.end()) + 1;
    Array<int, Dynamic, 1> component_sizes = Array<int, Dynamic, 1>::Zero(num_components); 
    for (int i = 0; i < components.size(); ++i)
        component_sizes(components[i]) += 1;

    // Open output file and write header 
    std::ofstream outfile(filename);
	outfile << std::setprecision(10); 
    outfile << "NUM_VERTICES\t" << boost::num_vertices(graph) << std::endl; 
    outfile << "NUM_EDGES\t" << boost::num_edges(graph) << std::endl; 
    outfile << "NUM_COMPONENTS\t" << num_components << std::endl;
    if (write_triangles)
        outfile << "NUM_TRIANGLES\t" << triangles.rows() << std::endl;
    if (write_tetrahedra)
        outfile << "NUM_TETRAHEDRA\t" << tetrahedra.rows() << std::endl;

    // Compute the degree distribution
    Array<int, Dynamic, 1> degree_dist = Array<int, Dynamic, 1>::Zero(degrees.maxCoeff() + 1);
    for (int i = 0; i < boost::num_vertices(graph); ++i)
        degree_dist(degrees(i)) += 1;
    for (int i = 0; i <= degrees.maxCoeff(); ++i) 
        outfile << "DEGREE_DIST\t" << i << '\t' << degree_dist(i) << std::endl; 

    // Write the vertices to the output file 
    for (int i = 0; i < boost::num_vertices(graph); ++i)
    {
        if (write_cluster_coefs)
            outfile << "VERTEX\t" << i << "\tCOMPONENT:" << components[i]
                                       << "\tDEGREE:" << degrees(i)
                                       << "\tCLUSTER:" << cluster_coefs(i) << std::endl;
        else 
            outfile << "VERTEX\t" << i << "\tCOMPONENT:" << components[i]
                                       << "\tDEGREE:" << degrees(i) << std::endl;
    }        

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

    // Write triangles to the output file, if desired 
    if (write_triangles)
    {
        for (int i = 0; i < triangles.rows(); ++i)
        {
            outfile << "TRIANGLE\t" << triangles(i, 0) << '\t'
                                    << triangles(i, 1) << '\t'
                                    << triangles(i, 2) << std::endl;
        } 
    }
    
    // Write tetrahedra to the output file, if desired 
    if (write_tetrahedra)
    {
        for (int i = 0; i < tetrahedra.rows(); ++i)
        {
            outfile << "TETRAHEDRA\t" << tetrahedra(i, 0) << '\t'
                                      << tetrahedra(i, 1) << '\t'
                                      << tetrahedra(i, 2) << '\t'
                                      << tetrahedra(i, 3) << std::endl; 
        }
    }

    // Close output file 
    outfile.close();
}

#endif 
