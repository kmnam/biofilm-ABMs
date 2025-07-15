/**
 * Classes and functions for undirected graphs.
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/15/2025
 */

#ifndef UNDIRECTED_WEIGHTED_GRAPHS_HPP
#define UNDIRECTED_WEIGHTED_GRAPHS_HPP

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <stack>
#include <queue>
#include <map>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/container_hash/hash.hpp>
#include "distances.hpp"
#include "mechanics.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::property<boost::edge_weight_t, double> EdgeProperty; 
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> Graph;

/**
 * Define and return a graph in which each cell is a vertex, and two vertices
 * are connected by an edge if (1) both cells are in group 1 and (2) the two
 * cells are within 2 * R distance of each other.
 *
 * @param cells Input population of cells.  
 * @param R Cell radius, including the EPS. 
 * @param Ldiv Cell division length. 
 * @returns Cell-cell neighbor graph. 
 */
template <typename T>
Graph getNeighborGraph(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                       const T R, const T Ldiv)
{
    // Get the cell-cell distance between each pair of neighboring cells in
    // the population
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, 2 * R, R, Ldiv);

    // Define a graph and add a vertex for each cell  
    Graph graph(cells.rows());

    // Add an edge between each pair of group 1 cells that are within 2 * R
    // distance 
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = neighbors(k, 0); 
        int j = neighbors(k, 1); 
        T dij = neighbors(k, Eigen::seq(2, 4)).matrix().norm();
        if (dij < 2 * R && cells(i, __colidx_group) == 1 && cells(j, __colidx_group) == 1)
            boost::add_edge(i, j, EdgeProperty(dij), graph);
    }

    return graph; 
}

/**
 * Get the connected components in the given graph.
 *
 * The i-th entry in the returned std::vector is the index of the connected
 * component containing vertex i.
 *
 * @param graph Input graph. 
 * @returns Indices of connected components containing each vertex.  
 */
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
 * @param graph Input graph. 
 * @returns Degrees of vertices in the graph. 
 */
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
 * @param graph Input graph. 
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
 * @param graph Input graph. 
 * @returns Array of triangle-forming vertices. 
 */
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
 * @param graph Input graph. 
 * @returns Array of tetrahedron-forming vertices. 
 */
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
 * Get the shortest path from vertex u to vertex v in the given graph.
 *
 * The i-th entry in the returned std::vector is the i-th vertex in the 
 * path (including u as the 0-th entry and v as the last). 
 *
 * @param graph Input graph.
 * @param u First vertex. 
 * @param v Second vertex.  
 * @returns Minimum weight path from u to v. 
 */
std::vector<int> getMinimumWeightPath(const Graph& graph, const int u, const int v)
{
    const int nv = boost::num_vertices(graph);

    // Define a map of vertices to their indices 
    auto idx_map = boost::get(boost::vertex_index, graph);

    // Define a vector of predecessor vertices 
    std::vector<boost::graph_traits<Graph>::vertex_descriptor> pred(nv);

    // Define distances to the source vertex 
    std::vector<double> dist; 
    for (int i = 0; i < nv; ++i)
        dist.push_back(std::numeric_limits<double>::infinity());

    // Make corresponding Boost property maps
    auto pred_map = boost::make_iterator_property_map(pred.begin(), idx_map); 
    auto dist_map = boost::make_iterator_property_map(dist.begin(), idx_map); 

    // Run Dijkstra's algorithm 
    boost::dijkstra_shortest_paths(
        graph, u, boost::predecessor_map(pred_map), boost::distance_map(dist_map)
    );
    std::vector<int> path; 

    // If there is no path from source to target, then return
    if (dist[v] == std::numeric_limits<double>::infinity())
        return path; 

    // Otherwise, traverse the path from target to source 
    for (int i = v; i != u; i = pred[i])
        path.push_back(i);
    path.push_back(u);
    std::reverse(path.begin(), path.end()); 

    return path;  
}

/**
 * Get a minimum cycle basis for the given graph, which is assumed to be 
 * connected.
 *
 * Each cycle is returned as a vector of vertex indices, with edges between
 * consecutive vertices (and the edge from the last vertex to the first).
 *
 * This implements the algorithm described by Horton, SIAM J. Comput. (1987).  
 *
 * @param graph Input graph. 
 * @returns Minimum cycle basis for the given graph.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getMinimumCycleBasis(const Graph& graph)
{
    // Get the minimum-weight path between each pair of vertices
    const int nv = boost::num_vertices(graph);
    std::map<std::pair<int, int>, std::vector<int> > paths; 
    for (int i = 0; i < nv; ++i)
    {
        for (int j = i + 1; j < nv; ++j)
        {
            paths[std::make_pair(i, j)] = getMinimumWeightPath(graph, i, j); 
        }
    }

    // Generate a lexicographically sorted vector of edges and a map of edge
    // indices with respect to this ordering  
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it;
    std::vector<std::pair<int, int> > edges; 
    std::unordered_map<std::pair<int, int>, int,
                       boost::hash<std::pair<int, int> > > edge_map; 
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int u = boost::source(edge, graph); 
        int v = boost::target(edge, graph);
        if (u < v)
            edges.push_back(std::make_pair(u, v));
        else 
            edges.push_back(std::make_pair(v, u));  
    }
    std::sort(edges.begin(), edges.end());
    int i = 0; 
    for (auto&& edge : edges)
    {
        edge_map[edge] = i; 
        i++; 
    }    

    // For each vertex and edge in the graph ...
    std::vector<std::pair<std::vector<int>, int> > cycles;
    for (int i = 0; i < nv; ++i)
    {
        for (it = boost::edges(graph); it.first != it.second; ++it.first)
        {
            boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
            int u = boost::source(edge, graph); 
            int v = boost::target(edge, graph);

            // Get the paths from i to u and from i to v 
            std::pair<int, int> pair1 = (i < u ? std::make_pair(i, u) : std::make_pair(u, i)); 
            std::pair<int, int> pair2 = (i < v ? std::make_pair(i, v) : std::make_pair(v, i));
            std::vector<int> path1 = paths[pair1]; 
            std::vector<int> path2 = paths[pair2]; 

            // Re-order the paths to ensure that they flow from i to u and 
            // from v to i
            if (pair1.first == u)
                std::reverse(path1.begin(), path1.end()); 
            if (pair2.first == i)
                std::reverse(path1.begin(), path1.end());  

            // Create the cycle formed by the path from i to u, the edge
            // (u, v), and the path from v to i 
            //
            // Keep track of each vertex that is encountered along the path, 
            // and ensure that none are encountered more than once 
            std::vector<int> cycle; 
            Matrix<int, Dynamic, 1> encountered = Matrix<int, Dynamic, 1>::Zero(nv);
            bool found_repeat_vertex = false;
            for (const int& w : path1)                    // Path from i to u
            {
                if (!encountered(w))
                {
                    encountered(w) = 1; 
                    cycle.push_back(w); 
                }
                else    // w has been encountered already (this should not happen)
                {
                    found_repeat_vertex = true; 
                    break; 
                }
            }
            if (found_repeat_vertex)
                continue;
            cycle.push_back(v);                           // Edge (u, v)
            for (int j = 1; j < path2.size() - 1; ++j)    // Path from v to i (skipping v and i) 
            {
                if (!encountered(w))
                {
                    encountered(w) = 1; 
                    cycle.push_back(w); 
                }
                else    // w has been encountered already (this could happen)
                {
                    found_repeat_vertex = true; 
                    break; 
                }
            }
            if (found_repeat_vertex)
                continue;

            // Now add the cycle to the collection 
            cycles.push_back(std::make_pair(cycle, cycle.size()));
        }
    }

    // Order the cycles by length 
    std::sort(
        cycles.begin(), cycles.end()
        [](const std::pair<std::vector<int>, int>& x, const std::pair<std::vector<int>, int>& y)
        {
            return x.second < y.second; 
        }
    );

    // If there are no cycles in the collection, return an empty matrix 
    if (cycles.size() == 0)
        return Matrix<T, Dynamic, Dynamic>::Zero(ne, 0);

    // Add the first cycle to the basis 
    const int ne = boost::num_edges(graph);
    Matrix<T, Dynamic, Dynamic> basis(ne, 1); 
    basis.col(0) = Matrix<T, Dynamic, 1>::Zero(ne);
    int u, v;
    std::pair<int, int> edge;  
    for (int i = 0; i < cycles[0].size() - 1; ++i)
    {
        u = cycles[0][i];
        v = cycles[0][i + 1];
        edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
        basis(edge_map[edge], 0) = 1; 
    }
    u = cycles[0][cycles[0].size() - 1]; 
    v = cycles[0][0]; 
    edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
    basis(edge_map[edge], 0) = 1;

    // Run through the cycles in order ...
    int cycle_idx = 1;  
    while (basis.cols() < ne - nv + 1)   // Expected number of cycles in the basis
    {
        // Get the vector corresponding to the cycle
        std::vector<int> cycle = cycles[cycle_idx];  
        Matrix<T, Dynamic, 1> cycle_vec = Matrix<T, Dynamic, 1>::Zero(ne);
        for (int i = 0; i < cycle.size() - 1; ++i)
        {
            u = cycle[i];
            v = cycle[i + 1];
            edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
            cycle_vec(edge_map[edge]) = 1; 
        }
        u = cycle[cycle.size() - 1]; 
        v = cycle[0]; 
        edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
        cycle_vec(edge_map[edge]) = 1;

        // Check that the vector is linearly independent to the previously
        // collected basis vectors
        //
        // Calculate the row echelon form of [basis | cycle] 
        Matrix<T, Dynamic, Dynamic> system(ne, basis.cols() + 1); 
        system(Eigen::all, Eigen::seq(0, basis.cols() - 1)) = basis; 
        system.col(basis.cols()) = cycle_vec;
        system = ::rowEchelonForm<T>(system);

        // Identify if there is an inconsistency in the row echelon form, 
        // in which case the cycle is linearly independent from the basis 
        bool found_inconsistency = false; 
        for (int i = 0; i < ne; ++i)
        {
            if ((system.row(i).head(basis.cols()).array() == 0).all() && system(i, basis.cols()) != 0)
            {
                found_inconsistency = true; 
                break; 
            }
        } 
        if (found_inconsistency)
        {
            basis.conservativeResize(ne, basis.cols() + 1); 
            basis.col(basis.cols() - 1) = cycle_vec;  
        }

        // Move onto the next cycle 
        cycle_idx++; 
    }

    return basis; 
}

/**
 * Write the given graph and its connectivity information to file. 
 *
 * @param graph Input graph. 
 * @param components Vector of connected component sizes, from largest to
 *                   smallest.
 * @param degrees Vector of vertex degrees. 
 * @param filename Path to output file. 
 * @param write_cluster_coefs If true, write local clustering coefficients
 *                            to file. 
 * @param cluster_coefs Vector of local clustering coefficients. 
 * @param write_triangles If true, write triangles to file. 
 * @param triangles Array of vertices that form triangles. 
 * @param write_tetrahedra If true, write tetrahedra to file. 
 * @param tetrahedra Array of vertices that form tetrahedra. 
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
    auto weights = boost::get(boost::edge_weight, graph);  
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        double dij = weights[edge];
        outfile << "EDGE\t" << i << '\t' << j << '\t' << dij << std::endl; 
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
