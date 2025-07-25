/**
 * Classes and functions for undirected graphs.
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/18/2025
 */

#ifndef UNDIRECTED_WEIGHTED_GRAPHS_HPP
#define UNDIRECTED_WEIGHTED_GRAPHS_HPP

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <stack>
#include <queue>
#include <limits>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
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
 * Construct a dictionary that stores an index for each edge in the graph, 
 * according to lexicographic order. 
 *
 * @param graph Input graph. 
 */
std::unordered_map<std::pair<int, int>, int,
                   boost::hash<std::pair<int, int> > > getEdgeOrdering(const Graph& graph)
{
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

    return edge_map;  
}

/**
 * Get the shortest-path tree of the given graph, using Dijkstra's algorithm.
 *
 * The tree is in fact a forest, in which all vertices that are not connected
 * to the root are isolated, and the connected component containing the root 
 * is a tree. 
 *
 * The i-th entry in the returned std::vector is the i-th vertex in the 
 * path (including u as the 0-th entry and v as the last). 
 *
 * @param graph Input graph.
 * @param root Root vertex. 
 * @returns Minimum weight paths from the source vertex to every other 
 *          vertex, alongside the tree itself as a separate graph.  
 */
std::pair<std::vector<std::vector<int> >, Graph> getMinimumWeightPathTree(const Graph& graph,
                                                                          const int root = 0)
{
    const int nv = boost::num_vertices(graph);

    // Define a map of vertices to their indices 
    auto idx_map = boost::get(boost::vertex_index, graph);

    // Define a vector of predecessor vertices 
    std::vector<boost::graph_traits<Graph>::vertex_descriptor> pred(nv);

    // Define distances to the root
    std::vector<double> dist; 
    for (int i = 0; i < nv; ++i)
        dist.push_back(std::numeric_limits<double>::max()); 

    // Make corresponding Boost property maps
    auto pred_map = boost::make_iterator_property_map(pred.begin(), idx_map); 
    auto dist_map = boost::make_iterator_property_map(dist.begin(), idx_map);

    // Run Dijkstra's algorithm 
    boost::dijkstra_shortest_paths(
        graph, root, boost::predecessor_map(pred_map).distance_map(dist_map)
    );

    // For each vertex ...
    Graph tree(nv); 
    std::vector<std::vector<int> > paths; 
    for (int i = 0; i < nv; ++i)
    {
        std::vector<int> path; 

        // If there is no path from root to the vertex, then store an empty
        // path
        if (dist[i] == std::numeric_limits<double>::max())
        {
            paths.push_back(path);
        }
        // Otherwise, traverse the path from the vertex to the root
        else 
        { 
            for (int j = i; j != root; j = pred[j])
                path.push_back(j);
            path.push_back(root); 
            std::reverse(path.begin(), path.end());
            paths.push_back(path);

            // Add each edge in the path to the tree
            if (path.size() > 1)
            {
                for (auto it = path.begin() + 1; it != path.end(); ++it)
                {
                    int u = *std::prev(it); 
                    int v = *it;
                    if (!boost::edge(u, v, tree).second)
                        boost::add_edge(u, v, EdgeProperty(1.0), tree); 
                }
            }
        }
    }

    return std::make_pair(paths, tree);  
}

/**
 * A helper function that computes the unique path from u to v in the given
 * Dijkstra shortest-path tree, which is defined in terms of the paths in
 * the tree (`tree_paths`), the predecessor map (`parents`), and a map 
 * indicating the depth of each vertex with respect to the root (`depths`).
 *
 * @param tree_paths Vector of paths from the root to each vertex in the
 *                   Dijkstra shortest-path tree. If a vertex is disconnected
 *                   from the root, the path is empty.  
 * @param parents Vector indicating the predecessor of each vertex along the
 *                path from the root. 
 * @param depths Vector indicating the depth of each vertex with respect to 
 *               the root. 
 * @param u First vertex. 
 * @param v Second vertex. 
 * @returns Unique path from u to v in the tree. 
 */
std::vector<int> getPathInMinimumWeightPathTree(const std::vector<std::vector<int> >& tree_paths,
                                                const std::vector<int>& parents, 
                                                const std::vector<int>& depths, 
                                                const int u, const int v)
{
    // Return an empty path if either u or v is disconnected from the root
    std::vector<int> path_uv; 
    if (depths[u] == -2 || depths[v] == -2)
        return path_uv; 

    // Find the lowest common ancestor of u and v
    //
    // In whichever path is longer, travel up the path until the depths are
    // the same  
    int curr_u = u; 
    int curr_v = v; 
    while (depths[curr_u] > depths[curr_v])
        curr_u = parents[curr_u]; 
    while (depths[curr_v] > depths[curr_u])
        curr_v = parents[curr_v];

    // Then climb up the two paths until we reach a common ancestor
    while (curr_u != curr_v)
    {
        curr_u = parents[curr_u]; 
        curr_v = parents[curr_v]; 
    }

    // Climb up the path from u to the lowest common ancestor
    int curr_vertex = u; 
    while (curr_vertex != curr_u)
    {
        path_uv.push_back(curr_vertex); 
        curr_vertex = parents[curr_vertex]; 
    }

    // Then climb back down the path from the lowest common ancestor to v
    std::vector<int> path_v = tree_paths[v];
    auto it = std::find(path_v.begin(), path_v.end(), curr_vertex); 
    while (curr_vertex != v)
    {
        path_uv.push_back(curr_vertex);
        it++; 
        curr_vertex = *it; 
    }
    path_uv.push_back(v); 

    return path_uv;  
}

/**
 * Define a map of all shortest paths between all pairs of vertices in the
 * given graph. 
 *
 * The output is a map that sends each pair of vertices, (u, v), to a
 * shortest path from u to v in the graph. 
 *
 * @param graph Input graph. 
 * @returns Minimum weight paths from the source vertex to every other 
 *          vertex. 
 */
std::map<std::pair<int, int>, std::vector<int> > getMinimumWeightPaths(const Graph& graph)
{
    const int nv = boost::num_vertices(graph); 

    // Get the minimum-weight-path tree rooted at each vertex 
    std::vector<std::vector<std::vector<int> > > tree_paths; 
    std::vector<std::vector<int> > parents; 
    std::vector<std::vector<int> > depths; 
    for (int i = 0; i < nv; ++i)
    {
        auto result = getMinimumWeightPathTree(graph, i);
        std::vector<std::vector<int> > tree_paths_i = result.first; 

        // Define the parent and depth of each vertex with respect to the root 
        std::vector<int> parents_i, depths_i; 
        for (int i = 0; i < nv; ++i)
        {
            parents_i.push_back(-2);    // Set both values to -2 by default 
            depths_i.push_back(-2); 
        } 
        parents_i[i] = -1;              // The root has no parent
        depths_i[i] = 0;                // The root has depth zero
        for (auto& path : tree_paths_i) 
        {
            // If the path contains at least one edge, traverse it 
            if (path.size() > 1)
            {
                int depth = 1; 
                for (auto it = path.begin() + 1; it != path.end(); ++it)
                {
                    auto prev = std::prev(it);
                    parents_i[*it] = *prev; 
                    depths_i[*it] = depth; 
                    depth++; 
                }
            }
        }
        tree_paths.push_back(tree_paths_i);
        parents.push_back(parents_i); 
        depths.push_back(depths_i);  
    }

    // For each pair of vertices ...
    std::map<std::pair<int, int>, std::vector<int> > min_paths;  
    for (int u = 0; u < nv; ++u)
    {
        for (int v = u + 1; v < nv; ++v)
        {
            // Get the path from u to v in each minimum-weight-path tree
            int min_length = std::numeric_limits<int>::max(); 
            std::vector<int> min_path;  
            for (int w = 0; w < nv; ++w)
            {
                std::vector<int> path = getPathInMinimumWeightPathTree(
                    tree_paths[w], parents[w], depths[w], u, v
                );

                // If the path is empty, then there is no path from u to 
                // v in the tree (which is actually a forest)
                if (path.size() > 0)
                {
                    // Is this path shorter than the current choice of
                    // shortest path? 
                    if (path.size() < min_length)
                    {
                        min_length = path.size(); 
                        min_path = path; 
                    }
                }
            }
            min_paths[std::make_pair(u, v)] = min_path; 
        }
    }

    return min_paths; 
}

/**
 * Get a minimum cycle basis for the given graph.
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
    // Get the minimum-weight paths between all pairs of vertices 
    const int nv = boost::num_vertices(graph);
    const int ne = boost::num_edges(graph);
    std::map<std::pair<int, int>, std::vector<int> > min_paths = getMinimumWeightPaths(graph);

    // Get the number of connected components 
    std::vector<int> components = getConnectedComponents(graph); 
    const int nc = *std::max_element(components.begin(), components.end()) + 1; 

    // For each vertex and edge in the graph ...
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it;
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
            std::vector<int> path1, path2;
            if (pair1.first < pair1.second)
                path1 = min_paths[pair1]; 
            else    // pair1.first == pair1.second 
                path1.push_back(pair1.first);  
            if (pair2.first < pair2.second)
                path2 = min_paths[pair2]; 
            else    // pair2.first == pair2.second 
                path2.push_back(pair2.first);

            // Check that these paths are nonempty, so that i, u, and v are 
            // mutually connected 
            //
            // It suffices to check only the path from i to u, because if 
            // this path is nonempty, so must the path from i to v
            if (path1.size() > 0)
            {
                // Check that (u, v) is not an edge in either path
                //
                // First check the path from i to u ... 
                bool found_edge = false; 
                for (auto it = path1.begin() + 1; it != path1.end(); ++it)
                {
                    int j = *std::prev(it); 
                    int k = *it; 
                    if ((j == u && k == v) || (j == v && k == u))
                    {
                        found_edge = true;
                        break;
                    } 
                }
                if (!found_edge)
                {
                    // ... then check the path from i to v 
                    for (auto it = path2.begin() + 1; it != path2.end(); ++it)
                    {
                        int j = *std::prev(it); 
                        int k = *it; 
                        if ((j == u && k == v) || (j == v && k == u))
                        {
                            found_edge = true;
                            break;
                        } 
                    }
                } 
                if (!found_edge)    // If (u, v) is not in either path ... 
                {
                    // Re-order the paths to ensure that they flow from i to u and 
                    // from v to i
                    if (pair1.first == u)
                        std::reverse(path1.begin(), path1.end()); 
                    if (pair2.first == i)
                        std::reverse(path2.begin(), path2.end()); 

                    // Create the cycle formed by the path from i to u, the edge
                    // (u, v), and the path from v to i 
                    //
                    // Keep track of each vertex that is encountered along the path, 
                    // and ensure that none are encountered more than once 
                    std::vector<int> cycle; 
                    Matrix<int, Dynamic, 1> encountered = Matrix<int, Dynamic, 1>::Zero(nv);
                    bool found_repeat_vertex = false;
                    
                    // Traverse path from i to u 
                    for (auto it = path1.begin(); it != path1.end(); ++it)
                    {
                        int w = *it; 
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

                    // Add edge (u, v)
                    if (!encountered(v) || v == i) 
                    {
                        encountered(v) = 1; 
                        cycle.push_back(v);
                    }
                    else    // v has been encountered already and v != i 
                    {
                        continue; 
                    }

                    // Traverse path from v to i
                    if (path2.size() == 1)        // (u, v = i) was the last edge 
                    {
                        cycle.pop_back();         // Remove the last vertex (v = i) 
                    }
                    else if (path2.size() == 2)   // (v, i) is the last edge 
                    {
                        // Do nothing 
                    } 
                    else    // path2.size() > 2 
                    {
                        for (auto it = path2.begin() + 1; it != path2.end(); ++it)
                        {
                            int w = *it; 
                            if (!encountered(w) || w == i)
                            {
                                encountered(w) = 1; 
                                cycle.push_back(w); 
                            }
                            else    // w has been encountered already and w != i (this could happen)
                            {
                                found_repeat_vertex = true; 
                                break; 
                            }
                        }
                        if (found_repeat_vertex)
                            continue;
                        cycle.pop_back();   // Remove the last vertex (v = i) 
                    }  

                    // Now add the cycle to the collection after removing the 
                    // repeated vertex
                    cycles.push_back(std::make_pair(cycle, cycle.size()));
                }
            }
        }
    }

    // Order the cycles by length 
    std::sort(
        cycles.begin(), cycles.end(),
        [](const std::pair<std::vector<int>, int>& x, const std::pair<std::vector<int>, int>& y)
        {
            return x.second < y.second; 
        }
    );

    // If there are no cycles in the collection, return an empty matrix 
    if (cycles.size() == 0)
        return Matrix<T, Dynamic, Dynamic>::Zero(ne, 0);

    // Get a lexicographic ordering of the edges 
    auto edge_map = ::getEdgeOrdering(graph);

    // Add the first cycle to the basis 
    Matrix<T, Dynamic, Dynamic> basis(ne, 1); 
    basis.col(0) = Matrix<T, Dynamic, 1>::Zero(ne);
    int u, v;
    std::pair<int, int> edge;
    std::vector<int> curr_cycle = cycles[0].first;  
    for (auto it = curr_cycle.begin() + 1; it != curr_cycle.end(); ++it)
    {
        u = *std::prev(it); 
        v = *it; 
        edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
        basis(edge_map[edge], 0) = 1; 
    }
    u = curr_cycle[curr_cycle.size() - 1];
    v = curr_cycle[0]; 
    edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
    basis(edge_map[edge], 0) = 1;

    // Run through the cycles in order ...
    int cycle_idx = 1;  
    while (basis.cols() < ne - nv + nc)   // Expected number of cycles in the basis
    {
        // Get the vector corresponding to the cycle
        curr_cycle = cycles[cycle_idx].first; 
        Matrix<T, Dynamic, 1> cycle_vec = Matrix<T, Dynamic, 1>::Zero(ne);
        for (auto it = curr_cycle.begin() + 1; it != curr_cycle.end(); ++it)
        {
            u = *std::prev(it); 
            v = *it; 
            edge = (u < v ? std::make_pair(u, v) : std::make_pair(v, u)); 
            cycle_vec(edge_map[edge]) = 1; 
        }
        u = curr_cycle[curr_cycle.size() - 1]; 
        v = curr_cycle[0]; 
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
 * Parse the graph in the given file.
 *
 * All information about point coordinates and higher-order simplices is
 * ignored.  
 *
 * @param filename Input filename.
 * @returns A Graph instance storing the graph in the file.   
 */
Graph readGraph(const std::string& filename)
{
    // Open input file
    std::ifstream infile(filename);
    Graph graph; 

    // The first line contains the number of vertices 
    std::string line, token; 
    std::stringstream ss; 
    std::getline(infile, line);
    ss << line;  
    std::getline(ss, token, '\t');      // "NUM_VERTICES"
    std::getline(ss, token);            // The number of vertices 
    const int nv = std::stoi(token); 
    for (int i = 0; i < nv; ++i)
        boost::add_vertex(graph);

    // Skip over all subsequent lines until we reach the edges 
    while (line.rfind("EDGE\t", 0) != 0)
        std::getline(infile, line);

    // Add each edge to the graph
    while (line.rfind("EDGE\t", 0) == 0)
    {
        ss.str(std::string());  
        ss << line; 
        std::getline(ss, token, '\t');      // "EDGE"
        std::getline(ss, token, '\t');      // First vertex
        int i = std::stoi(token); 
        std::getline(ss, token, '\t');      // Second vertex
        int j = std::stoi(token); 
        std::getline(ss, token);            // Distance between the two vertices
        double dij = std::stod(token); 
        boost::add_edge(i, j, EdgeProperty(dij), graph);

        // Get next line 
        std::getline(infile, line);
    }

    // Close input file and return 
    infile.close(); 
    
    return graph; 
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
            outfile << "TETRAHEDRON\t" << tetrahedra(i, 0) << '\t'
                                       << tetrahedra(i, 1) << '\t'
                                       << tetrahedra(i, 2) << '\t'
                                       << tetrahedra(i, 3) << std::endl; 
        }
    }

    // Close output file 
    outfile.close();
}

#endif
