/**
 * Classes and functions for graphs and simplicial complexes. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/26/2025
 */

#ifndef SIMPLICIAL_COMPLEXES_3D_HPP
#define SIMPLICIAL_COMPLEXES_3D_HPP

#include <iostream>
#include <memory>
#include <stack>
#include <queue>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/Matrix.h>
#include <gudhi/persistence_matrix_options.h>
#include "distances.hpp"
#include "mechanics.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::property<boost::edge_weight_t, double> EdgeProperty; 
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> Graph; 

/** ------------------------------------------------------------------- //
 *                            GRAPH UTILITIES                           //
 *  ------------------------------------------------------------------- */
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
 * @param graph Input graph. 
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
 * @param graph Input graph. 
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
    boost::property_map<Graph, boost::edge_weight_t>::type weights = boost::get(boost:edge_weight, graph);  
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        double dij = weights[*it]; 
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

/** ------------------------------------------------------------------- //
 *                         SIMPLICIAL COMPLEXES                         //
 *  ------------------------------------------------------------------- */
/**
 * Get the binomial coefficient, n choose k.
 *
 * @param n Total number of items.
 * @param k Number of items to choose. 
 * @returns n choose k. 
 */
int binom(const int n, const int k)
{
    if (n < k)
        return 0; 
    else if (k == 0 || n == k)
        return 1; 
    else 
        return binom(n - 1, k - 1) + binom(n - 1, k); 
}

/**
 * A simple quasi-recursive function for getting the power set of an ordered
 * set (i.e., a vector).
 *
 * @param vec Input ordered set. 
 * @param nonempty If true, skip the empty set. 
 * @returns Power set of input set. 
 */
std::vector<std::vector<int> > getPowerset(const std::vector<int>& vec, 
                                           const bool nonempty = true)
{
    std::vector<std::vector<int> > powerset;
    const int n = vec.size();  

    // Maintain a stack of sub-vectors 
    std::stack<std::pair<int, std::vector<int> > > stack;
    stack.push(std::make_pair(0, std::vector<int>({})));
    while (!stack.empty())
    {
        auto next = stack.top();
        stack.pop(); 
        int start = next.first; 
        std::vector<int> subset = next.second;
        if (!nonempty || subset.size() > 0)
            powerset.push_back(subset); 
        for (int i = start; i < n; ++i)
        {
            std::vector<int> new_subset(subset);
            new_subset.push_back(vec[i]);  
            stack.push(std::make_pair(i + 1, new_subset)); 
        } 
    }

    return powerset; 
}

/**
 * A simple quasi-recursive function for getting all k-combinations of an
 * ordered set (i.e., a vector). 
 *
 * @param vec Input ordered set. 
 * @param k Number of items to choose per combination. 
 * @returns All k-combinations of input set. 
 */
std::vector<std::vector<int> > getCombinations(const std::vector<int>& vec, 
                                               const int k)
{
    std::vector<std::vector<int> > combinations;
    const int n = vec.size();

    // Maintain a stack of sub-vectors 
    std::stack<std::pair<int, std::vector<int> > > stack;
    stack.push(std::make_pair(0, std::vector<int>({})));  
    while (!stack.empty())
    {
        auto next = stack.top();
        stack.pop(); 
        int start = next.first; 
        std::vector<int> path = next.second; 
        if (path.size() == k)
        {
            combinations.push_back(path); 
            continue; 
        }
        for (int i = start; i < n; ++i)
        {
            std::vector<int> newpath(path); 
            newpath.push_back(vec[i]); 
            stack.push(std::make_pair(i + 1, newpath)); 
        } 
    }

    return combinations;  
}

/**
 * A simple implementation of a trie. 
 */
struct TrieNode
{
    int letter;
    std::shared_ptr<TrieNode> parent; 
    std::vector<std::shared_ptr<TrieNode> > children;  
};

class Trie
{
    private:
        std::vector<TrieNode> nodes;
        std::shared_ptr<TrieNode> root;

        /**
         * Get the full string/value associated with the given node.
         *
         * The node is assumed to lie within the trie.  
         */
        std::vector<int> getValue(const std::shared_ptr<TrieNode>& ptr) const 
        {
            // Travel up the trie from the node to get the value 
            std::vector<int> value;
            std::shared_ptr<TrieNode> curr = ptr;  
            while (curr != this->root)
            {
                value.insert(value.begin(), curr->letter); 
                curr = curr->parent; 
            }

            return value; 
        } 

    public:
        /**
         * Initialize a trie with -1 at the root. 
         */
        Trie()
        {
            TrieNode node;
            node.letter = -1;
            node.parent = nullptr;
            this->nodes.push_back(node);
            this->root = std::make_shared<TrieNode>(node); 
        }

        ~Trie()
        {
        }

        /**
         * Return true if the trie contains a node for the full string/value.
         */
        bool containsValue(std::vector<int> value) const 
        {
            // Travel down the tree from the root, looking for a node that 
            // contains each successive entry in the value
            std::shared_ptr<TrieNode> curr = this->root;
            for (int i = 0; i < value.size(); ++i)
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node 
                    if (ptr->letter == value[i])
                    {
                        found_matching_child = true; 
                        curr = ptr; 
                        break;  
                    }
                }
                if (!found_matching_child)   // If there is no such node, return false
                    return false; 
            }

            return true;
        }

        /**
         * Clear the trie. 
         */
        void clear()
        {
            this->nodes.clear();
            TrieNode node;
            node.letter = -1;
            node.parent = nullptr;
            this->nodes.push_back(node);
            this->root = std::make_shared<TrieNode>(node); 
        }

        /**
         * Get the number of nodes in the trie. 
         */
        int getNumNodes() const 
        {
            return this->nodes.size(); 
        }

        /**
         * Get the height of the trie. 
         */ 
        int getHeight() const
        {
            // Keep track of nodes to be visited using a queue (for BFS), 
            // starting with the root
            int height = 0;  
            std::queue<std::shared_ptr<TrieNode> > queue;
            queue.push(this->root); 
            int nvisited = 0;
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                std::shared_ptr<TrieNode> curr = queue.front(); 
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue 
                for (auto& ptr : curr->children) 
                    queue.push(ptr);

                // Get the value of this node 
                std::vector<int> value = this->getValue(curr);
                if (height < value.size())
                    height = value.size();  
            }

            return height; 
        }

        /**
         * Insert the string/value into the trie. 
         */ 
        void insert(std::vector<int> value)
        {
            // Travel down the tree from the root, looking for the first 
            // letter at which the value deviates from the trie
            std::shared_ptr<TrieNode> curr = this->root; 
            for (int i = 0; i < value.size(); ++i)
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node 
                    if (ptr->letter == value[i])
                    {
                        found_matching_child = true; 
                        curr = ptr; 
                        break;  
                    }
                }

                // If there is no such node, add new nodes to cover the 
                // remaining values
                if (!found_matching_child)
                {
                    for (int j = i; j < value.size(); ++j)
                    {
                        // Define the new node and set its parent node  
                        TrieNode node; 
                        node.letter = value[j]; 
                        node.parent = curr; 
                        std::shared_ptr<TrieNode> ptr = std::make_shared<TrieNode>(node);
                        this->nodes.push_back(node);

                        // Also update the children of its parent node  
                        curr->children.push_back(ptr);

                        // Move on to the next node 
                        curr = ptr;  
                    }
                    break; 
                }
            }
        }

        /**
         * Insert the string/value into the trie. 
         */
        template <int Dim> 
        void insert(const Ref<const Array<int, Dim, 1> >& value)
        {
            // Travel down the tree from the root, looking for the first 
            // letter at which the value deviates from the trie
            std::shared_ptr<TrieNode> curr = this->root; 
            for (int i = 0; i < Dim; ++i)
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node 
                    if (ptr->letter == value(i))
                    {
                        found_matching_child = true; 
                        curr = ptr; 
                        break;  
                    }
                }

                // If there is no such node, add new nodes to cover the 
                // remaining values
                if (!found_matching_child)
                {
                    for (int j = i; j < Dim; ++j)
                    {
                        // Define the new node and set its parent node  
                        TrieNode node; 
                        node.letter = value(j); 
                        node.parent = curr; 
                        std::shared_ptr<TrieNode> ptr = std::make_shared<TrieNode>(node);
                        this->nodes.push_back(node);

                        // Also update the children of its parent node  
                        curr->children.push_back(ptr);

                        // Move on to the next node 
                        curr = ptr;  
                    }
                    break; 
                }
            }
        }

        /**
         *
         */
        std::vector<std::vector<int> > getSubstrings(const int length = -1) const 
        {
            // Keep track of nodes to be visited using a queue (for BFS), 
            // starting with the root 
            std::queue<std::shared_ptr<TrieNode> > queue;
            queue.push(this->root); 
            int nvisited = 0;
            std::unordered_set<std::vector<int>, boost::hash<std::vector<int> > > substrings;
            std::vector<std::vector<int> > substrings_longer; 
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                std::shared_ptr<TrieNode> curr = queue.front(); 
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue 
                for (auto& ptr : curr->children) 
                    queue.push(ptr);

                // Collect the value of this node if it has the desired 
                // length or more
                std::vector<int> value = this->getValue(curr); 
                if (length == -1 || value.size() >= length)
                    substrings_longer.push_back(value); 
            }

            // Run through each string ...
            for (auto&& substring : substrings_longer)
            {
                // Collect all possible substrings of the string
                int n = substring.size();
                if (n == length)
                {
                    substrings.insert(substring);
                } 
                else    // n > length
                {
                    // If a particular length of substring is desired, only 
                    // run through the substrings of that length 
                    if (length != -1)
                    {
                        for (auto&& combination : getCombinations(substring, length))
                            substrings.insert(combination);
                    }
                    // Otherwise, get all possible substrings 
                    else 
                    {
                        for (auto&& substring2 : getPowerset(substring))
                            substrings.insert(substring2); 
                    } 
                }
            }

            // Sort the set of collected substrings
            std::vector<std::vector<int> > substrings_sorted; 
            for (auto&& substring : substrings)
                substrings_sorted.push_back(substring);
            std::sort(substrings_sorted.begin(), substrings_sorted.end());

            return substrings_sorted;  
        }

        /**
         *
         */
        std::vector<std::vector<int> > getSuperstrings(const std::vector<int>& value,
                                                       const int length = -1) const 
        {
            if (length != -1 && length < value.size())
                throw std::runtime_error("Invalid length specified"); 

            // Keep track of nodes to be visited using a queue (for BFS), 
            // starting with the root 
            std::queue<std::shared_ptr<TrieNode> > queue;
            queue.push(this->root); 
            int nvisited = 0;
            std::unordered_set<std::vector<int>, boost::hash<std::vector<int> > > superstrings; 
            std::vector<std::vector<int> > superstrings_longer; 
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                std::shared_ptr<TrieNode> curr = queue.front(); 
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue 
                for (auto& ptr : curr->children) 
                    queue.push(ptr);

                // Collect the value of this node if it has the desired 
                // length or more
                std::vector<int> curr_value = this->getValue(curr); 
                if (length == -1)
                {
                    if (curr_value.size() >= value.size())
                        superstrings_longer.push_back(curr_value); 
                }
                else 
                {
                    if (curr_value.size() == length)
                        superstrings_longer.push_back(curr_value); 
                }
            }

            // Run through each string ...
            for (auto&& superstring : superstrings_longer)
            {
                // Does this string contain the query string?
                std::unordered_set<int> set(superstring.begin(), superstring.end());
                bool contains = true; 
                for (int i = 0; i < value.size(); ++i)
                {
                    if (set.count(value[i]) == 0)
                    {
                        contains = false; 
                        break; 
                    }
                }
                if (contains)
                    superstrings.insert(superstring); 
            }

            // Sort the set of collected superstrings
            std::vector<std::vector<int> > superstrings_sorted; 
            for (auto&& superstring : superstrings)
                superstrings_sorted.push_back(superstring); 
            std::sort(superstrings_sorted.begin(), superstrings_sorted.end());

            return superstrings_sorted;
        }
};

/**
 * A simple implementation of a simplex tree, inspired by Gudhi::Simplex_tree<>,
 * for 3-D simplicial complexes. 
 */
template <typename T>
class SimplicialComplex3D
{
    private:
        Array<T, Dynamic, 3> points;     // Array of point coordinates
        int dim;                         // Dimension
        Trie tree;                       // Simplex tree  

    public:
        /**
         * Default constructor, which takes in arrays of points, edges, 
         * triangles, and tetrahedra. 
         *
         * @param points Array of point coordinates. 
         * @param edges Array of vertices connected by edges. 
         * @param triangles Array of vertices that form triangles. 
         * @param tetrahedra Array of vertices that form tetrahedra. 
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points,
                            const Ref<const Array<int, Dynamic, 2> >& edges,
                            const Ref<const Array<int, Dynamic, 3> >& triangles, 
                            const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
        {
            this->points = points; 
            
            // Construct the simplex tree, one simplex at a time 
            for (int i = 0; i < points.rows(); ++i)
                this->tree.insert(std::vector<int>({i}));
            for (int i = 0; i < edges.rows(); ++i)
                this->tree.template insert<2>(edges.row(i).transpose());
            for (int i = 0; i < triangles.rows(); ++i)
                this->tree.template insert<3>(triangles.row(i).transpose());
            for (int i = 0; i < tetrahedra.rows(); ++i)
                this->tree.template insert<4>(tetrahedra.row(i).transpose()); 
        }

        /**
         * Alternative constructor, which takes in an array of points and an 
         * input graph, and forms the corresponding alpha-complex (i.e., 
         * each 3-clique is "filled in" to form a triangle, and each 4-clique
         * is "filled in" to form a tetrahedron).
         *
         * @param points Array of point coordinates. 
         * @param graph Input graph. 
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points, 
                            Graph& graph)
        {
            this->points = points;

            // Construct the simplex tree, starting with the 0-simplices 
            for (int i = 0; i < points.rows(); ++i)
                this->tree.insert(std::vector<int>({i}));
            
            // Parse the edges from the graph
            std::pair<boost::graph_traits<Graph>::edge_iterator, 
                      boost::graph_traits<Graph>::edge_iterator> it;
            for (it = boost::edges(graph); it.first != it.second; ++it.first)
            {
                boost::graph_traits<Graph>::edge_descriptor edge = *(it.first); 
                int j = boost::source(edge, graph); 
                int k = boost::target(edge, graph);
                std::vector<int> edge_vec { j, k }; 
                this->tree.insert(edge_vec); 
            }

            // Define the triangles and tetrahedra from the 3- and 4-cliques 
            // in the graph (this gives the alpha-complex of the graph)
            Array<int, Dynamic, 3> triangles = ::getTriangles<T>(graph); 
            Array<int, Dynamic, 4> tetrahedra = ::getTetrahedra<T>(graph);

            // Add them to the simplex tree 
            for (int i = 0; i < triangles.rows(); ++i)
                this->tree.template insert<3>(triangles.row(i).transpose()); 
            for (int i = 0; i < tetrahedra.rows(); ++i)
                this->tree.template insert<4>(tetrahedra.row(i).transpose());
        }

        /**
         * Alternative constructor, which takes in an array of points and 
         * a pre-constructed simplex tree. 
         *
         * @param points Array of point coordinates. 
         * @param tree Input simplex tree.
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points, 
                            Trie& tree)
        {
            this->points = points;
            this->tree = tree; 
        }

        /**
         * Trivial destructor. 
         */
        ~SimplicialComplex3D()
        {
        }

        /**
         * Return the array of point coordinates. 
         *
         * @returns Array of point coordinates.  
         */
        Array<T, Dynamic, 3> getPoints() const
        {
            return this->points; 
        }

        /**
         * Return the dimension of the complex. 
         *
         * @returns Dimension of the complex. 
         */
        int dimension() const 
        {
            return this->tree.getHeight() - 1; 
        }

        /**
         * Return the simplices of the given dimension. 
         *
         * @returns Array of vertices specifying each simplex of the given 
         *          dimension. 
         */
        template <int Dim>
        Array<int, Dynamic, Dim + 1> getSimplices() const
        {
            // Get the simplices of the desired dimension 
            std::vector<std::vector<int> > simplices_ = this->tree.getSubstrings(Dim + 1);

            // Re-organize as an array and return  
            Array<int, Dynamic, Dim + 1> simplices(simplices_.size(), Dim + 1);
            int i = 0; 
            for (auto it1 = simplices_.begin(); it1 != simplices_.end(); ++it1)
            {
                int j = 0;
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
                {
                    simplices(i, j) = *it2;
                    j++;
                }
                i++;  
            } 
            return simplices;
        }

        /**
         * Update the simplicial complex with the given points, edges, 
         * triangles, and tetrahedra. 
         *
         * @param points Array of point coordinates. 
         * @param edges Array of vertices connected by edges. 
         * @param triangles Array of vertices that form triangles. 
         * @param tetrahedra Array of vertices that form tetrahedra. 
         */
        void update(const Ref<const Array<T, Dynamic, 3> >& points,
                    const Ref<const Array<int, Dynamic, 2> >& edges,
                    const Ref<const Array<int, Dynamic, 3> >& triangles, 
                    const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
        {
            this->points = points;
            this->tree.clear();  
            
            // Construct the simplex tree, one simplex at a time 
            for (int i = 0; i < points.rows(); ++i)
                this->tree.insert({i});
            for (int i = 0; i < edges.rows(); ++i)
                this->tree.template insert<2>(edges.row(i).transpose()); 
            for (int i = 0; i < triangles.rows(); ++i)
                this->tree.template insert<3>(triangles.row(i).transpose()); 
            for (int i = 0; i < tetrahedra.rows(); ++i)
                this->tree.template insert<4>(tetrahedra.row(i).transpose()); 
        }

        int getNumFullDimCofaces(std::vector<int>& simplex) const
        {
            return this->tree.getSuperstrings(simplex).size(); 
        }

        int getNumFullDimCofaces(const Ref<const Array<int, Dynamic, 1> >& simplex) const
        {
            std::vector<int> simplex_;
            for (int i = 0; i < simplex.size(); ++i)
                simplex_.push_back(simplex(i));

            return this->tree.getSuperstrings(simplex_).size(); 
        }

        /**
         * Return the topological boundary of the simplicial complex.
         *
         * This is the subcomplex of faces that do not feature as a subface
         * of 2 or more full-dimensional faces.
         *
         * @returns The boundary simplicial complex.  
         */
        SimplicialComplex3D<T> getBoundary() const 
        {
            int maxdim = this->dimension();
            Array<int, Dynamic, 1> points_in_boundary = Array<int, Dynamic, 1>::Zero(this->points.rows());
            Trie tree_boundary;

            // If the simplicial complex is 0-dimensional, return it as is 
            if (maxdim == 0)
            {
                Array<T, Dynamic, 3> bpoints(this->points.rows(), 3);
                for (int i = 0; i < this->points.rows(); ++i)
                {
                    bpoints.row(i) = this->points.row(i);
                    tree_boundary.insert({i}); 
                }
                return SimplicialComplex3D<T>(bpoints, tree_boundary); 
            } 

            // Run through all non-full-dimensional simplices in the tree ...
            for (int dim = 0; dim < maxdim; ++dim)
            {
                std::vector<std::vector<int> > simplices = this->tree.getSubstrings(dim + 1);
                for (int i = 0; i < simplices.size(); ++i)
                {
                    // ... and check the number of full-dimensional simplices that 
                    // have this simplex as a face 
                    std::vector<std::vector<int> > cofaces = this->tree.getSuperstrings(simplices[i], maxdim + 1); 

                    // If this number is 0 or 1, include as part of the boundary 
                    if (cofaces.size() < 2)
                    {
                        for (int j = 0; j < simplices[i].size(); ++j)
                            points_in_boundary(simplices[i][j]) = 1;
                        tree_boundary.insert(simplices[i]); 
                    } 
                } 
            }
          
            // Get the subset of points that lie within the boundary 
            Array<T, Dynamic, 3> bpoints(points_in_boundary.sum(), 3);
            int i = 0;
            for (int j = 0; j < points.rows(); ++j)
            {
                if (points_in_boundary(j))
                {
                    bpoints.row(i) = points.row(j); 
                    i++;
                }
            } 

            return SimplicialComplex3D<T>(bpoints, tree_boundary); 
        }

        /**
         * Get the given boundary homomorphism as a matrix.
         */
        template <int Dim>
        Matrix<T, Dynamic, Dynamic> getBoundaryHomomorphism() const
        {
            // If the dimension is zero, then this is the zero map
            if (Dim == 0)
                return Matrix<T, Dynamic, Dynamic>::Zero(1, this->points.rows()); 

            // If the dimension exceeds the maximal dimension of the complex,
            // then this is again the zero map
            const int maxdim = this->tree.getHeight() - 1;  
            if (Dim > maxdim)
                return Matrix<T, Dynamic, Dynamic>::Zero(0, 0);  

            // Get the k- and (k-1)-faces in the complex, where k = Dim, 
            // with all vertices sorted in ascending order
            std::vector<std::vector<int> > faces1 = this->tree.getSubstrings(Dim + 1);
            std::vector<std::vector<int> > faces2 = this->tree.getSubstrings(Dim);  

            // This is an m-by-n matrix, where m is the number of (k-1)-faces
            // and n is the number of k-faces
            const int n = faces1.size();
            const int m = faces2.size();
            Matrix<T, Dynamic, Dynamic> del = Matrix<T, Dynamic, Dynamic>::Zero(m, n);

            // Store the indices of the (k-1)-faces in the above ordering
            std::unordered_map<std::vector<int>, int, boost::hash<std::vector<int> > > indices; 
            for (int i = 0; i < m; ++i)
                indices[faces2[i]] = i;  

            // For each k-face ...
            for (int i = 0; i < n; ++i)
            {
                // Get all the (k-1)-dimensional faces of the k-face
                std::vector<std::vector<int> > subfaces = getCombinations(faces1[i], Dim);
                int j = 0; 
                for (auto& subface : subfaces)
                {
                    for (int k = 0; k < Dim; ++k)
                        std::cout << subface[k] << " ";
                    std::cout << std::endl; 
                    // The corresponding entry is the sign of the permutation
                    // (0, ..., j), which is (-1)^j
                    del(i, indices[subface]) = (j % 2 == 0 ? 1 : -1);
                    j++;  
                }
            }

            return del;
        } 

        /**
         * Get the combinatorial Laplacian. 
         */
        template <int Dim> 
        Matrix<T, Dynamic, Dynamic> getCombinatorialLaplacian() const
        {
            // Get the number of simplices of the given dimension 
            std::vector<std::vector<int> > simplices = this->tree.getSubstrings(Dim + 1); 
            int ndim = simplices.size();  
            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(ndim, ndim);

            // Populate the matrix ... 
            for (int i = 0; i < ndim; ++i)
            {
                for (int j = 0; j < ndim; ++j)
                {
                    // If i == j, then set the entry to the number of
                    // (Dim + 1)-simplices that contain this simplex as 
                    // a face 
                    if (i == j)
                    {
                        std::vector<std::vector<int> > cofaces = this->tree.getSuperstrings(simplices[i], Dim + 2); 
                        laplacian(i, j) = cofaces.size(); 
                    }
                    // Otherwise, set the entry to -1 if the two simplices
                    // share a common coface
                    else 
                    {
                        // Run through the cofaces of simplex i ...
                        bool found_common_coface = false;  
                        for (auto& simplex1 : this->tree.getSuperstrings(simplices[i], Dim + 2))
                        {
                            // Run through the cofaces of simplex j ... 
                            for (auto& simplex2 : this->tree.getSuperstrings(simplices[j], Dim + 2))
                            {
                                bool match = true; 
                                for (int k = 0; k < Dim + 2; ++k)
                                {
                                    if (simplex1[k] != simplex2[k])
                                        match = false; 
                                }
                                if (match)
                                {
                                    laplacian(i, j) = -1;
                                    found_common_coface = true; 
                                    break; 
                                }
                            }
                            if (found_common_coface)
                                break; 
                        }
                    } 
                }
            }

            return laplacian; 
        }

        /**
         * Get the vector of Betti numbers for the simplicial complex. 
         *
         * @param p Characteristic of coefficient field. 
         * @returns Betti numbers of the simplicial complex. 
         */
        Array<int, Dynamic, 1> getBettiNumbers(const int p = 2)
        {
            /*
            // Compute the homology groups of the simplicial complex in the
            // given coefficient field  
            Persistent_cohomology pcoh(this->tree, true);
            pcoh.init_coefficients(p); 
            pcoh.compute_persistent_cohomology();

            // Collect the Betti numbers 
            Array<int, Dynamic, 1> betti(4);
            for (int i = 0; i < 4; ++i)
                betti(i) = pcoh.betti_number(i); 
            
            return betti; 
            */
            return Array<int, Dynamic, 1>::Zero(4);  
        }

        /**
         * Get a collection of representative cycles for the simplicial 
         * complex.
         *
         * @param p Characteristic of coefficient field. 
         * @returns Collection of representative cycles.  
         */
        /*
        void getRepresentativeCycles(const int p = 2)
        {
            // Reset the filtration 
            this->tree.clear_filtration();  
            
            // Define an chain matrix decomposition
            Chain_matrix chain(this->tree.num_simplices());

            // Assign keys to each simplex 
            int simplex_id = 0;
            std::vector<std::vector<int> > simplices_sorted; 
            for (int dim = 0; dim <= this->tree.dimension(); ++dim)
            {
                for (auto& simplex : this->tree.skeleton_simplex_range(dim))
                {
                    if (this->tree.dimension(simplex) == dim)
                    {
                        this->tree.assign_key(simplex, simplex_id);
                        std::vector<int> simplex_vertices; 
                        for (auto vertex : this->tree.simplex_vertex_range(simplex))
                            simplex_vertices.push_back(vertex); 
                        simplex_id++;
                        simplices_sorted.push_back(simplex_vertices); 
                    } 
                }
            }

            // Initialize the trivial filtration
            this->tree.initialize_filtration(); 

            // For each simplex ...
            std::vector<std::vector<int> > boundaries; 
            for (auto it = simplices_sorted.begin(); it != simplices_sorted.end(); ++it)
            {
                auto simplex = this->tree.find(*it); 

                // Collect the boundary faces of this simplex
                std::vector<int> boundary; 
                for (auto& bd : this->tree.boundary_simplex_range(simplex))
                    boundary.push_back(this->tree.key(bd));
                std::sort(boundary.begin(), boundary.end());
                boundaries.push_back(boundary);
                
                // Insert into the matrix
                chain.insert_boundary(boundary);
            }

            // Get representative cycles 
            auto cycles = chain.get_representative_cycles();
            for (auto& cycle : cycles)
            {
                std::cout << chain.get_column_dimension(cycle[0]) << "-cycle: ";
                for (auto index : cycle)
                {
                    std::cout << "[" << index << ":";
                    auto simplex = this->tree.find(simplices_sorted[index]);  
                    for (auto vertex : this->tree.simplex_vertex_range(simplex))
                        std::cout << vertex << ",";
                    std::cout << "];";
                }
                std::cout << std::endl; 
            }
        }
        */
};

/**
 * Alpha-wrapping. 
 */
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel; 
using Point_3 = K::Point_3; 
using Mesh = CGAL::Surface_mesh<Point_3>; 

template <typename T>
Mesh getAlphaWrapping(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                      const T R)
{
    // Get the center and endpoints of every cell 
    Array<double, Dynamic, 3> points(3 * cells.rows(), 3);
    for (int i = 0; i < cells.rows(); ++i)
    {
        Array<T, 3, 1> r = cells(i, __colseq_r); 
        Array<T, 3, 1> p = r - cells(i, __colidx_half_l) * cells(i, __colseq_n);
        Array<T, 3, 1> q = r + cells(i, __colidx_half_l) * cells(i, __colseq_n); 
        points.row(3 * i) = p.template cast<double>();
        points.row(3 * i + 1) = r.template cast<double>(); 
        points.row(3 * i + 2) = q.template cast<double>();  
    }

    // Get the alpha-wrapping
    const double offset = R; 
    const double alpha = R / 2; 
    Mesh wrap; 
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    std::cout << "Done with alpha-wrapping: " 
              << CGAL::num_vertices(wrap) << " vertices, " 
              << CGAL::num_faces(wrap) << " faces\n";

    return wrap;  
}

#endif
