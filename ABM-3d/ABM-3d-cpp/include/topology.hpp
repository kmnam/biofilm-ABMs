/**
 * Classes and functions for graphs and simplicial complexes. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/10/2025
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
#include <CGAL/Gmpz.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include "distances.hpp"
#include "mechanics.hpp"
#include "fields.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::property<boost::edge_weight_t, double> EdgeProperty; 
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> Graph;
typedef CGAL::Gmpz ET; 
typedef CGAL::Quadratic_program<int> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution; 

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

/** ------------------------------------------------------------------- //
 *                                 TRIES                                //
 *  ------------------------------------------------------------------- */
/**
 * A simple implementation of a trie. 
 */
struct TrieNode
{
    int letter;
    int level;      // Length of string corresponding to the node  
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
        std::vector<int> getString(const std::shared_ptr<TrieNode>& ptr) const 
        {
            // Travel up the trie from the node to get the value 
            std::vector<int> string;
            std::shared_ptr<TrieNode> curr = ptr;  
            while (curr != this->root)
            {
                string.insert(string.begin(), curr->letter); 
                curr = curr->parent; 
            }

            return string; 
        } 

    public:
        /**
         * Initialize a trie with -1 at the root. 
         */
        Trie()
        {
            TrieNode node;
            node.letter = -1;
            node.level = 0; 
            node.parent = nullptr;
            this->nodes.push_back(node);
            this->root = std::make_shared<TrieNode>(node); 
        }

        /**
         * Trivial destructor. 
         */
        ~Trie()
        {
        }

        /**
         * Return true if the trie contains a node for the full string/value.
         */
        bool containsString(std::vector<int> string) const 
        {
            // Travel down the tree from the root, looking for a node that 
            // contains each successive entry in the string
            std::shared_ptr<TrieNode> curr = this->root;
            for (int i = 0; i < string.size(); ++i)
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node 
                    if (ptr->letter == string[i])
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
         *
         * The result is not an empty trie, but rather a one-node trie. 
         */
        void clear()
        {
            this->nodes.clear();
            TrieNode node;
            node.letter = -1;
            node.level = 0; 
            node.parent = nullptr;
            this->nodes.push_back(node);
            this->root = std::make_shared<TrieNode>(node); 
        }

        /**
         * Get the number of nodes in the trie.
         *
         * @returns Number of nodes in the trie.  
         */
        int getNumNodes() const 
        {
            return this->nodes.size(); 
        }

        /**
         * Get the height of the trie (defined as the length of the longest
         * string).
         *
         * @returns Height of the trie. 
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

                // Get the string stored in this node
                std::vector<int> string = this->getString(curr);
                if (height < string.size())
                    height = string.size();  
            }

            return height; 
        }

        /**
         * Insert the given string/value into the trie.
         *
         * @param string Input string. 
         */ 
        void insert(std::vector<int> string)
        {
            // Travel down the tree from the root, looking for the first 
            // letter at which the string deviates from the trie
            std::shared_ptr<TrieNode> curr_ptr = this->root;
            int curr_idx = 0;
            int first_mismatch = -1;  
            while (curr_idx < string.size()) 
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr_ptr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node
                    if (ptr->letter == string[curr_idx])
                    {
                        found_matching_child = true; 
                        curr_ptr = ptr;
                        break;  
                    }
                }

                // Continue looping through the string if a matching child
                // node was found, but break if not 
                if (!found_matching_child)
                {
                    first_mismatch = curr_idx;
                    break;
                }
                else 
                {
                    curr_idx++; 
                }
            }

            // If only a substring of the string is in the trie, add new
            // nodes to cover the remaining letters
            if (first_mismatch != -1)
            {
                for (int j = first_mismatch; j < string.size(); ++j)
                {
                    // Define the new node and set its parent node  
                    TrieNode node; 
                    node.letter = string[j];
                    node.level = j + 1; 
                    node.parent = curr_ptr; 
                    std::shared_ptr<TrieNode> ptr = std::make_shared<TrieNode>(node);
                    this->nodes.push_back(node);

                    // Also update the children of its parent node  
                    curr_ptr->children.push_back(ptr);

                    // Move on to the next node 
                    curr_ptr = ptr;  
                }
            }
        }

        /**
         * Insert the string/value into the trie. 
         */
        template <int Dim> 
        void insert(const Ref<const Array<int, Dim, 1> >& string)
        {
            // Travel down the tree from the root, looking for the first 
            // letter at which the string deviates from the trie
            std::shared_ptr<TrieNode> curr_ptr = this->root;
            int curr_idx = 0;
            int first_mismatch = -1;
            while (curr_idx < string.size()) 
            {
                // For each child node ... 
                std::vector<std::shared_ptr<TrieNode> > children = curr_ptr->children;
                bool found_matching_child = false; 
                for (auto& ptr : children)
                {
                    // ... check that the letter matches the corresponding
                    // letter for the child node 
                    if (ptr->letter == string(curr_idx))
                    {
                        found_matching_child = true; 
                        curr_ptr = ptr; 
                        break;  
                    }
                }

                // Continue looping through the string if a matching child
                // node was found, but break if not 
                if (!found_matching_child)
                {
                    first_mismatch = curr_idx;
                    break;
                }
                else 
                {
                    curr_idx++; 
                }
            }

            // If only a substring of the string is in the trie, add new
            // nodes to cover the remaining letters
            if (first_mismatch != -1)
            {
                for (int j = first_mismatch; j < Dim; ++j)
                {
                    // Define the new node and set its parent node  
                    TrieNode node; 
                    node.letter = string(j);
                    node.level = j + 1; 
                    node.parent = curr_ptr; 
                    std::shared_ptr<TrieNode> ptr = std::make_shared<TrieNode>(node);
                    this->nodes.push_back(node);

                    // Also update the children of its parent node  
                    curr_ptr->children.push_back(ptr);

                    // Move on to the next node 
                    curr_ptr = ptr;  
                }
            }
        }

        /**
         * Returns a vector of all the strings stored in the trie.
         *
         * @param sort If true, sort in lexicographic order.
         * @param nonempty If true, skip over the empty string. 
         * @param length If nonnegative, return only the strings with this 
         *               length. 
         * @returns All strings stored in the trie. 
         */
        std::vector<std::vector<int> > getStrings(const bool sort = true,
                                                  const bool nonempty = true,
                                                  const int length = -1) const 
        {
            // Keep track of nodes to be visited using a queue (for BFS), 
            // starting with the root 
            std::queue<std::shared_ptr<TrieNode> > queue;
            queue.push(this->root); 
            int nvisited = 0;
            std::vector<std::vector<int> > strings; 
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                std::shared_ptr<TrieNode> curr_ptr = queue.front();
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue
                //
                // Skip over this if the current node has level less than 
                // the desired length, in which case we don't need to traverse
                // further down the trie 
                if (length == -1 || curr_ptr->level < length)
                { 
                    for (auto& ptr : curr_ptr->children) 
                        queue.push(ptr);
                }

                // Collect the string corresponding to this node if it has
                // the desired length
                int curr_length = curr_ptr->level;
                if (length == -1)
                {
                    if (!nonempty || curr_length > 0)
                        strings.push_back(this->getString(curr_ptr)); 
                }
                else if (curr_length == length)
                {
                    strings.push_back(this->getString(curr_ptr)); 
                } 
            }

            // Sort in lexicographical order and return 
            //
            // Note that neither DFS nor BFS are guaranteed to return the
            // strings in this order, since the nodes in each level may not
            // be properly ordered
            if (sort) 
                std::sort(strings.begin(), strings.end());

            return strings;  
        }

        /**
         * Returns a vector of all substrings of the strings stored in the
         * trie.
         *
         * A substring of a string (s1, ..., sN) may be any subset of the 
         * string that respects the ordering, but need not be contiguous. 
         * Therefore, (s1, s3) is a substring, but (s2, s1) is not. 
         *
         * @param nonempty If true, skip over the empty string. 
         * @param length If nonnegative, return only the strings with this 
         *               length.
         * @returns All strings stored in the trie. 
         */
        std::vector<std::vector<int> > getSubstrings(const bool nonempty = true,
                                                     const int length = -1) const 
        {
            // First get all strings in the trie
            std::vector<std::vector<int> > strings = this->getStrings(false, nonempty, -1);

            // For each string ... 
            std::unordered_set<std::vector<int>, boost::hash<std::vector<int> > > substrings;
            for (auto&& string : strings)
            {
                // Collect all possible substrings of the string
                int n = string.size();
                if (n == length)
                {
                    substrings.insert(string);
                } 
                else if (n > length)
                {
                    // If a particular length of substring is desired, only 
                    // run through the substrings of that length 
                    if (length != -1)
                    {
                        for (auto&& combination : getCombinations(string, length))
                            substrings.insert(combination);
                    }
                    // Otherwise, get all possible substrings 
                    else 
                    {
                        for (auto&& substring : getPowerset(string, nonempty))
                            substrings.insert(substring);
                    } 
                }
            }

            // Sort the substrings in lexicographic order 
            std::vector<std::vector<int> > substrings_sorted(
                substrings.begin(), substrings.end()
            );
            std::sort(substrings_sorted.begin(), substrings_sorted.end());

            return substrings_sorted;  
        }

        /**
         * Returns a vector of all strings stored in the trie that contain 
         * the given substring. 
         *
         * A substring of a string (s1, ..., sN) may be any subset of the 
         * string that respects the ordering, but need not be contiguous. 
         * Therefore, (s1, s3) is a substring, but (s2, s1) is not.
         */
        std::vector<std::vector<int> > getSuperstrings(const std::vector<int>& substring,
                                                       const int length = -1) const 
        {
            // If the length is given, it must be at least the length of 
            // the substring
            if (length != -1 && length < substring.size())
                throw std::runtime_error("Invalid length specified");

            // First get all strings in the trie
            std::vector<std::vector<int> > strings = this->getStrings(false, true, -1);

            // For each string ... 
            std::vector<std::vector<int> > superstrings;
            for (auto&& string : strings)
            {
                // If the string has the desired length ... 
                if ((length != -1 && string.size() == length) ||
                    (length == -1 && string.size() >= substring.size()))
                {
                    // Run through the string ... 
                    std::vector<int> idx; 
                    int i = 0;    // Index of current character in substring
                    int j = 0;    // Index of current character in string
                    while (i < substring.size() && j < string.size())
                    {
                        // Is the j-th character in the string the same as 
                        // the i-th character in the substring? 
                        if (string[j] == substring[i])
                            i++;
                        j++; 
                    }
                    if (i == substring.size())
                        superstrings.push_back(string);
                } 
            }

            // Sort the superstrings in lexicographic order 
            std::sort(superstrings.begin(), superstrings.end());

            return superstrings;
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
         * Return the number of points. 
         *
         * @returns Number of points. 
         */
        int getNumPoints() const 
        {
            return this->points.rows(); 
        }

        /**
         * Return the dimension of the complex.
         *
         * This is the dimension of the highest-dimensional simplex in 
         * the complex, which is one minus the length of the longest
         * tuple/string in the trie.
         *
         * @returns Dimension of the complex. 
         */
        int dimension() const 
        {
            return this->tree.getHeight() - 1; 
        }

        /**
         * Return the number of simplices of the given dimension. 
         *
         * An exception is thrown if the given dimension exceeds the 
         * dimension of the complex. 
         *
         * @returns Number of simplices of the given dimension. 
         */
        int getNumSimplices(const int dim = -1) const 
        {
            if (dim > this->dimension())
                throw std::runtime_error(
                    "Input dimension exceeds maximum dimension"
                );
            
            // Count the number of simplices of the given dimension, which
            // is the number of substrings of length (dim + 1)
            //
            // If dim == -1, then count the total number of simplices 
            if (dim == -1)
                return this->tree.getSubstrings(true).size();
            else 
                return this->tree.getSubstrings(true, dim + 1).size();  
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
            //
            // These are the tuples/strings with length (Dim + 1) 
            std::vector<std::vector<int> > simplices_ = this->tree.getSubstrings(
                true, Dim + 1
            );

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
            Array<int, Dynamic, 1> boundary_indicator
                = Array<int, Dynamic, 1>::Zero(this->points.rows());
            Array<T, Dynamic, 3> boundary_points; 
            Trie boundary_tree;

            // If the simplicial complex is 0-dimensional (there are only 
            // isolated points), return it as is 
            if (maxdim == 0)
            {
                boundary_indicator = Array<int, Dynamic, 1>::Ones(this->points.rows());
                for (int i = 0; i < this->points.rows(); ++i)
                    boundary_tree.insert({i}); 
            } 
            else     // Otherwise ... 
            {
                // Get all simplices in the complex 
                std::vector<std::vector<int> > simplices = this->tree.getSubstrings(true); 

                // Run through all non-full-dimensional simplices in the tree ...
                for (auto&& simplex : simplices)
                {
                    int dim = simplex.size() - 1; 
                    if (dim < maxdim)
                    {
                        // Generate all full-dimensional simplices that have 
                        // this simplex as a face 
                        std::vector<std::vector<int> > cofaces
                            = this->tree.getSuperstrings(simplex, maxdim + 1);

                        // If this number is 0 or 1, include as part of the 
                        // boundary 
                        if (cofaces.size() < 2)
                        {
                            for (int i = 0; i < simplex.size(); ++i)
                                boundary_indicator(simplex[i]) = 1;
                            boundary_tree.insert(simplex); 
                        } 
                    }
                }
            }
          
            // Get the subset of points that lie within the boundary 
            boundary_points.resize(boundary_indicator.sum(), 3); 
            int i = 0;
            for (int j = 0; j < this->points.rows(); ++j)
            {
                if (boundary_indicator(j))
                {
                    boundary_points.row(i) = this->points.row(j); 
                    i++;
                }
            } 

            return SimplicialComplex3D<T>(boundary_points, boundary_tree); 
        }

        /**
         * Return the given boundary homomorphism, which (following Munkres,
         * Elements of Algebraic Topology) calculates the boundaries of
         * simplices of the given dimension as a linear combination of
         * simplices of the given dimension minus one.
         *
         * @param dim Input dimension.
         * @returns Boundary homomorphism from the group of (dim)-chains to 
         *          to the group of (dim - 1)-chains. 
         */
        template <int p = 0>
        Matrix<Fp<p>, Dynamic, Dynamic> getBoundaryHomomorphism(const int dim) const 
        {
            // If the dimension is zero or greater than the maximum dimension,
            // then raise an exception 
            if (dim == 0 || dim > this->dimension())
                throw std::runtime_error(
                    "Invalid input dimension for boundary homomorphism"
                );

            // Get a sorted list of the simplices with the appropriate dimensions
            std::vector<std::vector<int> > faces1 = this->tree.getSubstrings(true, dim + 1);
            std::vector<std::vector<int> > faces2 = this->tree.getSubstrings(true, dim);
            const int n1 = faces1.size(); 
            const int n2 = faces2.size();  

            // Initialize the matrix with the appropriate dimensions
            Matrix<Fp<p>, Dynamic, Dynamic> del = Matrix<Fp<p>, Dynamic, Dynamic>::Zero(n2, n1); 

            // Store the indices of the output faces in the above ordering as 
            // a dictionary 
            std::unordered_map<std::vector<int>, int, boost::hash<std::vector<int> > > indices; 
            for (int i = 0; i < n2; ++i)
                indices[faces2[i]] = i;  

            // For each face in the domain ...
            for (int i = 0; i < n1; ++i)
            {
                std::vector<int> face = faces1[i]; 

                // Get all codimension-one subfaces
                //
                // Each face in the domain consists of dim + 1 points, so 
                // each subface arises from excluding each point 
                for (int j = 0; j < dim + 1; ++j)
                {
                    std::vector<int> subface; 
                    for (int k = 0; k < dim + 1; ++k)
                    {
                        if (j != k)
                            subface.push_back(face[k]); 
                    }

                    // Get the index of the subface in the ordering 
                    int idx = indices[subface]; 

                    // The corresponding entry in the matrix is (-1)^j
                    del(idx, i) = Fp<p>(j % 2 == 0 ? 1 : -1);   
                }
            }

            return del;
        } 

        /**
         * Returns the combinatorial Laplacian.
         *
         * This function assumes that the underlying field is the rationals
         * (characteristic zero). 
         *
         * @param dim Input dimension.
         * @returns Combinatorial Laplacian matrix.  
         */
        Matrix<Fp<0>, Dynamic, Dynamic> getCombinatorialLaplacian(const int dim) const 
        {
            // Get the boundary homomorphisms 
            //
            // If dim equals the dimension of the complex, then set del1 = 0 
            //
            // Similarly, if dim == 0, then set del2 = 0
            const int maxdim = this->dimension();
            Matrix<Fp<0>, Dynamic, Dynamic> lap;  
            if (dim < 0 || dim > this->dimension())
            {
                throw std::runtime_error(
                    "Invalid input dimension for combinatorial Laplacian"
                ); 
            }
            else if (dim == 0 && maxdim == 0)
            {
                const int n = this->points.rows(); 
                return Matrix<Fp<0>, Dynamic, Dynamic>::Zero(n, n); 
            }
            else if (dim == 0)        // maxdim != 0
            {
                Matrix<Fp<0>, Dynamic, Dynamic> del1 = this->getBoundaryHomomorphism<0>(1);
                lap = del1 * del1.transpose(); 
            }
            else if (dim == maxdim)   // dim, maxdim != 0
            {
                Matrix<Fp<0>, Dynamic, Dynamic> del2 = this->getBoundaryHomomorphism<0>(dim);
                lap = del2.transpose() * del2; 
            }
            else     // Otherwise, get both boundary homomorphisms  
            { 
                Matrix<Fp<0>, Dynamic, Dynamic> del1 = this->getBoundaryHomomorphism<0>(dim + 1); 
                Matrix<Fp<0>, Dynamic, Dynamic> del2 = this->getBoundaryHomomorphism<0>(dim); 

                // Note that del1 has shape (n2, n1) and del2 has shape (n3, n2),
                // where n1 = # (dim + 1)-simplices, n2 = # (dim)-simplices, 
                // n3 = # (dim - 1)-simplices
                //
                // Therefore, the Laplacian has shape (n2, n2)
                lap = del1 * del1.transpose() + del2.transpose() * del2;
            }
            
            return lap; 
        }

        /**
         * Get a basis of cycles for the homology group over rational
         * coefficients of the given dimension. 
         *
         * @param dim Input dimension.
         * @returns Basis of cycles for the homology group. 
         */
        Matrix<Fp<0>, Dynamic, Dynamic> getZeroCharHomology(const int dim) const 
        {
            return ::kernel<Fp<0> >(this->getCombinatorialLaplacian(dim)); 
        }

        /**
         * Get a basis of cycles for the homology group over a field of 
         * characteristic p > 0 of the given dimension.
         *
         * This function should also work with p == 0, but is less 
         * efficient that computing the kernel of the combinatorial
         * Laplacian. 
         *
         * @param dim Input dimension.
         * @returns Basis of cycles for the homology group. 
         */
        template <int p = 2>
        Matrix<Fp<p>, Dynamic, Dynamic> getPrimeCharHomology(const int dim) const 
        {
            Matrix<Fp<p>, Dynamic, Dynamic> cycles;
            if (dim > 0 && dim < this->dimension())
            {
                // Get the boundary homomorphisms and the corresponding
                // quotient space basis 
                Matrix<Fp<p>, Dynamic, Dynamic> del1 = this->getBoundaryHomomorphism<p>(dim + 1);
                Matrix<Fp<p>, Dynamic, Dynamic> del2 = this->getBoundaryHomomorphism<p>(dim);
                cycles = ::quotientSpace<Fp<p> >(del2, del1); 
            }
            else if (dim == 0)
            {
                // In this case, the kernel of the 0-th boundary homomorphism
                // is all of F^d, where F is the field and d is the number 
                // of 0-simplices
                if (this->dimension() == 0)
                {
                    // In this case, the image of the 1st boundary homomorphism
                    // is zero, so the quotient is all of F^d
                    int n = this->points.rows(); 
                    cycles = Matrix<Fp<p>, Dynamic, Dynamic>::Identity(n, n); 
                }
                else 
                {
                    // In this case, the image of the 1st boundary homomorphism 
                    // may be nontrivial, and the quotient is F^d modulo the image
                    Matrix<Fp<p>, Dynamic, Dynamic> del1 = this->getBoundaryHomomorphism<p>(dim + 1);
                    cycles = ::quotientSpace<Fp<p> >(del1);
                } 
            }
            else    // dim != 0 and dim == this->dimension() 
            {
                // In this case, the image of the (dim + 1)-th boundary 
                // homomorphism is zero
                Matrix<Fp<p>, Dynamic, Dynamic> del2 = this->getBoundaryHomomorphism<p>(dim);
                cycles = ::kernel<Fp<p> >(del2);
            }

            return cycles; 
        }

        /**
         * Get the vector of Betti numbers over rational coefficients for
         * the simplicial complex. 
         *
         * @returns Betti numbers of the simplicial complex. 
         */
        Array<int, Dynamic, 1> getZeroCharBettiNumbers() const 
        {
            Array<int, Dynamic, 1> betti = Array<int, Dynamic, 1>::Zero(4);
            const int maxdim = this->dimension(); 

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (maxdim == 0)
            {
                betti(0) = this->points.rows(); 
                return betti; 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= maxdim; ++i)
                {
                    Matrix<Fp<0>, Dynamic, Dynamic> cycles = this->getZeroCharHomology(i); 
                    betti(i) = cycles.cols(); 
                }
                return betti; 
            }
        }

        /**
         * Get the vector of Betti numbers over a field of characteristic 
         * p > 0 for the simplicial complex. 
         *
         * @returns Betti numbers of the simplicial complex. 
         */
        template <int p = 2>
        Array<int, Dynamic, 1> getPrimeCharBettiNumbers() const 
        {
            Array<int, Dynamic, 1> betti = Array<int, Dynamic, 1>::Zero(4);
            const int maxdim = this->dimension(); 

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (maxdim == 0)
            {
                betti(0) = this->points.rows(); 
                return betti; 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= maxdim; ++i)
                {
                    Matrix<Fp<p>, Dynamic, Dynamic> cycles = this->getPrimeCharHomology<p>(i); 
                    betti(i) = cycles.cols(); 
                }
                return betti; 
            }
        }

        /**
         * Return true if the input chain is a cycle of the given dimension. 
         *
         * @param chain Input chain, as a vector of coefficients over the 
         *              simplices of the appropriate dimension. 
         * @param dim Cycle dimension.
         * @returns True if the chain is a cycle, false otherwise.  
         */
        template <int p = 2>
        bool isCycle(const Ref<const Matrix<Fp<p>, Dynamic, 1> >& chain, 
                     const int dim)
        {
            // Check that the dimension of the input vector is correct
            if (dim < 0 || dim > this->dimension())
                throw std::runtime_error("Invalid input dimension"); 
            else if (chain.size() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // Is the vector in the kernel of the boundary homomorphism? 
            if (dim > 0)
            {
                Matrix<Fp<p>, Dynamic, Dynamic> del = this->getBoundaryHomomorphism<p>(dim); 
                return ((del * chain).array() == 0).all(); 
            } 
            else    // If dim == 0, then all vectors are in the kernel 
            {
                return true; 
            }
        }

        /**
         * Return true if the input chain is a boundary of the given dimension.
         *
         * @param chain Input chain, as a vector of coefficients over the 
         *              simplices of the appropriate dimension. 
         * @param dim Boundary dimension (i.e., dimension of the boundary
         *            homomorphism minus 1). 
         * @returns True if the chain is a boundary, false otherwise.  
         */
        template <int p = 2>
        bool isBoundary(const Ref<const Matrix<Fp<p>, Dynamic, 1> >& chain, 
                        const int dim)
        {
            // Check that the dimension of the input vector is correct
            if (dim < 0 || dim > this->dimension())
                throw std::runtime_error("Invalid input dimension"); 
            else if (chain.size() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // Is the vector in the image of the boundary homomorphism? 
            if (dim < this->dimension())
            {
                // Get the row echelon form of [del | chain]
                Matrix<Fp<p>, Dynamic, Dynamic> del = this->getBoundaryHomomorphism<p>(dim + 1); 
                Matrix<Fp<p>, Dynamic, Dynamic> system(del.rows(), del.cols() + 1); 
                system(Eigen::all, Eigen::seq(0, del.cols() - 1)) = del;
                system.col(del.cols()) = chain; 
                system = ::rowEchelonForm<Fp<p> >(system);

                // If there is an inconsistency in the row echelon form, this
                // means that the chain is not in the column space of del and 
                // is therefore not a boundary  
                for (int i = 0; i < system.rows(); ++i)
                {
                    if ((system.row(i).head(del.cols()).array() == 0).all() &&
                        system(i, del.cols()) != 0)
                    {
                        return false; 
                    }
                }
                return true;  
            } 
            else    // If dim is maximal, then only the zero vector is in the image
            {
                return (chain.array() == 0).all(); 
            }
        }

        /**
         * Return true if the two cycles (1) are indeed cycles, and (2) are 
         * homologous. 
         *
         * @param chain1 First input chain, as a vector of coefficients over
         *               the simplices of the appropriate dimension. 
         * @param chain2 Second input chain, as a vector of coefficients over
         *               the simplices of the appropriate dimension.
         * @param dim Cycle dimension. 
         * @returns True if the two chains are homologous, false otherwise. 
         */
        template <int p = 2>
        bool areHomologousCycles(const Ref<const Matrix<Fp<p>, Dynamic, 1> >& chain1, 
                                 const Ref<const Matrix<Fp<p>, Dynamic, 1> >& chain2, 
                                 const int dim)
        {
            return (
                this->isCycle<p>(chain1, dim) &&
                this->isCycle<p>(chain2, dim) &&
                this->isBoundary<p>(chain1 - chain2, dim)
            ); 
        }

        /**
         * Get a collection of minimal cycles for the simplicial complex 
         * over rational coefficients.  
         *
         * @param dim Input dimension. 
         * @returns Collection of minimal cycles. 
         */
        template <int p = 2>
        Matrix<double, Dynamic, Dynamic> minimizeCycles(const int dim)
        {
            if (dim < 0 || dim > this->dimension())
            {
                throw std::runtime_error(
                    "Invalid input dimension for minimal cycle calculation"
                ); 
            }
            Matrix<Fp<p>, Dynamic, Dynamic> cycles = this->getPrimeCharHomology<p>(dim);
            Matrix<double, Dynamic, Dynamic> opt_cycles(cycles.rows(), cycles.cols()); 

            // If the input dimension is maximal, then optimization is not 
            // necessary
            if (dim == this->dimension())
            {
                for (int i = 0; i < cycles.rows(); ++i)
                {
                    for (int j = 0; j < cycles.cols(); ++j)
                    {
                        opt_cycles(i, j) = static_cast<double>(cycles(i, j).value); 
                    }
                } 
            }
            // Otherwise ... 
            else 
            {
                // Get the boundary homomorphism from the (dim + 1)-th chain group
                Matrix<Fp<p>, Dynamic, Dynamic> del = this->getBoundaryHomomorphism<p>(dim + 1);
                const int n1 = del.cols();    // Number of (dim + 1)-simplices
                const int n2 = del.rows();    // Number of (dim)-simplices

                // For each cycle, solve the corresponding linear programming
                // problem 
                //
                // We follow the approach of Obayashi, SIAM J Appl Algebra
                // Geometry (2018) (see Eqn. 9)
                for (int j = 0; j < cycles.cols(); ++j)
                {
                    // We must minimize the 1-norm of z = z1 + \del w, where 
                    // z1 is the basic cycle and w is any (dim + 1)-chain
                    Matrix<Fp<p>, Dynamic, 1> cycle = cycles.col(j); 

                    // Define the linear program in standard format, and 
                    // solve for the minimal cycle
                    Matrix<double, Dynamic, Dynamic> A(n2, n1 + n2);
                    Matrix<double, Dynamic, 1> b(n2);
                    A(Eigen::all, Eigen::seq(0, n2 - 1))
                        = Matrix<double, Dynamic, Dynamic>::Identity(n2, n2);
                    for (int i = 0; i < n2; ++i)
                    {
                        for (int j = n2; j < n1 + n2; ++j)
                        {
                            A(i, j) = -static_cast<double>(del(i, j - n2).value); 
                        }
                        b(i) = static_cast<double>(cycle(i).value); 
                    }
                    CGAL::Quadratic_program<int> lp = defineL1LinearProgram(n2, A, b); 
                    Solution solution = CGAL::solve_linear_program(lp, ET());
                    int i = 0; 
                    for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
                    {
                        opt_cycles(i, j) = CGAL::to_double(*it);  
                        i++;
                        if (i == n2)
                            break; 
                    }
                }
            }
            
            return opt_cycles; 
        }
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
