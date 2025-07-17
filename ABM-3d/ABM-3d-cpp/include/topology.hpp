/**
 * Classes and functions for tries and simplicial complexes. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/17/2025
 */

#ifndef SIMPLICIAL_COMPLEXES_3D_HPP
#define SIMPLICIAL_COMPLEXES_3D_HPP

#include <iostream>
#include <memory>
#include <utility>
#include <tuple>
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
#include <CGAL/Gmpz.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include "fields.hpp"
#include "graphs.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::property<boost::edge_weight_t, double> EdgeProperty; 
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> Graph;
typedef CGAL::Gmpz ET; 
typedef CGAL::Quadratic_program<int> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution; 

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
        Graph one_skeleton;              // 1-skeleton as a Boost::Graph object  

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

            // Populate the 1-skeleton (with uniform weights) 
            for (int i = 0; i < points.rows(); ++i)
                boost::add_vertex(this->one_skeleton); 
            for (int i = 0; i < edges.rows(); ++i)
                boost::add_edge(edges(i, 0), edges(i, 1), EdgeProperty(1.0), this->one_skeleton); 
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
            for (int i = 0; i < points.rows(); ++i)
                boost::add_vertex(this->one_skeleton); 

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
                boost::add_edge(j, k, EdgeProperty(1.0), this->one_skeleton); 
            }

            // Define the triangles and tetrahedra from the 3- and 4-cliques 
            // in the graph (this gives the alpha-complex of the graph)
            Array<int, Dynamic, 3> triangles = ::getTriangles(graph); 
            Array<int, Dynamic, 4> tetrahedra = ::getTetrahedra(graph);

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

            // Populate the 1-skeleton (with uniform weights) 
            for (int i = 0; i < points.rows(); ++i)
                boost::add_vertex(this->one_skeleton); 
            std::vector<std::vector<int> > edges = this->tree.getSubstrings(true, 2); 
            for (const auto& edge : edges)
            {
                int j = edge[0]; 
                int k = edge[1]; 
                boost::add_edge(j, k, EdgeProperty(1.0), this->one_skeleton);
            } 
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
            else if (dim == 0)
                return this->points.rows(); 
            else if (dim == 1)
                return boost::num_edges(this->one_skeleton);
            else    // dim > 1 
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

            // Populate the 1-skeleton (with uniform weights)
            this->one_skeleton.clear(); 
            for (int i = 0; i < points.rows(); ++i)
                boost::add_vertex(this->one_skeleton); 
            for (int i = 0; i < edges.rows(); ++i)
                boost::add_edge(edges(i, 0), edges(i, 1), EdgeProperty(1.0), this->one_skeleton); 
        }

        /**
         * Return the shortest-path tree rooted at the given vertex of the
         * 1-skeleton. 
         *
         * @param root Choice of root vertex. 
         * @returns Shortest-path tree rooted at the given vertex. 
         */
        std::pair<std::vector<std::vector<int> >, Graph> getMinimumWeightPathTree(const int root = 0) const 
        {
            return ::getMinimumWeightPathTree(this->one_skeleton, root); 
        }

        /**
         * Return the shortest path between each pair of vertices in the
         * 1-skeleton.
         *
         * @returns Map of shortest paths in the simplicial complex. 
         */
        std::map<std::pair<int, int>, std::vector<int> > getMinimumWeightPaths() const 
        {
            return ::getMinimumWeightPaths(this->one_skeleton); 
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
        template <typename U>
        Matrix<U, Dynamic, Dynamic> getRealBoundaryHomomorphism(const int dim) const 
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
            Matrix<U, Dynamic, Dynamic> del = Matrix<U, Dynamic, Dynamic>::Zero(n2, n1); 

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
                    del(idx, i) = (j % 2 == 0 ? 1 : -1);   
                }
            }

            return del;
        }

        /**
         * Return the given boundary homomorphism, which (following Munkres,
         * Elements of Algebraic Topology) calculates the boundaries of
         * simplices of the given dimension as a linear combination of
         * simplices of the given dimension minus one.
         *
         * This boundary homomorphism is computed over Z/2Z coefficients. 
         *
         * @param dim Input dimension.
         * @returns Boundary homomorphism from the group of (dim)-chains to 
         *          to the group of (dim - 1)-chains. 
         */
        Matrix<Z2, Dynamic, Dynamic> getZ2BoundaryHomomorphism(const int dim) const 
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
            Matrix<Z2, Dynamic, Dynamic> del = Matrix<Z2, Dynamic, Dynamic>::Zero(n2, n1); 

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
                    del(idx, i) = Z2(1);   // 1 == -1 in Z/2Z
                }
            }

            return del;
        } 

        /**
         * Returns the combinatorial Laplacian over the real/rational numbers.
         *
         * @param dim Input dimension.
         * @returns Combinatorial Laplacian matrix.  
         */
        template <typename U>
        Matrix<U, Dynamic, Dynamic> getCombinatorialLaplacian(const int dim) const 
        {
            // Get the boundary homomorphisms 
            //
            // If dim equals the dimension of the complex, then set del1 = 0 
            //
            // Similarly, if dim == 0, then set del2 = 0
            const int maxdim = this->dimension();
            Matrix<U, Dynamic, Dynamic> lap;  
            if (dim < 0 || dim > this->dimension())
            {
                throw std::runtime_error(
                    "Invalid input dimension for combinatorial Laplacian"
                ); 
            }
            else if (dim == 0 && maxdim == 0)
            {
                const int n = this->points.rows(); 
                lap = Matrix<U, Dynamic, Dynamic>::Zero(n, n); 
            }
            else if (dim == 0)        // maxdim != 0
            {
                Matrix<U, Dynamic, Dynamic> del1 = this->getRealBoundaryHomomorphism<U>(1);
                lap = del1 * del1.transpose(); 
            }
            else if (dim == maxdim)   // dim, maxdim != 0
            {
                Matrix<U, Dynamic, Dynamic> del2 = this->getRealBoundaryHomomorphism<U>(dim);
                lap = del2.transpose() * del2; 
            }
            else     // Otherwise, get both boundary homomorphisms  
            { 
                Matrix<U, Dynamic, Dynamic> del1 = this->getRealBoundaryHomomorphism<U>(dim + 1); 
                Matrix<U, Dynamic, Dynamic> del2 = this->getRealBoundaryHomomorphism<U>(dim); 

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
         * Get a basis of cycles for the homology group over real/rational
         * coefficients of the given dimension. 
         *
         * @param dim Input dimension.
         * @returns Basis of cycles for the homology group. 
         */
        template <typename U>
        Matrix<U, Dynamic, Dynamic> getRealHomology(const int dim) const 
        {
            return ::kernel<U>(this->getCombinatorialLaplacian<U>(dim)); 
        }

        /**
         * Get a basis of cycles for the homology group over Z/2Z coefficients
         * of the given dimension.
         *
         * @param dim Input dimension.
         * @returns Basis of cycles for the homology group. 
         */
        Matrix<Z2, Dynamic, Dynamic> getZ2Homology(const int dim) const 
        {
            Matrix<Z2, Dynamic, Dynamic> cycles;
            if (dim > 0 && dim < this->dimension())
            {
                // Get the boundary homomorphisms and the corresponding
                // quotient space basis 
                Matrix<Z2, Dynamic, Dynamic> del1 = this->getZ2BoundaryHomomorphism(dim + 1);
                Matrix<Z2, Dynamic, Dynamic> del2 = this->getZ2BoundaryHomomorphism(dim);
                cycles = ::quotientSpace<Z2>(del2, del1); 
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
                    cycles = Matrix<Z2, Dynamic, Dynamic>::Identity(n, n); 
                }
                else 
                {
                    // In this case, the image of the 1st boundary homomorphism 
                    // may be nontrivial, and the quotient is F^d modulo the image
                    Matrix<Z2, Dynamic, Dynamic> del1 = this->getZ2BoundaryHomomorphism(dim + 1);
                    cycles = ::quotientSpace<Z2>(del1);
                } 
            }
            else    // dim != 0 and dim == this->dimension() 
            {
                // In this case, the image of the (dim + 1)-th boundary 
                // homomorphism is zero
                Matrix<Z2, Dynamic, Dynamic> del2 = this->getZ2BoundaryHomomorphism(dim);
                cycles = ::kernel<Z2>(del2);
            }

            return cycles; 
        }

        /**
         * Get the vector of Betti numbers over real/rational coefficients
         * for the simplicial complex. 
         *
         * @returns Betti numbers of the simplicial complex. 
         */
        Array<int, Dynamic, 1> getRealBettiNumbers() const 
        {
            Array<int, Dynamic, 1> betti = Array<int, Dynamic, 1>::Zero(4);
            const int maxdim = this->dimension(); 

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (maxdim == 0)
            {
                betti(0) = this->points.rows(); 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= maxdim; ++i)
                {
                    Matrix<Rational, Dynamic, Dynamic> cycles = this->getRealHomology<Rational>(i); 
                    betti(i) = cycles.cols(); 
                }
            }
            
            return betti; 
        }

        /**
         * Get the vector of Betti numbers over Z/2Z coefficients for the 
         * simplicial complex. 
         *
         * @returns Betti numbers of the simplicial complex. 
         */
        Array<int, Dynamic, 1> getZ2BettiNumbers() const 
        {
            Array<int, Dynamic, 1> betti = Array<int, Dynamic, 1>::Zero(4);
            const int maxdim = this->dimension(); 

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (maxdim == 0)
            {
                betti(0) = this->points.rows(); 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= maxdim; ++i)
                {
                    Matrix<Z2, Dynamic, Dynamic> cycles = this->getZ2Homology(i); 
                    betti(i) = cycles.cols(); 
                }
            }

            return betti; 
        }

        /**
         * Return true if the input chain is a cycle of the given dimension
         * over Z/2Z coefficients.
         *
         * @param chain Input chain, as a vector of coefficients over the 
         *              simplices of the appropriate dimension. 
         * @param dim Cycle dimension.
         * @returns True if the chain is a cycle, false otherwise.  
         */
        bool isCycle(const Ref<const Matrix<Z2, Dynamic, 1> >& chain, 
                     const int dim) const
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
                Matrix<Z2, Dynamic, Dynamic> del = this->getZ2BoundaryHomomorphism(dim); 
                return ((del * chain).array() == 0).all(); 
            } 
            else    // If dim == 0, then all vectors are in the kernel 
            {
                return true; 
            }
        }

        /**
         * Return a boolean vector that encodes, for each column in the given
         * matrix, whether the corresponding chain of simplices is a cycle 
         * of the given dimension over Z/2Z coefficients. 
         *
         * @param chains Array of input chains, each as a vector of
         *               coefficients over the simplices of the appropriate
         *               dimension. 
         * @param dim Cycle dimension.
         * @returns Boolean vector indicating whether each chain is a cycle. 
         */
        Matrix<int, Dynamic, 1> areCycles(const Ref<const Matrix<Z2, Dynamic, Dynamic> >& chains, 
                                          const int dim) const
        {
            // Check that the dimension of the input vectors is correct
            if (dim < 0 || dim > this->dimension())
                throw std::runtime_error("Invalid input dimension"); 
            else if (chains.rows() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // Is the vector in the kernel of the boundary homomorphism?
            Matrix<int, Dynamic, 1> are_cycles;  
            if (dim > 0)
            {
                are_cycles = Matrix<int, Dynamic, 1>::Zero(chains.cols()); 
                Matrix<Z2, Dynamic, Dynamic> del = this->getZ2BoundaryHomomorphism(dim);
                for (int i = 0; i < chains.cols(); ++i) 
                    are_cycles(i) = ((del * chains.col(i)).array() == 0).all(); 
            } 
            else    // If dim == 0, then all vectors are in the kernel 
            {
                are_cycles = Matrix<int, Dynamic, 1>::Ones(chains.cols()); 
            }

            return are_cycles; 
        }

        /**
         * Return true if the input chain is a boundary of the given dimension
         * over Z/2Z coefficients.
         *
         * @param chain Input chain, as a vector of coefficients over the 
         *              simplices of the appropriate dimension. 
         * @param dim Boundary dimension (i.e., dimension of the boundary
         *            homomorphism minus 1). 
         * @returns True if the chain is a boundary, false otherwise.  
         */
        bool isBoundary(const Ref<const Matrix<Z2, Dynamic, 1> >& chain, 
                        const int dim) const
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
                Matrix<Z2, Dynamic, Dynamic> del = this->getZ2BoundaryHomomorphism(dim + 1); 
                Matrix<Z2, Dynamic, Dynamic> system(del.rows(), del.cols() + 1); 
                system(Eigen::all, Eigen::seq(0, del.cols() - 1)) = del;
                system.col(del.cols()) = chain; 
                system = ::rowEchelonForm<Z2>(system);

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
         * Return a boolean vector that encodes, for each column in the given
         * matrix, whether the corresponding chain of simplices is a boundary 
         * of the given dimension over Z/2Z coefficients. 
         *
         * @param chains Array of input chains, each as a vector of
         *               coefficients over the simplices of the appropriate
         *               dimension. 
         * @param dim Boundary dimension (i.e., dimension of the boundary
         *            homomorphism minus 1). 
         * @returns True if the chain is a boundary, false otherwise.  
         */
        Matrix<int, Dynamic, 1> areBoundaries(const Ref<const Matrix<Z2, Dynamic, Dynamic> >& chains, 
                                              const int dim) const
        {
            // Check that the dimension of the input vectors is correct
            if (dim < 0 || dim > this->dimension())
                throw std::runtime_error("Invalid input dimension"); 
            else if (chains.rows() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // Is the vector in the image of the boundary homomorphism?
            Matrix<int, Dynamic, 1> are_boundaries = Matrix<int, Dynamic, 1>::Ones(chains.cols()); 
            if (dim < this->dimension())
            {
                // Get the row echelon form of [del | chains]
                Matrix<Z2, Dynamic, Dynamic> del = this->getZ2BoundaryHomomorphism(dim + 1); 
                Matrix<Z2, Dynamic, Dynamic> system(del.rows(), del.cols() + chains.cols()); 
                system(Eigen::all, Eigen::seq(0, del.cols() - 1)) = del;
                system(Eigen::all, Eigen::lastN(chains.cols())) = chains; 
                system = ::rowEchelonForm<Z2>(system);

                // If there is an inconsistency in the row echelon form, this
                // means that the chain is not in the column space of del and 
                // is therefore not a boundary
                for (int j = 0; j < chains.cols(); ++j)
                {
                    for (int i = 0; i < system.rows(); ++i)
                    {
                        if ((system.row(i).head(del.cols()).array() == 0).all() &&
                            system(i, del.cols() + j) != 0)
                        {
                            are_boundaries(j) = 0;
                            break; 
                        }
                    }
                }
            } 
            else    // If dim is maximal, then only the zero vector is in the image
            {
                for (int j = 0; j < chains.cols(); ++j)
                    are_boundaries(j) = (chains.col(j).array() == 0).all(); 
            }

            return are_boundaries; 
        }

        /**
         * Return true if the two cycles (1) are indeed cycles, and (2) are 
         * homologous, over Z/2Z coefficients.  
         *
         * @param chain1 First input chain, as a vector of coefficients over
         *               the simplices of the appropriate dimension. 
         * @param chain2 Second input chain, as a vector of coefficients over
         *               the simplices of the appropriate dimension.
         * @param dim Cycle dimension. 
         * @returns True if the two chains are homologous, false otherwise. 
         */
        bool areHomologousCycles(const Ref<const Matrix<Z2, Dynamic, 1> >& chain1, 
                                 const Ref<const Matrix<Z2, Dynamic, 1> >& chain2, 
                                 const int dim) const
        {
            Matrix<Z2, Dynamic, Dynamic> chains(chain1.size(), 2); 
            chains.col(0) = chain1; 
            chains.col(1) = chain2; 
            return (this->areCycles(chains, dim) && this->isBoundary(chain1 - chain2, dim)); 
        }

        /**
         * A helper function for `annotateEdges()` and `getMinimalFirstHomology()`
         * that calculates the sentinel cycles associated with the minimum-
         * weight-path tree rooted at the given vertex. 
         *
         * See Busaryev et al., SWAT LNCS (2010) and Dey et al., LNCS (2018) 
         * for details.
         *
         * @param root Root vertex for the minimum-weight-path tree.
         * @returns An array indicating the sentinel and non-sentinel edges 
         *          in the 1-skeleton, as well as arrays of coefficient 
         *          vectors for each sentinel cycle. 
         */
        std::pair<Matrix<int, Dynamic, 2>, Matrix<Z2, Dynamic, Dynamic> > getSentinelCycles(const int root = 0) const 
        {
            // Get a spanning tree of the 1-skeleton
            auto result = ::getMinimumWeightPathTree(this->one_skeleton, root);
            std::vector<std::vector<int> > tree_paths = result.first; 
            Graph tree = result.second;

            // Get an edge ordering map for the edges in the 1-skeleton 
            std::unordered_map<std::pair<int, int>, int,
                               boost::hash<std::pair<int, int> > > edge_map = getEdgeOrdering(this->one_skeleton); 

            // Re-order the edges so that the non-tree edges come first
            const int nv = this->points.rows(); 
            const int ne = boost::num_edges(this->one_skeleton);
            Matrix<int, Dynamic, 2> edges = this->getSimplices<1>();
            Matrix<int, Dynamic, 2> edges_reordered(ne, 2);
            int idx1 = 0;
            int idx2 = ne - nv + 1;  
            for (int i = 0; i < ne; ++i)
            {
                if (!boost::edge(edges(i, 0), edges(i, 1), tree).second)
                {
                    edges_reordered.row(idx1) = edges.row(i);
                    idx1++; 
                } 
                else 
                {
                    edges_reordered.row(idx2) = edges.row(i); 
                    idx2++;
                }
            }

            // Define the parent and depth of each vertex with respect to
            // the root in the tree 
            std::vector<int> parents(nv), depths(nv);
            parents[root] = -1;    // The root has no parent 
            depths[root] = 0;      // The root has depth zero
            for (auto& path : tree_paths) 
            {
                int depth = 1; 
                for (auto it = path.begin() + 1; it != path.end(); ++it)
                {
                    auto prev = std::prev(it);
                    parents[*it] = *prev; 
                    depths[*it] = depth; 
                    depth++; 
                }
            }

            // For each non-tree edge (u, v), get the path from u to v in
            // the tree and form the cycle formed by this path and (u, v)
            Matrix<Z2, Dynamic, Dynamic> sentinel_cycles
                = Matrix<Z2, Dynamic, Dynamic>::Zero(ne, ne - nv + 1); 
            for (int i = 0; i < ne - nv + 1; ++i)
            {
                int u = edges_reordered(i, 0); 
                int v = edges_reordered(i, 1); 

                // Get the path from u to v in the tree 
                std::vector<int> path_uv = getPathInMinimumWeightPathTree(
                    tree_paths, parents, depths, u, v
                );

                // Which edges are in the path? (There must be at most 2 
                // edges in the path, since (u, v) is not in the tree)
                for (auto it = path_uv.begin() + 1; it != path_uv.end(); ++it)
                {
                    int j = *std::prev(it); 
                    int k = *it;
                    std::pair<int, int> pair = (j < k ? std::make_pair(j, k) : std::make_pair(k, j)); 
                    int ei = edge_map[pair]; 
                    sentinel_cycles(ei, i) = 1; 
                }

                // Finally account for the edge (u, v) in the cycle 
                std::pair<int, int> pair = (u < v ? std::make_pair(u, v) : std::make_pair(v, u));
                int ei = edge_map[pair];  
                sentinel_cycles(ei, i) = 1;  
            }

            return std::make_pair(edges_reordered, sentinel_cycles); 
        }

        /**
         * Annotate the edges in the simplicial complex, according to the 
         * procedure outlined by Busaryev et al., SWAT LNCS (2010).
         *
         * @param root Root vertex for the minimum-weight-path tree. 
         * @returns Combination of (1) an array indicating the sentinel and
         *          non-sentinel edges, (2) the homology basis corresponding
         *          to the sentinel cycles, and (3) the corresponding array of
         *          edge annotations. 
         */
        std::tuple<Matrix<int, Dynamic, 2>,
                   Matrix<Z2, Dynamic, Dynamic>,
                   Matrix<Z2, Dynamic, Dynamic> > annotateEdges(const int root = 0) const
        {
            // Get the sentinel edges and cycles from the minimum-weight-path
            // tree rooted at the given vertex 
            auto result = this->getSentinelCycles(root); 
            Matrix<int, Dynamic, 2> edges_reordered = result.first; 
            Matrix<Z2, Dynamic, Dynamic> sentinel_cycles = result.second; 

            // Get the earliest basis of [del2 | Z], where del2 is the 
            // boundary homomorphism on 2-chains and Z is the set of 
            // candidate cycles
            Matrix<Z2, Dynamic, Dynamic> del2 = this->getZ2BoundaryHomomorphism(2);
            const int nt = del2.cols(); 
            Matrix<Z2, Dynamic, Dynamic> system(ne, nt + ne - nv + 1);
            system(Eigen::all, Eigen::seq(0, nt - 1)) = del2; 
            system(Eigen::all, Eigen::lastN(ne - nv + 1)) = sentinel_cycles;
            Matrix<Z2, Dynamic, Dynamic> del2_colspace = ::columnSpace<Z2>(del2);  
            Matrix<Z2, Dynamic, Dynamic> total_colspace = ::columnSpace<Z2>(system);
            const int g = total_colspace.cols() - del2_colspace.cols();
            
            // Whichever columns in the column space basis for the total 
            // system does not feature in the column space basis for del2
            // form a homology cycle basis 
            Matrix<Z2, Dynamic, Dynamic> h1_basis = total_colspace(Eigen::all, Eigen::lastN(g)); 

            // Now compute the edge annotations by solving the linear system
            // Zbar * X = Z, where Zbar is the matrix of column space basis
            // vectors for [del2 | Z], and Z is the matrix of homology cycle
            // basis vectors 
            Matrix<Z2, Dynamic, Dynamic> coefs = ::solve<Z2>(total_colspace, sentinel_cycles);

            // Pick out the last g rows of the solution matrix
            coefs = coefs(Eigen::lastN(g), Eigen::all).eval(); 

            // Add zero coefficients for the non-sentinel (tree) edges
            coefs.conservativeResize(coefs.rows(), coefs.cols() + nv - 1);
            coefs(Eigen::all, Eigen::lastN(nv - 1)) = Matrix<Z2, Dynamic, Dynamic>::Zero(g, nv - 1); 

            return std::make_tuple(edges_reordered, h1_basis, coefs); 
        }

        /**
         * Calculate a minimal homology basis for the first homology group 
         * over Z/2Z coefficients, following the procedure outlined by 
         * Dey et al. SCG (2010), LNCS (2018), and using the edge annotation 
         * procedure due to Busaryev et al. SWAT LNCS (2010).
         *
         * The first part of this function is identical to annotateEdges(). 
         *
         * @param root Root vertex for the minimum-weight-path tree. 
         * @returns Combination of (1) an array indicating the "sentinel" and
         *          "non-sentinel" edges, (2) the homology basis corresponding
         *          to the sentinel cycles, and (3) the corresponding array of
         *          edge annotations. 
         */
        Matrix<Z2, Dynamic, Dynamic> getMinimalFirstHomology() const
        {
            // --------------------------------------------------------- //
            //                  COMPUTE EDGE ANNOTATIONS                 //
            // --------------------------------------------------------- //
            // Get the sentinel edges and cycles from the minimum-weight-path
            // tree rooted at the given vertex 
            auto result = this->getSentinelCycles(0); 
            Matrix<int, Dynamic, 2> edges_reordered = result.first; 
            Matrix<Z2, Dynamic, Dynamic> sentinel_cycles = result.second; 

            // Get the earliest basis of [del2 | Z], where del2 is the 
            // boundary homomorphism on 2-chains and Z is the set of 
            // candidate cycles
            Matrix<Z2, Dynamic, Dynamic> del2 = this->getZ2BoundaryHomomorphism(2);
            const int nt = del2.cols(); 
            Matrix<Z2, Dynamic, Dynamic> system(ne, nt + ne - nv + 1);
            system(Eigen::all, Eigen::seq(0, nt - 1)) = del2; 
            system(Eigen::all, Eigen::lastN(ne - nv + 1)) = sentinel_cycles;
            Matrix<Z2, Dynamic, Dynamic> del2_colspace = ::columnSpace<Z2>(del2);  
            Matrix<Z2, Dynamic, Dynamic> total_colspace = ::columnSpace<Z2>(system);
            const int g = total_colspace.cols() - del2_colspace.cols();
            
            // Whichever columns in the column space basis for the total 
            // system does not feature in the column space basis for del2
            // form a homology cycle basis 
            Matrix<Z2, Dynamic, Dynamic> h1_basis = total_colspace(Eigen::all, Eigen::lastN(g)); 

            // Now compute the edge annotations by solving the linear system
            // Zbar * X = Z, where Zbar is the matrix of column space basis
            // vectors for [del2 | Z], and Z is the matrix of homology cycle
            // basis vectors 
            Matrix<Z2, Dynamic, Dynamic> coefs = ::solve<Z2>(total_colspace, sentinel_cycles);

            // Pick out the last g rows of the solution matrix
            coefs = coefs(Eigen::lastN(g), Eigen::all).eval(); 

            // Add zero coefficients for the non-sentinel (tree) edges
            coefs.conservativeResize(coefs.rows(), coefs.cols() + nv - 1);
            coefs(Eigen::all, Eigen::lastN(nv - 1)) = Matrix<Z2, Dynamic, Dynamic>::Zero(g, nv - 1); 

            // --------------------------------------------------------- //
            //              COMPUTE MINIMAL HOMOLOGY BASIS               //
            // --------------------------------------------------------- //
            // Generate sentinel cycles from the minimum-weight-path tree 
            // rooted at each vertex (we already have the cycles for root = 0) 
            Matrix<Z2, Dynamic, Dynamic> candidate_cycles = sentinel_cycles;
            const int nv = this->points.rows();
            const int ne = boost::num_edges(this->one_skeleton);  
            for (int i = 1; i < nv; ++i)
            {
                sentinel_cycles = this->getSentinelCycles(i);

                // Gather whichever sentinel cycles have not yet been
                // encountered
                for (int j = 0; j < sentinel_cycles.cols(); ++j)
                {
                    bool encountered = false;
                    for (int k = 0; k < candidate_cycles.cols(); ++k)
                    {
                        if (((sentinel_cycles.col(j) - candidate_cycles.col(k)).array() == 0).all())
                        {
                            encountered = true; 
                            break; 
                        } 
                    }
                    if (!encountered)
                    {
                        candidate_cycles.conservativeResize(ne, candidate_cycles.cols() + 1); 
                        candidate_cycles.col(candidate_cycles.cols() - 1) = sentinel_cycles.col(j); 
                    }
                }
            }

            // Calculate the annotation for each candidate cycle 
            Matrix<Z2, Dynamic, Dynamic> annotations
                = Matrix<Z2, Dynamic, Dynamic>::Zero(g, candidate_cycles.cols());
            for (int j = 0; j < candidate_cycles.cols(); ++j)
            {
                // Sum the annotations over the edges in the candidate cycle
                for (int i = 0; i < candidate_cycles.rows(); ++i)
                {
                    if (candidate_cycles(i, j))
                    {
                        annotations.col(j) += coefs.col(i); 
                    }
                } 
            }

            // Sort the candidate cycles by length 
            std::vector<std::pair<int, int> > weights; 
            for (int j = 0; j < candidate_cycles.cols(); ++j)
            {
                int weight = 0; 
                for (int i = 0; i < candidate_cycles.rows(); ++i)
                {
                    if (candidate_cycles(i, j))
                        weight++; 
                }
                weights.push_back(std::make_pair(j, weight)); 
            } 
            std::sort(
                weights.begin(), weights.end(),
                [](const std::pair<int, int>& x, const std::pair<int, int>& y)
                {
                    return x.second < y.second; 
                }
            );

            // Now run through the candidate cycles ...
            //
            // Identify the first non-trivial candidate cycle (i.e., first 
            // candidate cycle with nonzero annotation vector) 
            Matrix<Z2, Dynamic, Dynamic> minimal_basis(ne, 0);
            Matrix<Z2, Dynamic, Dynamic> minimal_annotations(g, 0);
            auto it = weights.begin();  
            int curr_idx = it->first;

            // Run until we have gathered g cycles ... 
            while (minimal_basis.cols() < g)
            {
                // If the next candidate cycle is trivial, skip it 
                if ((annotations.col(curr_idx).array() == 0).all())
                {
                    it++;
                    curr_idx = it->first;  
                }
                else     // Otherwise ... 
                {
                    // If we have not yet gathered any cycles, gather this one
                    int ncurr = minimal_basis.cols();  
                    if (ncurr == 0)
                    {
                        minimal_basis.conservativeResize(ne, 1); 
                        minimal_basis.col(ncurr) = candidate_cycles.col(curr_idx); 
                        minimal_annotations.conservativeResize(g, 1); 
                        minimal_annotations.col(ncurr) = annotations.col(curr_idx); 
                    }
                    // Otherwise, check if the corresponding annotation vector
                    // is independent of the others
                    else 
                    { 
                        system.resize(g, ncurr + 1); 
                        system(Eigen::all, Eigen::seqN(0, ncurr)) = minimal_annotations;
                        system.col(ncurr) = annotations.col(curr_idx);
                        system = ::rowEchelonForm<Z2>(system);
                        bool found_inconsistency = false; 
                        for (int i = 0; i < system.rows(); ++i)
                        {
                            if ((system.row(i).head(ncurr).array() == 0).all() && system(i, ncurr) != 0)
                            {
                                found_inconsistency = true;
                                break; 
                            }
                        } 
                        if (found_inconsistency)
                        {
                            minimal_basis.conservativeResize(ne, ncurr + 1); 
                            minimal_basis.col(ncurr) = candidate_cycles.col(curr_idx); 
                            minimal_annotations.conservativeResize(g, ncurr + 1); 
                            minimal_annotations.col(ncurr) = annotations.col(curr_idx); 
                        }
                    }

                    // Move onto the next candidate cycle in the ordering 
                    it++; 
                    curr_idx = it->first; 
                }
            }

            return minimal_basis;  
        }

        /**
         * Calculate a basis of homology classes for the given dimension and 
         * minimize each homology class representative according to its 1-norm
         * using linear programming. 
         *
         * @param dim Input dimension. 
         * @returns Collection of minimal cycles. 
         */
        Matrix<double, Dynamic, Dynamic> minimizeCycles(const int dim) const
        {
            if (dim < 0 || dim > this->dimension())
            {
                throw std::runtime_error(
                    "Invalid input dimension for minimal cycle calculation"
                ); 
            }
            Matrix<Z2, Dynamic, Dynamic> cycles = this->getZ2Homology(dim);
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
                Matrix<Z2, Dynamic, Dynamic> del = this->getZ2BoundaryHomomorphism(dim + 1);
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
                    Matrix<Z2, Dynamic, 1> cycle = cycles.col(j); 

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
