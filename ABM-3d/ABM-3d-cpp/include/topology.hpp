/**
 * Classes and functions for tries and simplicial complexes. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/24/2025
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
#include <CGAL/MP_Float.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include "fields.hpp"
#include "graphs.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::property<boost::edge_weight_t, double> EdgeProperty; 
typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> Graph;
typedef CGAL::MP_Float ET; 
typedef CGAL::Quadratic_program_solution<ET> Solution;

enum class CycleMinimizeMode
{
    MINIMIZE_CYCLE_SIZE = 0,
    MINIMIZE_CYCLE_VOLUME = 1,
    MINIMIZE_CYCLE_WEIGHT = 2
};

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
    std::weak_ptr<TrieNode> parent; 
    std::vector<std::weak_ptr<TrieNode> > children;  
};

class Trie
{
    private:
        // Dynamically allocated TrieNode objects, one for each node in 
        // the trie 
        std::vector<std::shared_ptr<TrieNode> > nodes;

        // Pointer to the root node
        std::weak_ptr<TrieNode> root;

        /**
         * Get the full string/value associated with the given node.
         *
         * The node is assumed to lie within the trie, so that the associated
         * std::shared_ptr has not expired.  
         *
         * @param sptr std::shared_ptr instance pointing to input node. 
         * @returns Corresponding string.  
         */
        std::vector<int> getString(const std::shared_ptr<TrieNode>& sptr) const 
        {
            // Check whether the root has expired 
            if (this->root.expired())
                throw std::runtime_error("Weak_ptr to root node has expired");

            // Travel up the trie from the node to get the value 
            std::vector<int> string;
            std::weak_ptr<TrieNode> curr = sptr; 
            while (curr.lock() != this->root.lock())
            {
                auto sptr = curr.lock();    // This should not have expired  
                string.insert(string.begin(), sptr->letter);

                // Check whether the parent node has expired
                if (!(sptr->parent).expired())
                    curr = sptr->parent;
                else 
                    throw std::runtime_error(
                        "Weak_ptr to intermediate node has expired"
                    ); 
            }

            return string; 
        }

        /**
         * Get the full string/value associated with the given node.
         *
         * Whether the given std::weak_ptr has expired is tested explicitly.
         *
         * @param wptr std::weak_ptr instance pointing to input node. 
         * @returns Corresponding string.  
         */
        std::vector<int> getString(const std::weak_ptr<TrieNode>& wptr) const 
        {
            // Check whether the root has expired 
            if (this->root.expired())
                throw std::runtime_error("Weak_ptr to root node has expired");
            if (wptr.expired())
                throw std::runtime_error("Weak_ptr to intermediate node has expired"); 

            // Travel up the trie from the node to get the value 
            std::vector<int> string;
            std::weak_ptr<TrieNode> curr = wptr;  
            while (curr.lock() != this->root.lock())   // This should always work 
            {
                auto sptr = curr.lock();    // This should not have expired  
                string.insert(string.begin(), sptr->letter);

                // Check whether the parent node has expired
                if (!(sptr->parent).expired())
                    curr = sptr->parent;
                else 
                    throw std::runtime_error(
                        "Weak_ptr to intermediate node has expired"
                    ); 
            }

            return string; 
        } 


    public:
        /**
         * Initialize a trie with -1 at the root. 
         */
        Trie()
        {
            // Dynamically allocated a root node 
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->letter = -1;
            node->level = 0; 
            this->nodes.push_back(node);

            // Make a weak_ptr that points to the root node 
            this->root = node; 
        }

        /**
         * Trivial destructor. 
         */
        ~Trie()
        {
        }

        /**
         * Return true if the trie contains a node for the full string/value.
         *
         * @param string Input string. 
         * @returns True if the trie contains the string, false otherwise. 
         */
        bool containsString(const std::vector<int>& string) const 
        {
            std::weak_ptr<TrieNode> curr; 
            if (!this->root.expired())
                curr = this->root; 
            else 
                throw std::runtime_error("Weak_ptr to root node has expired");

            // Travel down the tree from the root, looking for a node that 
            // contains each successive entry in the string
            for (const char c : string)
            {
                // For each child node ...
                auto sptr = curr.lock();    // This should not have expired  
                std::vector<std::weak_ptr<TrieNode> > children = sptr->children;
                bool found_matching_child = false; 
                for (auto& child_wptr : children)
                {
                    // Check that the weak_ptr has not expired 
                    if (auto child_sptr = child_wptr.lock())
                    {
                        // Check that the letter matches the corresponding
                        // letter for the child node 
                        if (child_sptr->letter == c)
                        {
                            found_matching_child = true; 
                            curr = child_sptr; 
                            break;  
                        }
                    }
                    else 
                    {
                        throw std::runtime_error(
                            "Weak_ptr to intermediate node has expired"
                        ); 
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

            // Dynamically allocated a new root node 
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->letter = -1;
            node->level = 0; 
            this->nodes.push_back(node);

            // Make a weak_ptr that points to the root node 
            this->root = node; 
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
            std::queue<std::weak_ptr<TrieNode> > queue;
            std::weak_ptr<TrieNode> curr; 
            if (!this->root.expired())
                queue.push(this->root); 
            else
                throw std::runtime_error("Weak_ptr to root node has expired");
                 
            int nvisited = 0;
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                curr = queue.front(); 
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue
                auto sptr = curr.lock();    // This should not have expired
                std::vector<std::weak_ptr<TrieNode> > children = sptr->children;
                for (auto& child_wptr : children)
                {
                    // Check that the weak_ptr has not expired 
                    if (!child_wptr.expired())
                    {
                        // If not, add to the queue 
                        queue.push(child_wptr); 
                    }
                    else 
                    {
                        throw std::runtime_error(
                            "Weak_ptr to intermediate node has expired"
                        ); 
                    }
                }

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
            std::weak_ptr<TrieNode> curr; 
            if (!this->root.expired())
                curr = this->root; 
            else
                throw std::runtime_error("Weak_ptr to root node has expired");
            int curr_idx = 0;
            int first_mismatch = -1;  
            while (curr_idx < string.size()) 
            {
                // For each child node ...
                auto sptr = curr.lock();    // This should not have expired 
                std::vector<std::weak_ptr<TrieNode> > children = sptr->children;
                bool found_matching_child = false; 
                for (auto& child_wptr : children)
                {
                    // Check that the weak_ptr has not expired 
                    if (auto child_sptr = child_wptr.lock())
                    {
                        // Check that the letter matches the corresponding
                        // letter for the child node
                        if (child_sptr->letter == string[curr_idx])
                        {
                            found_matching_child = true; 
                            curr = child_sptr;
                            break;  
                        }
                    }
                    else 
                    {
                        throw std::runtime_error(
                            "Weak_ptr to intermediate node has expired"
                        ); 
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
                    std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>(); 
                    node->letter = string[j];
                    node->level = j + 1; 
                    node->parent = curr;
                    this->nodes.push_back(node);

                    // Also update the children of its parent node
                    auto sptr = curr.lock();    // This should not have expired
                    std::weak_ptr<TrieNode> child_wptr = node; 
                    sptr->children.push_back(child_wptr);

                    // Move on to the next node 
                    curr = child_wptr; 
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
            std::weak_ptr<TrieNode> curr;
            if (!this->root.expired())
                curr = this->root; 
            else
                throw std::runtime_error("Weak_ptr to root node has expired");
            int curr_idx = 0;
            int first_mismatch = -1;
            while (curr_idx < string.size()) 
            {
                // For each child node ... 
                auto sptr = curr.lock();    // This should not have expired 
                std::vector<std::weak_ptr<TrieNode> > children = sptr->children;
                bool found_matching_child = false; 
                for (auto& child_wptr : children)
                {
                    // Check that the weak_ptr has not expired 
                    if (auto child_sptr = child_wptr.lock())
                    {
                        // Check that the letter matches the corresponding
                        // letter for the child node
                        if (child_sptr->letter == string(curr_idx))
                        {
                            found_matching_child = true; 
                            curr = child_sptr;
                            break;  
                        }
                    }
                    else 
                    {
                        throw std::runtime_error(
                            "Weak_ptr to intermediate node has expired"
                        ); 
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
                    std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>(); 
                    node->letter = string(j);
                    node->level = j + 1; 
                    node->parent = curr;
                    this->nodes.push_back(node);

                    // Also update the children of its parent node  
                    auto sptr = curr.lock();    // This should not have expired
                    std::weak_ptr<TrieNode> child_wptr = node; 
                    sptr->children.push_back(child_wptr);

                    // Move on to the next node 
                    curr = child_wptr; 
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
            std::queue<std::weak_ptr<TrieNode> > queue; 
            std::weak_ptr<TrieNode> curr; 
            if (!this->root.expired())
                queue.push(this->root); 
            else
                throw std::runtime_error("Weak_ptr to root node has expired");
            int nvisited = 0;
            std::vector<std::vector<int> > strings; 
            
            // While we have nodes left to visit ...
            while (nvisited < this->nodes.size()) 
            {
                // Pop the next node from the queue 
                curr = queue.front(); 
                queue.pop();
                nvisited++;

                // Push the children of this node onto the queue
                //
                // Skip over this if the current node has level less than 
                // the desired length, in which case we don't need to traverse
                // further down the trie
                auto sptr = curr.lock();    // This should not have expired
                if (length == -1 || sptr->level < length)
                {
                    std::vector<std::weak_ptr<TrieNode> > children = sptr->children;
                    for (auto& child_wptr : children)
                    {
                        // Check that the weak_ptr has not expired 
                        if (!child_wptr.expired())
                        {
                            // If not, add to the queue 
                            queue.push(child_wptr); 
                        }
                        else 
                        {
                            throw std::runtime_error(
                                "Weak_ptr to intermediate node has expired"
                            ); 
                        }
                    }
                }

                // Collect the string corresponding to this node if it has
                // the desired length
                int curr_length = sptr->level; 
                if (length == -1)
                {
                    if (!nonempty || curr_length > 0)
                        strings.push_back(this->getString(sptr)); 
                }
                else if (curr_length == length)
                {
                    strings.push_back(this->getString(sptr)); 
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
            for (const auto& string : strings)
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
                        for (const auto& combination : getCombinations(string, length))
                            substrings.insert(combination);
                    }
                    // Otherwise, get all possible substrings 
                    else 
                    {
                        for (const auto& substring : getPowerset(string, nonempty))
                            substrings.insert(substring);
                    } 
                }
            }

            // Sort the substrings in lexicographic order 
            std::vector<std::vector<int> > substrings_sorted;
            for (const auto& v : substrings)
                substrings_sorted.push_back(v); 
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
         * Empty constructor, which yields an empty simplicial complex.
         * Dimension is set to -1.  
         */
        SimplicialComplex3D()
        {
            this->points.resize(0, 3); 
            this->dim = -1; 
        }

        /**
         * Default constructor, which takes in arrays of points, edges, 
         * triangles, and tetrahedra.
         *
         * The edges, triangles, and tetrahedra and assumed to be specified
         * in lexicographical order.  
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

            // Update the dimension of the complex 
            this->dim = this->tree.getHeight() - 1; 
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
                
                // Add the edge into the simplex tree
                std::vector<int> edge_vec;
                if (j < k)
                {
                    edge_vec.push_back(j); 
                    edge_vec.push_back(k);  
                }
                else 
                {
                    edge_vec.push_back(k);
                    edge_vec.push_back(j); 
                }
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

            // Update the dimension of the complex 
            this->dim = this->tree.getHeight() - 1; 
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

            // Update the dimension of the complex 
            this->dim = this->tree.getHeight() - 1; 
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
            return this->dim; 
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
            if (dim > this->dim)
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
         * Update the point coordinates in the simplicial complex. The number
         * of points in the array should match the number of vertices in the
         * complex.  
         *
         * @param points Array of point coordinates. 
         */
        void updatePoints(const Ref<const Array<T, Dynamic, 3> >& points)
        {
            if (points.rows() != this->points.rows())
                throw std::runtime_error(
                    "Given number of points does not match number of vertices"
                ); 
            this->points = points;
        }

        /**
         * Update the simplicial complex with the given points, edges, 
         * triangles, and tetrahedra.
         *
         * The edges, triangles, and tetrahedra and assumed to be specified
         * in lexicographical order.  
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

            // Update the dimension of the complex 
            this->dim = this->tree.getHeight() - 1;
        }

        /**
         * Update the simplicial complex with the edges, triangles, and
         * tetrahedra in the given file, together with the given point 
         * coordinates.  
         *
         * @param filename Input filename.
         * @param points Array of point coordinates.  
         */
        void read(const std::string& filename,
                  const Ref<const Array<T, Dynamic, 3> >& points) 
        {
            // Open input file
            std::ifstream infile(filename);

            // The first line contains the number of vertices 
            std::string line, token; 
            std::stringstream ss; 
            std::getline(infile, line);
            ss << line;  
            std::getline(ss, token, '\t');      // "NUM_VERTICES"
            std::getline(ss, token);            // The number of vertices 
            const int nv = std::stoi(token);
            
            // Check that the input array of point coordinates matches the 
            // number of vertices in the graph 
            if (points.rows() != nv)
                throw std::runtime_error(
                    "Given number of points does not match number of vertices "
                    "in input file"
                ); 

            // The second line contains the number of edges 
            std::getline(infile, line); 
            ss.str(std::string());
            ss.clear(); 
            ss << line; 
            std::getline(ss, token, '\t');      // "NUM_EDGES"
            std::getline(ss, token);            // The number of edges
            const int ne = std::stoi(token);
            Array<int, Dynamic, 2> edges = Array<int, Dynamic, 2>::Zero(ne, 2);

            // Skip over the next line; the fourth line contains the number
            // of triangles 
            std::getline(infile, line);
            std::getline(infile, line);
            ss.str(std::string());
            ss.clear(); 
            ss << line; 
            std::getline(ss, token, '\t');      // "NUM_TRIANGLES"
            std::getline(ss, token);            // The number of triangles
            const int nt = std::stoi(token);
            Array<int, Dynamic, 3> triangles = Array<int, Dynamic, 3>::Zero(nt, 3);

            // The fifth line contains the number of tetrahedra
            std::getline(infile, line);
            ss.str(std::string());
            ss.clear(); 
            ss << line; 
            std::getline(ss, token, '\t');      // "NUM_TETRAHEDRA"
            std::getline(ss, token);            // The number of tetrahedra
            const int nte = std::stoi(token);
            Array<int, Dynamic, 4> tetrahedra = Array<int, Dynamic, 4>::Zero(nte, 4);

            // Skip over all subsequent lines until we reach the edges
            bool reached_eof = false;  
            while (!reached_eof && line.rfind("EDGE\t", 0) != 0)
            {
                std::getline(infile, line);
                reached_eof = infile.eof(); 
            }

            // Collect each edge
            int ei = 0;  
            while (!reached_eof && line.rfind("EDGE\t", 0) == 0)
            {
                ss.str(std::string());
                ss.clear(); 
                ss << line; 
                std::getline(ss, token, '\t');      // "EDGE"
                std::getline(ss, token, '\t');      // First vertex
                int i = std::stoi(token);
                std::getline(ss, token, '\t');      // Second vertex
                int j = std::stoi(token);
                if (i < j)               // No need to parse distance
                {
                    edges(ei, 0) = i; 
                    edges(ei, 1) = j;
                }
                else 
                {
                    edges(ei, 0) = j; 
                    edges(ei, 1) = i;  
                }
                ei++;  

                // Get next line 
                std::getline(infile, line);
                reached_eof = infile.eof(); 
            }

            // Skip over all subsequent lines until we reach the triangles
            while (!reached_eof && line.rfind("TRIANGLE\t", 0) != 0)
            {
                std::getline(infile, line);
                reached_eof = infile.eof(); 
            }

            // Collect each triangle
            int ti = 0;  
            while (!reached_eof && line.rfind("TRIANGLE\t", 0) == 0)
            {
                ss.str(std::string());
                ss.clear(); 
                ss << line; 
                std::getline(ss, token, '\t');      // "TRIANGLE"
                std::getline(ss, token, '\t');      // First vertex
                int i = std::stoi(token); 
                std::getline(ss, token, '\t');      // Second vertex
                int j = std::stoi(token);
                std::getline(ss, token);            // Third vertex 
                int k = std::stoi(token);
                if (i < j && i < k)
                { 
                    triangles(ti, 0) = i;
                    if (j < k)
                    {
                        triangles(ti, 1) = j;
                        triangles(ti, 2) = k;
                    }
                    else 
                    {
                        triangles(ti, 1) = k; 
                        triangles(ti, 2) = j; 
                    }
                }
                else if (j < i && j < k)
                {
                    triangles(ti, 0) = j;
                    if (i < k)
                    {
                        triangles(ti, 1) = i;
                        triangles(ti, 2) = k;
                    }
                    else 
                    {
                        triangles(ti, 1) = k; 
                        triangles(ti, 2) = i; 
                    }
                }
                else    // k is the smallest 
                {
                    triangles(ti, 0) = k;
                    if (i < j)
                    {
                        triangles(ti, 1) = i;
                        triangles(ti, 2) = j;
                    }
                    else 
                    {
                        triangles(ti, 1) = j; 
                        triangles(ti, 2) = i; 
                    }
                }
                ti++;  

                // Get next line 
                std::getline(infile, line);
                reached_eof = infile.eof(); 
            }

            // Collect each tetrahedron 
            ti = 0;  
            while (!reached_eof && line.rfind("TETRAHEDRON\t", 0) == 0)
            {
                ss.str(std::string());
                ss.clear(); 
                ss << line; 
                std::getline(ss, token, '\t');      // "TETRAHEDRON"
                std::getline(ss, token, '\t');      // First vertex
                int i = std::stoi(token); 
                std::getline(ss, token, '\t');      // Second vertex
                int j = std::stoi(token);
                std::getline(ss, token, '\t');      // Third vertex 
                int k = std::stoi(token);
                std::getline(ss, token);            // Fourth vertex 
                int m = std::stoi(token);
                std::vector<int> idx {i, j, k, m}; 
                std::sort(idx.begin(), idx.end()); 
                tetrahedra(ti, 0) = idx[0];
                tetrahedra(ti, 1) = idx[1];
                tetrahedra(ti, 2) = idx[2];
                tetrahedra(ti, 3) = idx[3]; 
                ti++;  

                // Get next line 
                std::getline(infile, line);
                reached_eof = infile.eof(); 
            }

            // Close input file and return 
            infile.close();

            // Update all stored data 
            this->update(points, edges, triangles, tetrahedra); 
        }

        /**
         * Write the simplicial complex to file.
         *
         * @param filename Output filename.  
         */
        void write(const std::string& filename) const 
        {
            std::vector<int> components = ::getConnectedComponents(this->one_skeleton); 
            Array<int, Dynamic, 1> degrees = ::getDegrees(this->one_skeleton);
            Array<T, Dynamic, 1> cluster_coefs = ::getLocalClusteringCoefficients<T>(
                this->one_skeleton
            ); 
            ::writeGraph(
                this->one_skeleton, components, degrees, filename, true, 
                cluster_coefs, true, this->getSimplices<2>(), true, 
                this->getSimplices<3>() 
            ); 
        }

        /**
         * Return arrays that indicate the connected component that contains
         * each vertex in the simplicial complex.
         *
         * @returns  
         */
        Matrix<int, Dynamic, 1> getConnectedComponents() const 
        {
            std::vector<int> components = ::getConnectedComponents(this->one_skeleton); 
            Matrix<int, Dynamic, 1> vertices_in_components(this->points.rows());
            for (int i = 0; i < this->points.rows(); ++i)
                vertices_in_components(i) = components[i];

            return vertices_in_components;  
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
         * Return a map that stores the lexicographical index of each edge
         * in the 1-skeleton.
         *
         * @returns Map of lexicographical indices for the edges in the 
         *          simplicial complex.  
         */
        std::unordered_map<std::pair<int, int>, int, boost::hash<std::pair<int, int> > > getEdgeOrdering() const 
        {
            return ::getEdgeOrdering(this->one_skeleton); 
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
            Array<int, Dynamic, 1> boundary_indicator
                = Array<int, Dynamic, 1>::Zero(this->points.rows());
            Array<T, Dynamic, 3> boundary_points; 
            Trie boundary_tree;

            // If the simplicial complex is 0-dimensional (there are only 
            // isolated points), return it as is 
            if (this->dim == 0)
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
                    if (dim < this->dim)
                    {
                        // Generate all full-dimensional simplices that have 
                        // this simplex as a face 
                        std::vector<std::vector<int> > cofaces
                            = this->tree.getSuperstrings(simplex, this->dim + 1); 

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
         * Return the subcomplex induced by the given subset of vertices.
         *
         * The vertices in the subcomplex are re-indexed 0, 1, ... according
         * to the ordering given in the input vector. In other words, if the
         * input vector is [1, 4, 2], then the vertices are re-indexed as 
         * 1 -> 0, 4 -> 1, 2 -> 2. 
         *
         * @params Input array of vertices. 
         * @returns The induced subcomplex.
         */
        SimplicialComplex3D<T> getSubcomplex(const std::vector<int>& vertices)
        {
            // Initialize a new simplex tree for the subcomplex 
            Trie subcomplex_tree; 

            // Generate a map for the vertex re-indexing 
            std::unordered_map<int, int> vertex_map;
            int i = 0; 
            for (auto it = vertices.begin(); it != vertices.end(); ++it)
            {
                vertex_map[*it] = i;
                i++; 
            } 

            // For each simplex in the complex ... 
            std::vector<std::vector<int> > simplices = this->tree.getSubstrings(true); 
            for (auto&& simplex : simplices)
            {
                // Are any of the vertices in the simplex *not* among the given
                // vertices? 
                bool in_subcomplex = true; 
                for (const int v : simplex)
                {
                    if (vertex_map.find(v) == vertex_map.end())
                    {
                        in_subcomplex = false;
                        break; 
                    }
                }

                // If not, then add the simplex to the subcomplex
                if (in_subcomplex) 
                {
                    // Re-index the simplex vertices 
                    std::vector<int> simplex_reindexed; 
                    for (auto it = simplex.begin(); it != simplex.end(); ++it)
                    {
                        int j = vertex_map[*it];

                        // Insert the vertex into the simplex while maintaining
                        // an ascending order
                        auto it2 = simplex_reindexed.begin(); 
                        while (it2 != simplex_reindexed.end())
                        {
                            if (j < *it2)
                                break; 
                            it2++; 
                        }
                        simplex_reindexed.insert(it2, j);   // Insert before *it2 
                    }
                    subcomplex_tree.insert(simplex_reindexed);
                } 
            } 

            return SimplicialComplex3D<T>(this->points(vertices, Eigen::all), subcomplex_tree); 
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
            if (dim == 0 || dim > this->dim)
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
            if (dim == 0 || dim > this->dim)
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
            Matrix<U, Dynamic, Dynamic> lap;  
            if (dim < 0 || dim > this->dim)
            {
                throw std::runtime_error(
                    "Invalid input dimension for combinatorial Laplacian"
                ); 
            }
            else if (dim == 0 && this->dim == 0)
            {
                const int n = this->points.rows(); 
                lap = Matrix<U, Dynamic, Dynamic>::Zero(n, n); 
            }
            else if (dim == 0)           // this->dim != 0
            {
                Matrix<U, Dynamic, Dynamic> del1 = this->getRealBoundaryHomomorphism<U>(1);
                lap = del1 * del1.transpose(); 
            }
            else if (dim == this->dim)   // dim, this->dim != 0
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
            if (dim > 0 && dim < this->dim)
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
                if (this->dim == 0)
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
            else    // dim != 0 and dim == this->dim
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

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (this->dim == 0)
            {
                betti(0) = this->points.rows(); 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= this->dim; ++i)
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

            // If the complex dimension is zero, then simply return the 
            // number of points as the 0-th Betti number
            if (this->dim == 0)
            {
                betti(0) = this->points.rows(); 
            }
            // If not, then compute each combinatorial Laplacian 
            else 
            {
                // Calculate the dimension of each homology group
                for (int i = 0; i <= this->dim; ++i)
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
            if (dim < 0 || dim > this->dim)
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
            if (dim < 0 || dim > this->dim)
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
            if (dim < 0 || dim > this->dim)
                throw std::runtime_error("Invalid input dimension"); 
            else if (chain.size() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // If the chain is zero, return true 
            if ((chain.array() == 0).all())
                return true; 

            // Is the vector in the image of the boundary homomorphism? 
            if (dim < this->dim)
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
            // If dim is maximal, then only the zero vector is in the image,
            // but the vector must be nonzero at this point 
            else 
            {
                return false; 
            }
        }

        /**
         * Return a boolean vector that encodes, for each column in the given
         * matrix, whether the corresponding chain of simplices is a boundary 
         * of the given dimension over Z/2Z coefficients.
         *
         * All chains are assumed to be nonzero.  
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
            if (dim < 0 || dim > this->dim)
                throw std::runtime_error("Invalid input dimension"); 
            else if (chains.rows() != this->getNumSimplices(dim))
                throw std::runtime_error(
                    "Input vector does not represent chain of given dimension"
                );

            // Is the vector in the image of the boundary homomorphism?
            Matrix<int, Dynamic, 1> are_boundaries = Matrix<int, Dynamic, 1>::Ones(chains.cols()); 
            if (dim < this->dim)
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
            bool are_cycles = (this->areCycles(chains, dim).array() == 1).all();
            bool diff_is_boundary = this->isBoundary(chain1 - chain2, dim);
            return (are_cycles && diff_is_boundary); 
        }

        /**
         * Lift the given cycle over Z/2Z coefficients to a cycle over real/
         * rational coefficients.
         *
         * @param cycle Input cycle over Z/2Z coefficients.
         * @param dim Input dimension. 
         * @returns Lifted cycle over real/rational coefficients. 
         */
        template <typename U>
        Matrix<U, Dynamic, 1> liftCycle(const Ref<const Matrix<Z2, Dynamic, 1> >& cycle,
                                        const int dim) const
        {
            // Start by setting each coefficient to +1
            Matrix<U, Dynamic, 1> lifted(cycle.size());
            for (int i = 0; i < cycle.size(); ++i)
                lifted(i) = static_cast<U>(cycle(i));

            // Get the boundary of the non-oriented chain 
            Matrix<U, Dynamic, Dynamic> del = this->getRealBoundaryHomomorphism<U>(dim);
            Matrix<U, Dynamic, 1> boundary = del * lifted;

            // Every coefficient in the boundary should be even, since 
            // each face in the cycle goes to zero over Z/2Z coefficients
            //
            // Obtain the chain such that del * chain = boundary / 2
            Matrix<U, Dynamic, Dynamic> system(del.rows(), del.cols() + 1);
            system(Eigen::all, Eigen::seq(0, del.cols() - 1)) = del; 
            system.col(del.cols()) = boundary / 2;
            Matrix<U, Dynamic, Dynamic> system_reduced = ::rowEchelonForm<U>(system);
            Matrix<U, Dynamic, 1> chain = ::solve<U>(del, boundary / 2).col(0);

            // Then calculate the reoriented lifted cycle 
            return lifted - 2 * chain;  
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
         * @param vertices_in_component
         * @param edges_in_component 
         * @param edge_map 
         * @returns An array indicating the sentinel and non-sentinel edges 
         *          in the 1-skeleton, as well as arrays of coefficient 
         *          vectors for each sentinel cycle. 
         */
        std::pair<Matrix<int, Dynamic, 2>, Matrix<Z2, Dynamic, Dynamic> >
            getSentinelCycles(const int root,
                              const Ref<const Matrix<int, Dynamic, 1> >& vertices_in_component,
                              const Ref<const Matrix<int, Dynamic, 1> >& edges_in_component,  
                              std::unordered_map<std::pair<int, int>, int,
                                                 boost::hash<std::pair<int, int> > >& edge_map) const
        {
            // Get a spanning forest of the 1-skeleton, rooted at the given
            // root vertex
            auto result = this->getMinimumWeightPathTree(root);
            std::vector<std::vector<int> > tree_paths = result.first; 
            Graph tree = result.second;

            // Re-order the edges so that the non-tree edges come first, 
            // counting only the vertices and edges that lie in the connected
            // component  
            const int nv = vertices_in_component.sum(); 
            const int ne = edges_in_component.sum();
            const int nv_total = this->points.rows(); 
            const int ne_total = boost::num_edges(this->one_skeleton);
            Matrix<int, Dynamic, 2> edges_total = this->getSimplices<1>();

            // There are a total of ne edges in the connected component, 
            // and a total of nv - 1 edges in the tree, so there should be 
            // ne - nv + 1 non-tree edges 
            Matrix<int, Dynamic, 2> edges_reordered(ne, 2);
            int idx1 = 0;
            int idx2 = ne - nv + 1;  
            for (int i = 0; i < ne_total; ++i)
            {
                // Check that the edge is in the connected component ... 
                if (edges_in_component(i))
                {
                    // ... then check that it is in the tree 
                    if (!boost::edge(edges_total(i, 0), edges_total(i, 1), tree).second)
                    {
                        edges_reordered.row(idx1) = edges_total.row(i);
                        idx1++; 
                    } 
                    else 
                    {
                        edges_reordered.row(idx2) = edges_total.row(i); 
                        idx2++;
                    }
                }
            }

            // Define the parent and depth of each vertex with respect to
            // the root in the tree
            //
            // Set to -2 by default, indicating vertices that are disconnected
            // from the root  
            std::vector<int> parents, depths; 
            for (int i = 0; i < nv_total; ++i)
            {
                parents.push_back(-2);
                depths.push_back(-2); 
            }
            parents[root] = -1;    // The root has no parent 
            depths[root] = 0;      // The root has depth zero
            for (auto& path : tree_paths) 
            {
                // If the path is nontrivial, then update parents and depths
                // along the path 
                if (path.size() > 1)
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
            }

            // For each non-tree edge (u, v) within the connected component,
            // get the path from u to v in the tree and form the cycle formed
            // by this path and (u, v)
            Matrix<Z2, Dynamic, Dynamic> sentinel_cycles
                = Matrix<Z2, Dynamic, Dynamic>::Zero(ne_total, ne - nv + 1); 
            for (int i = 0; i < ne - nv + 1; ++i)
            {
                int u = edges_reordered(i, 0); 
                int v = edges_reordered(i, 1);

                // Is (u, v) in the connected component?
                std::pair<int, int> pair = std::make_pair(u, v); 
                int ei = edge_map[pair]; 
                if (edges_in_component(ei)) 
                {
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
                        pair = (j < k ? std::make_pair(j, k) : std::make_pair(k, j)); 
                        ei = edge_map[pair]; 
                        sentinel_cycles(ei, i) = 1; 
                    }

                    // Finally account for the edge (u, v) in the cycle 
                    pair = std::make_pair(u, v); 
                    ei = edge_map[pair];  
                    sentinel_cycles(ei, i) = 1;
                } 
            }

            return std::make_pair(edges_reordered, sentinel_cycles); 
        }

        /**
         * Calculate a minimal homology basis for the first homology group 
         * over Z/2Z coefficients, following the procedure outlined by 
         * Dey et al. SCG (2010), LNCS (2018), and using the edge annotation 
         * procedure due to Busaryev et al. SWAT LNCS (2010).
         *
         * @param verbose If true, print intermittent output to stdout. 
         */
        Matrix<Z2, Dynamic, Dynamic> getMinimalFirstHomology(const bool verbose = false) const
        {
            // If the complex is 0-dimensional, then raise an exception 
            if (this->dim == 0)
                throw std::runtime_error(
                    "Cannot compute minimal first homology for 0-dimensional complex"
                );

            // If the complex is 1-dimensional, then return the minimal 
            // cycle basis for the 1-skeleton 
            if (this->dim == 1)
                return getMinimumCycleBasis<Z2>(this->one_skeleton);

            // ---------------------------------------------------------- //
            //                IDENTIFY CONNECTED COMPONENTS               //
            // ---------------------------------------------------------- //
            if (verbose)
                std::cout << "Identifying connected components ...\n";

            // Get the connected components of the complex 
            Matrix<int, Dynamic, 1> components = this->getConnectedComponents();

            // Choose one vertex from each connected component
            const int nv_total = this->points.rows();
            const int n_components = components.maxCoeff() + 1;
            std::vector<int> roots; 
            for (int i = 0; i < n_components; ++i)
            {
                for (int j = 0; j < nv_total; ++j)
                {
                    if (components(j) == i)
                    {
                        roots.push_back(j); 
                        break; 
                    }
                }
            }

            // Get a map for the edges in the 1-skeleton
            Matrix<int, Dynamic, 2> edges = this->getSimplices<1>();  
            auto edge_map = this->getEdgeOrdering();
            const int ne_total = edges.rows();

            // Get the triangles as well 
            Matrix<int, Dynamic, 3> triangles = this->getSimplices<2>();  
            const int nt_total = triangles.rows();

            // Partition the vertices, edges, and triangles across the 
            // connected components
            Matrix<int, Dynamic, Dynamic> vertices_in_components
                = Matrix<int, Dynamic, Dynamic>::Zero(nv_total, n_components);
            for (int i = 0; i < nv_total; ++i)
            {
                vertices_in_components(i, components(i)) = 1;
            }
            Matrix<int, Dynamic, Dynamic> edges_in_components
                = Matrix<int, Dynamic, Dynamic>::Zero(ne_total, n_components); 
            for (int i = 0; i < ne_total; ++i)
            {
                int u = edges(i, 0); 
                edges_in_components(i, components(u)) = 1; 
            }
            Matrix<int, Dynamic, Dynamic> triangles_in_components
                = Matrix<int, Dynamic, Dynamic>::Zero(nt_total, n_components);
            for (int i = 0; i < nt_total; ++i)
            {
                int u = triangles(i, 0); 
                triangles_in_components(i, components(u)) = 1;
            }

            if (verbose)
                std::cout << "... identified " << n_components << " connected components\n";

            // Get the boundary homomorphism on 2-chains 
            Matrix<Z2, Dynamic, Dynamic> del2 = this->getZ2BoundaryHomomorphism(2);

            // Initialize array of minimal homology basis vectors for the 
            // entire complex
            Matrix<Z2, Dynamic, Dynamic> minimal_basis_total(ne_total, 0);

            // Separately calculate the minimal homology basis for each
            // connected component ... 
            for (int component_idx = 0; component_idx < n_components; ++component_idx)
            {
                // ---------------------------------------------------------- //
                //                  COMPUTE EDGE ANNOTATIONS                  //
                // ---------------------------------------------------------- //
                int root = roots[component_idx];
                if (verbose)
                {
                    std::cout << "Calculating minimal homology basis for component "
                              << component_idx << " ...\n" 
                              << "- Root for calculating edge annotations: "
                              << root << std::endl;
                } 

                // Get the connected component containing the root
                int nv = vertices_in_components.col(component_idx).sum();
                std::vector<int> vertices_in_component_idx; 
                for (int j = 0; j < nv_total; ++j)
                {
                    if (vertices_in_components(j, component_idx) == 1)
                    {
                        vertices_in_component_idx.push_back(j);
                    } 
                }

                // Define an array indicating the edges in the connected component 
                int ne = edges_in_components.col(component_idx).sum();
                std::vector<int> edges_in_component_idx; 
                for (int j = 0; j < ne_total; ++j)
                {
                    if (edges_in_components(j, component_idx) == 1)
                    {
                        edges_in_component_idx.push_back(j); 
                    } 
                }

                // Define an array indicating the triangles in the connected 
                // component
                int nt = triangles_in_components.col(component_idx).sum();
                std::vector<int> triangles_in_component_idx; 
                for (int j = 0; j < nt_total; ++j)
                {
                    if (triangles_in_components(j, component_idx) == 1)
                    {
                        triangles_in_component_idx.push_back(j); 
                    } 
                }
                if (verbose)
                {
                    std::cout << "- " << nv << " vertices, " << ne << " edges, "
                              << nt << " triangles\n";
                } 

                // If this subcomplex is 0-dimensional, then skip 
                if (ne == 0)
                {
                    continue;
                } 
                // If this subcomplex is 1-dimensional, then return the minimal 
                // cycle basis for the 1-skeleton of the subcomplex  
                else if (nt == 0)
                {
                    if (verbose)
                    {
                        std::cout << "- No triangles, calculating minimal "
                                  << "cycle basis for 1-skeleton\n"; 
                    } 
                    Graph subgraph;

                    // Define a map for the vertices in the subgraph that 
                    // sends the index of the vertex w.r.t the entire complex
                    // to the index w.r.t the subgraph 
                    std::unordered_map<int, int> vertex_map; 
                    for (int j = 0; j < nv; ++j)
                    {
                        int v = vertices_in_component_idx[j];  
                        boost::add_vertex(subgraph);
                        vertex_map[v] = j;    // v = index w.r.t entire graph
                    }

                    // Add each edge to the subgraph 
                    for (int j = 0; j < ne; ++j)
                    {
                        int idx = edges_in_component_idx[j]; 
                        int u = vertex_map[edges(idx, 0)]; 
                        int v = vertex_map[edges(idx, 1)]; 
                        boost::add_edge(u, v, EdgeProperty(1.0), subgraph);
                    }
                    
                    // Define a map that sends i to the i-th edge in the
                    // subgraph, w.r.t the edge ordering in the subgraph
                    //
                    // Compare with getEdgeOrdering() 
                    std::pair<boost::graph_traits<Graph>::edge_iterator, 
                              boost::graph_traits<Graph>::edge_iterator> it;
                    std::vector<std::pair<int, int> > edges_subgraph; 
                    for (it = boost::edges(subgraph); it.first != it.second; ++it.first)
                    {
                        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
                        int u = boost::source(edge, subgraph); 
                        int v = boost::target(edge, subgraph);
                        if (u < v)
                            edges_subgraph.push_back(std::make_pair(u, v));
                        else 
                            edges_subgraph.push_back(std::make_pair(v, u));  
                    }
                    std::sort(edges_subgraph.begin(), edges_subgraph.end());
                    
                    // Get a minimum cycle basis of the subgraph
                    Matrix<Z2, Dynamic, Dynamic> cycle_basis = getMinimumCycleBasis<Z2>(subgraph);

                    // Re-define the cycle basis array in terms of the entire 
                    // complex 
                    Matrix<Z2, Dynamic, Dynamic> cycle_basis_extended
                        = Matrix<Z2, Dynamic, Dynamic>::Zero(ne_total, cycle_basis.cols());
                    for (int j = 0; j < cycle_basis.cols(); ++j)
                    {
                        for (int k = 0; k < ne; ++k)
                        {
                            // Get the endpoints of the k-th edge in the subgraph,
                            // w.r.t the edge ordering in the subgraph
                            int u = edges_subgraph[k].first; 
                            int v = edges_subgraph[k].second;
                            
                            // Get the indices of the endpoints w.r.t the vertex
                            // ordering in the entire complex 
                            int u2 = vertices_in_component_idx[u]; 
                            int v2 = vertices_in_component_idx[v];  

                            // Find the index of this edge w.r.t the edge ordering
                            // in the entire complex 
                            int idx = edge_map[std::make_pair(u2, v2)];
                            cycle_basis_extended(idx, j) = cycle_basis(k, j); 
                        }
                    }

                    // Add each cycle basis element to the homology basis
                    int g = cycle_basis_extended.cols();  
                    minimal_basis_total.conservativeResize(
                        ne_total, minimal_basis_total.cols() + g
                    ); 
                    minimal_basis_total(Eigen::all, Eigen::lastN(g)) = cycle_basis_extended;
                    if (verbose)
                        std::cout << "- Identified basis of " << g << " cycles\n"; 
                    
                    // Skip to the next connected component 
                    continue;  
                }

                // Get the sentinel edges and cycles from the minimum-weight-path
                // tree rooted at the current root
                auto result = this->getSentinelCycles(
                    root,
                    vertices_in_components.col(component_idx),
                    edges_in_components.col(component_idx),
                    edge_map
                );
                Matrix<int, Dynamic, 2> edges_reordered = result.first; 
                Matrix<Z2, Dynamic, Dynamic> sentinel_cycles = result.second;
                if (verbose)
                {
                    std::cout << "- Calculated sentinel cycles for root = " << root
                              << std::endl; 
                } 

                // Find the submatrix of del2 that concerns only the current
                // connected component, as well as the projection of the 
                // sentinel cycles onto the edge coordinates for the current
                // component 
                Matrix<Z2, Dynamic, Dynamic> del2_component(ne, nt);
                Matrix<Z2, Dynamic, Dynamic> sentinel_cycles_proj(ne, ne - nv + 1);  
                for (int j = 0; j < ne; ++j)
                {
                    int ei = edges_in_component_idx[j]; 
                    for (int k = 0; k < nt; ++k)
                    {
                        int ti = triangles_in_component_idx[k]; 
                        del2_component(j, k) = del2(ei, ti); 
                    }
                }
                for (int j = 0; j < ne - nv + 1; ++j)
                    sentinel_cycles_proj.col(j) = sentinel_cycles(edges_in_component_idx, j); 

                // Get the earliest basis of [del2 | Z], where del2 is the 
                // boundary homomorphism on 2-chains and Z is the set of 
                // sentinel cycles
                Matrix<Z2, Dynamic, Dynamic> system(ne, nt + ne - nv + 1);
                system(Eigen::all, Eigen::seq(0, nt - 1)) = del2_component; 
                system(Eigen::all, Eigen::lastN(ne - nv + 1)) = sentinel_cycles_proj;
                Matrix<Z2, Dynamic, Dynamic> del2_colspace = ::columnSpace<Z2>(del2_component);  
                Matrix<Z2, Dynamic, Dynamic> total_colspace = ::columnSpace<Z2>(system);
                int g = total_colspace.cols() - del2_colspace.cols();
                
                // Now compute the edge annotations by solving the linear system
                // Zbar * X = Z, where Zbar is the matrix of column space basis
                // vectors for [del2 | Z], and Z is the matrix of homology cycle
                // basis vectors 
                Matrix<Z2, Dynamic, Dynamic> coefs_component = ::solve<Z2>(
                    total_colspace, sentinel_cycles_proj
                );

                // Pick out the last g rows of the solution matrix
                coefs_component = coefs_component(Eigen::lastN(g), Eigen::all).eval();

                // Add zero coefficients for the non-sentinel (tree) edges
                coefs_component.conservativeResize(g, coefs_component.cols() + nv - 1); 
                coefs_component(Eigen::all, Eigen::lastN(nv - 1))
                    = Matrix<Z2, Dynamic, Dynamic>::Zero(g, nv - 1);

                // Reorder and extend the annotations according to the
                // lexicographic ordering over the entire complex
                Matrix<Z2, Dynamic, Dynamic> coefs_lex
                    = Matrix<Z2, Dynamic, Dynamic>::Zero(g, ne_total); 
                for (int j = 0; j < ne; ++j)
                {
                    // Get the j-th edge in the sentinel/non-sentinel ordering
                    //
                    // Note that this ordering only runs over the edges in 
                    // the component 
                    std::pair<int, int> pair = std::make_pair(
                        edges_reordered(j, 0), edges_reordered(j, 1)
                    );

                    // Get the index of the edge in the lexicographical ordering 
                    int ei = edge_map[pair]; 

                    // Reorder the annotations accordingly 
                    coefs_lex.col(ei) = coefs_component.col(j); 
                }
                if (verbose)
                    std::cout << "- Calculated edge annotations\n"; 

                // ---------------------------------------------------------- //
                //               COMPUTE MINIMAL HOMOLOGY BASIS               //
                // ---------------------------------------------------------- //
                // Generate sentinel cycles from the minimum-weight-path tree 
                // rooted at each vertex in the connected component (we already
                // have the cycles for the first root)
                //
                // First, generate a set of candidate cycles, each represented
                // by a binary string
                std::unordered_set<std::string> candidate_set;
                int n_candidate_cycles = sentinel_cycles.cols();  
                for (int j = 0; j < n_candidate_cycles; ++j)
                {
                    std::string cycle_str = "";  
                    for (int k = 0; k < ne_total; ++k)
                    {
                        if (sentinel_cycles(k, j) == 1)
                            cycle_str += "1"; 
                        else 
                            cycle_str += "0"; 
                    }
                    candidate_set.insert(cycle_str); 
                }
                if (verbose)
                {
                    std::cout << "- Accumulated candidate cycles ("
                              << candidate_set.size() << " total)\n"; 
                } 
                for (int j = 1; j < nv; ++j)
                {
                    // Get the j-th vertex in the component 
                    int vi = vertices_in_component_idx[j];

                    // Get the corresponding sentinel cycles  
                    result = this->getSentinelCycles(
                        vi,
                        vertices_in_components.col(component_idx),
                        edges_in_components.col(component_idx),
                        edge_map
                    );
                    sentinel_cycles = result.second;
                    if (verbose)
                    {
                        std::cout << "- Calculated sentinel cycles for root = " << vi
                                  << std::endl; 
                    } 

                    // Gather whichever sentinel cycles have not yet been
                    // encountered
                    for (int k = 0; k < sentinel_cycles.cols(); ++k)
                    {
                        std::string cycle_str = "";  
                        for (int m = 0; m < ne_total; ++m)
                        {
                            if (sentinel_cycles(m, k) == 1)
                                cycle_str += "1"; 
                            else 
                                cycle_str += "0"; 
                        }
                        if (candidate_set.find(cycle_str) == candidate_set.end())
                            candidate_set.insert(cycle_str); 
                    }
                    n_candidate_cycles = candidate_set.size(); 
                    if (verbose)
                    {
                        std::cout << "- Accumulated candidate cycles ("
                                  << n_candidate_cycles << " total)\n"; 
                    } 
                }
                Matrix<Z2, Dynamic, Dynamic> candidate_cycles(ne_total, n_candidate_cycles);
                int j = 0;
                for (const std::string& cycle_str : candidate_set)
                {
                    for (int k = 0; k < ne_total; ++k)
                    {
                        candidate_cycles(k, j) = cycle_str[k] - '0'; 
                    }
                    j++; 
                } 

                // Calculate the annotation for each candidate cycle 
                Matrix<Z2, Dynamic, Dynamic> annotations
                    = Matrix<Z2, Dynamic, Dynamic>::Zero(g, n_candidate_cycles); 
                for (int j = 0; j < n_candidate_cycles; ++j)
                {
                    // Sum the annotations over the edges in the candidate cycle
                    for (int k = 0; k < ne_total; ++k)
                    {
                        // If edge k is in the j-th candidate cycle ... 
                        if (candidate_cycles(k, j))
                        {
                            annotations.col(j) += coefs_lex.col(k); 
                        }
                    } 
                }
                if (verbose)
                    std::cout << "- Calculated candidate cycle annotations\n"; 

                // Sort the candidate cycles by length 
                std::vector<std::pair<int, int> > weights; 
                for (int j = 0; j < candidate_cycles.cols(); ++j)
                {
                    int weight = 0; 
                    for (int k = 0; k < candidate_cycles.rows(); ++k)
                    {
                        if (candidate_cycles(k, j))
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
                Matrix<Z2, Dynamic, Dynamic> minimal_basis(ne_total, 0);
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
                            minimal_basis.conservativeResize(ne_total, 1); 
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
                            for (int j = 0; j < system.rows(); ++j)
                            {
                                bool row_zero = (system.row(j).head(ncurr).array() == 0).all(); 
                                bool b_nonzero = (system(j, ncurr) != 0); 
                                if (row_zero && b_nonzero)
                                {
                                    found_inconsistency = true;
                                    break; 
                                }
                            } 
                            if (found_inconsistency)
                            {
                                minimal_basis.conservativeResize(ne_total, ncurr + 1); 
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
                if (verbose)
                {
                    std::cout << "- Identified " << g << " minimal homology basis "
                              << "elements\n";
                } 

                // Collect the minimal basis for the current connected component
                minimal_basis_total.conservativeResize(
                    ne_total, minimal_basis_total.cols() + g
                ); 
                minimal_basis_total(Eigen::all, Eigen::lastN(g)) = minimal_basis;  
            }

            return minimal_basis_total;  
        }

        /**
         * Given a collection of cycles representing different homology 
         * classes for the given dimension, minimize each cycle according to
         * its weighted 1-norm using linear programming, following the approach
         * of Dey et al., SIAM J Comput (2011) and Obayashi, SIAM J Appl
         * Algebra Geometry (2018).
         *
         * It is assumed that the input cycles are indeed cycles. 
         *
         * There are three different minimization modes, each of which 
         * seeks to minimize an objective of the form w^T * |x|, where x is 
         * a vector of real-valued cycle coefficients and w is a vector of 
         * simplex weights. The three modes are:
         * - MINIMIZE_CYCLE_SIZE: Here, we set w = 1 for each simplex, so
         *                        that we minimize the 1-norm of x.
         * - MINIMIZE_CYCLE_VOLUME: Here, we set the k-th entry of w to the
         *                          volume of the k-th simplex (1 for points,
         *                          length for edges, area for triangles).
         * - MINIMIZE_CYCLE_WEIGHT: Here, we set w to the given vector of 
         *                          pre-computed weights. 
         *
         * @param cycles Input collection of cycles. 
         * @param dim Input dimension.
         * @param min_mode Minimization mode.  
         * @param weights Input simplex weights. Only used if `min_mode` is 
         *                MINIMIZE_CYCLE_WEIGHT.  
         * @param verbose If true, print intermittent output to stdout.
         * @param lp_verbose If true, set verbosity flag in the CGAL linear
         *                   program solver to 1.  
         * @returns Collection of minimal cycles. 
         */
        Matrix<double, Dynamic, Dynamic> minimizeCycles(const Ref<const Matrix<Z2, Dynamic, Dynamic> >& cycles, 
                                                        const int dim,
                                                        const CycleMinimizeMode min_mode,
                                                        const Ref<const Matrix<double, Dynamic, 1> >& weights,
                                                        const bool verbose = false,
                                                        const bool lp_verbose = false) const 
        {
            if (dim < 0 || dim > this->dim || dim > 2)
            {
                throw std::runtime_error(
                    "Invalid input dimension for minimal cycle calculation"
                ); 
            }

            // Get the simplices in the complex 
            Matrix<int, Dynamic, 2> edges = this->getSimplices<1>();
            Matrix<int, Dynamic, 3> triangles = this->getSimplices<2>(); 
            Matrix<int, Dynamic, 4> tetrahedra = this->getSimplices<3>(); 
            auto simplices = std::make_tuple(edges, triangles, tetrahedra);
            int n1_total, n2_total; 
            if (dim == 0)
            {
                n1_total = edges.rows(); 
                n2_total = this->points.rows(); 
            }
            else if (dim == 1) 
            {
                n1_total = triangles.rows(); 
                n2_total = edges.rows(); 
            }
            else    // dim == 2
            {
                n1_total = tetrahedra.rows(); 
                n2_total = triangles.rows(); 
            }

            // Check that the given cycles have the correct dimension 
            if (n2_total != cycles.rows())
            {
                throw std::runtime_error(
                    "Input cycle coefficients do not match input dimension" 
                ); 
            }
            Matrix<double, Dynamic, Dynamic> opt_cycles
                = Matrix<double, Dynamic, Dynamic>::Zero(n2_total, cycles.cols()); 

            // If the input dimension is maximal, then optimization is not 
            // necessary
            if (dim == this->dim)
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
                if (verbose)
                {
                    std::cout << "Minimizing " << cycles.cols() << " cycles ...\n"
                              << "Identifying connected components ...\n";
                }

                // Get the connected components of the complex 
                Matrix<int, Dynamic, 1> components = this->getConnectedComponents();
                const int n_components = components.maxCoeff() + 1;

                // Partition the upper- and lower-dimensional simplices across
                // the connected components 
                Matrix<int, Dynamic, Dynamic> faces_in_components 
                    = Matrix<int, Dynamic, Dynamic>::Zero(n2_total, n_components); 
                Matrix<int, Dynamic, Dynamic> cofaces_in_components 
                    = Matrix<int, Dynamic, Dynamic>::Zero(n1_total, n_components);
                if (dim == 0)
                {
                    for (int i = 0; i < n2_total; ++i)
                        faces_in_components(i, components(i)) = 1;
                    for (int i = 0; i < n1_total; ++i)
                        cofaces_in_components(i, components(edges(i, 0))) = 1; 
                }
                else if (dim == 1)
                {
                    for (int i = 0; i < n2_total; ++i)
                        faces_in_components(i, components(edges(i, 0))) = 1;
                    for (int i = 0; i < n1_total; ++i)
                        cofaces_in_components(i, components(triangles(i, 0))) = 1; 
                }
                else    // dim == 2
                {
                    for (int i = 0; i < n2_total; ++i)
                        faces_in_components(i, components(triangles(i, 0))) = 1;
                    for (int i = 0; i < n1_total; ++i)
                        cofaces_in_components(i, components(tetrahedra(i, 0))) = 1; 
                }

                if (verbose)
                    std::cout << "... identified " << n_components << " connected components\n";

                // Get the boundary homomorphism from the (dim + 1)-th and 
                // (dim)-th chain groups
                Matrix<double, Dynamic, Dynamic> del = this->getRealBoundaryHomomorphism<double>(dim + 1);

                // Calculate the simplex weights, if not given
                Matrix<double, Dynamic, 1> simplex_weights(n2_total);  
                if (min_mode == CycleMinimizeMode::MINIMIZE_CYCLE_SIZE)
                {
                    simplex_weights = Matrix<double, Dynamic, 1>::Ones(n2_total); 
                }
                else if (min_mode == CycleMinimizeMode::MINIMIZE_CYCLE_WEIGHT)
                {
                    simplex_weights = weights; 
                }
                else    // min_mode == CycleMinimizeMode::MINIMIZE_CYCLE_VOLUME
                {
                    if (dim == 0)    // In this case, assign the same weight to each point 
                    {
                        simplex_weights = Matrix<double, Dynamic, 1>::Ones(n2_total); 
                    } 
                    else if (dim == 1)    // Calculate edge lengths 
                    {
                        for (int k = 0; k < n2_total; ++k)
                        {
                            int u = edges(k, 0); 
                            int v = edges(k, 1); 
                            simplex_weights(k) = static_cast<double>(
                                (this->points.row(u) - this->points.row(v)).matrix().norm()
                            ); 
                        }
                    }
                    else                  // Calculate triangle areas 
                    {
                        for (int k = 0; k < n2_total; ++k)
                        {
                            int u = triangles(k, 0); 
                            int v = triangles(k, 1); 
                            int w = triangles(k, 2); 

                            // Use Heron's formula 
                            double s1 = static_cast<double>((this->points.row(u) - this->points.row(v)).matrix().norm());
                            double s2 = static_cast<double>((this->points.row(u) - this->points.row(w)).matrix().norm()); 
                            double s3 = static_cast<double>((this->points.row(v) - this->points.row(w)).matrix().norm());
                            double s = 0.5 * (s1 + s2 + s3); 
                            simplex_weights(k) = sqrt(s * (s - s1) * (s - s2) * (s - s3)); 
                        }
                    }
                }
                if (verbose)
                    std::cout << "- Calculated simplex weights\n"; 
           
                // For each cycle ... 
                for (int j = 0; j < cycles.cols(); ++j)
                {
                    // Identify the connected component containing the cycle
                    Matrix<Z2, Dynamic, 1> cycle = cycles.col(j);
                    int component_idx; 
                    for (int k = 0; k < n2_total; ++k)
                    {
                        if (cycle(k) == 1)
                        {
                            // Get the first vertex in the k-th face
                            int v;
                            if (dim == 0)
                                v = k;
                            else if (dim == 1)
                                v = edges(k, 0); 
                            else    // dim == 2
                                v = triangles(k, 0);  
                            component_idx = components(v); 
                            break;  
                        }
                    }
                    if (verbose)
                    {
                        double weight = 0; 
                        for (int k = 0; k < cycle.size(); ++k)
                        {
                            if (cycle(k) == 1)
                                weight += simplex_weights(k); 
                        }
                        std::cout << "Minimizing cycle " << j << " ...\n"
                                  << "- Component index: " << component_idx << std::endl 
                                  << "- Initial weight: " << weight << std::endl;
                    } 

                    // Lift the cycle to one with real coefficients 
                    Matrix<double, Dynamic, 1> cycle_lifted
                        = this->liftCycle<Rational>(cycle, dim).template cast<double>(); 

                    // Identify the submatrix of del that pertains to this 
                    // connected component
                    std::vector<int> faces_in_component_idx, cofaces_in_component_idx; 
                    for (int k = 0; k < n2_total; ++k)
                    {
                        if (faces_in_components(k, component_idx) == 1)
                            faces_in_component_idx.push_back(k);
                    } 
                    for (int k = 0; k < n1_total; ++k)
                    {
                        if (cofaces_in_components(k, component_idx) == 1)
                            cofaces_in_component_idx.push_back(k); 
                    }
                    Matrix<double, Dynamic, Dynamic> del_sub = del(
                        faces_in_component_idx, cofaces_in_component_idx
                    );
                    Matrix<double, Dynamic, 1> cycle_sub = cycle_lifted(faces_in_component_idx);
                    Matrix<double, Dynamic, 1> weights_sub = simplex_weights(faces_in_component_idx); 
                    int n2 = faces_in_component_idx.size(); 
                    int n1 = cofaces_in_component_idx.size();
                    if (verbose)
                    {
                        std::cout << "- Lifted cycle to integer coefficients "
                                  << "and projected boundary homomorphism and "
                                  << "cycle coefficients\n" 
                                  << "- " << n2 << " faces, " << n1 << " cofaces\n"; 
                    } 
                   
                    // Now we define and solve the corresponding linear
                    // programming problem
                    //
                    // The constraints are always given by z = z1 + \del w, 
                    // where z1 is the basic cycle and w is any (dim + 1)-chain
                    //
                    // We define these constraints as A * x = b 
                    Matrix<double, Dynamic, Dynamic> A(n2, n1 + n2); 
                    Matrix<double, Dynamic, 1> b(n2);
                    A(Eigen::all, Eigen::seq(0, n2 - 1))
                        = Matrix<double, Dynamic, Dynamic>::Identity(n2, n2);
                    for (int k = 0; k < n2; ++k)
                    {
                        for (int m = n2; m < n1 + n2; ++m)
                        {
                            A(k, m) = -del_sub(k, m - n2);  
                        }
                        b(k) = cycle_sub(k);  
                    }
                    if (verbose)
                        std::cout << "- Defined linear program constraints\n"; 

                    // Define the linear program in standard format
                    CGAL::Quadratic_program<double> lp = defineL1LinearProgram<double>(n2, A, b, -1);

                    // Update the objective function with the weights 
                    const int nvars_total = n2 + n1; 
                    const int nvars_obj = n2; 
                    for (int k = 0; k < nvars_obj; ++k)
                        lp.set_c(nvars_total + k, weights_sub(k));

                    // Solve the linear program
                    CGAL::Quadratic_program_options options; 
                    options.set_verbosity(lp_verbose); 
                    Solution solution = CGAL::solve_linear_program(lp, ET(), options);
                    if (verbose)
                    {
                        std::cout << "- Solved linear program, optimal = "
                                  << solution.is_optimal() << ", infeasible = "
                                  << solution.is_infeasible() << ", unbounded = "
                                  << solution.is_unbounded() << ", objective = "
                                  << solution.objective_value() << std::endl;
                    } 

                    // The first n2 variables form the cycle coefficients 
                    Matrix<double, Dynamic, 1> opt_cycle(n2);
                    int k = 0;
                    for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
                    {
                        opt_cycle(k) = CGAL::to_double(*it); 
                        k++;
                        if (k == n2)
                            break; 
                    }

                    // Re-index the cycle coefficients w.r.t the simplex 
                    // ordering in the entire complex
                    for (int k = 0; k < n2; ++k)
                        opt_cycles(faces_in_component_idx[k], j) = opt_cycle(k); 
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
