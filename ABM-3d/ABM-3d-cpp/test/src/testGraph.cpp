/**
 * Test module for functions in `../include/graphs.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/16/2025
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/graphs.hpp"

using namespace Eigen;

/**
 * Return true if the two arrays are the same up to a permutation of the
 * columns.
 *
 * @param A First array. 
 * @param B Second array. 
 * @returns True if the two arrays are the same up to a permutation of the 
 *          columns, false otherwise.  
 */
bool matchArraysUpToPermutedCols(const Ref<const Array<int, Dynamic, Dynamic> >& A,
                                 const Ref<const Array<int, Dynamic, Dynamic> >& B)
{
    // Check that the arrays have the same shape 
    if (A.rows() != B.rows() || A.cols() != B.cols())
        return false; 

    // Check that each column in A is present in B
    Array<int, Dynamic, 1> found_cols = Array<int, Dynamic, 1>::Zero(B.cols()); 
    for (int i = 0; i < A.cols(); ++i)
    {
        int found_col = -1; 
        
        // Run over all columns of B that have not already been accounted for
        for (int j = 0; j < B.cols(); ++j)
        {
            if (!found_cols(j) && (A.col(i) == B.col(j)).all())
            {
                found_col = j; 
                break;
            }
        }

        // If the column in A does not exist in B (ignoring all columns of 
        // B that have already contributed a column in A), return false 
        if (found_col == -1)
            return false;

        // Keep track of the contributing column in B
        found_cols(found_col) = 1;  
    }

    return true; 
}

/**
 * Generate a graph with three consecutive 4-cycles. 
 */
Graph graph1()
{
    Graph graph(8); 
    boost::add_edge(0, 1, EdgeProperty(1.0), graph); 
    boost::add_edge(1, 2, EdgeProperty(1.0), graph); 
    boost::add_edge(2, 3, EdgeProperty(1.0), graph); 
    boost::add_edge(0, 4, EdgeProperty(1.0), graph); 
    boost::add_edge(1, 5, EdgeProperty(1.0), graph); 
    boost::add_edge(2, 6, EdgeProperty(1.0), graph); 
    boost::add_edge(3, 7, EdgeProperty(1.0), graph); 
    boost::add_edge(4, 5, EdgeProperty(1.0), graph); 
    boost::add_edge(5, 6, EdgeProperty(1.0), graph); 
    boost::add_edge(6, 7, EdgeProperty(1.0), graph);

    return graph;  
}

TEST_CASE("Tests for getMinimumWeightPathTree()", "[getMinimumWeightPathTree()]")
{
    Graph graph; 

    // ------------------------------------------------------------ // 
    // Graph with three consecutive 4-cycles 
    // ------------------------------------------------------------ //
    graph = graph1();
    auto result = getMinimumWeightPathTree(graph, 0);
    std::vector<std::vector<int> > tree_paths = result.first; 
    Graph tree = result.second;
    REQUIRE(boost::num_vertices(tree) == 8); 
    REQUIRE(boost::num_edges(tree) == 7); 

    // Path from 0 to itself 
    REQUIRE(tree_paths[0].size() == 1);
    REQUIRE(tree_paths[0][0] == 0);

    // Path from 0 to 1
    REQUIRE(tree_paths[1].size() == 2);
    REQUIRE(tree_paths[1][0] == 0); 
    REQUIRE(tree_paths[1][1] == 1);

    // Path from 0 to 2
    REQUIRE(tree_paths[2].size() == 3); 
    REQUIRE(tree_paths[2][0] == 0); 
    REQUIRE(tree_paths[2][1] == 1); 
    REQUIRE(tree_paths[2][2] == 2);

    // Path from 0 to 3 
    REQUIRE(tree_paths[3].size() == 4); 
    REQUIRE(tree_paths[3][0] == 0); 
    REQUIRE(tree_paths[3][1] == 1); 
    REQUIRE(tree_paths[3][2] == 2); 
    REQUIRE(tree_paths[3][3] == 3);

    // Path from 0 to 4
    REQUIRE(tree_paths[4].size() == 2); 
    REQUIRE(tree_paths[4][0] == 0); 
    REQUIRE(tree_paths[4][1] == 4);

    // Path from 0 to 5: should be either 0, 1, 5 or 0, 4, 5
    REQUIRE(tree_paths[5].size() == 3); 
    REQUIRE(tree_paths[5][0] == 0); 
    REQUIRE((tree_paths[5][1] == 1 || tree_paths[5][1] == 4));
    REQUIRE(tree_paths[5][2] == 5);

    // Path from 0 to 6: should be either the path from 0 to 5 plus 6, or 
    // the path from 0 to 2 plus 6
    REQUIRE(tree_paths[6].size() == 4);
    REQUIRE(tree_paths[6][0] == 0);
    REQUIRE((tree_paths[6][2] == tree_paths[2][2] ^ tree_paths[6][2] == tree_paths[5][2]));
    if (tree_paths[6][2] == tree_paths[2][2])
        REQUIRE(tree_paths[6][1] == tree_paths[2][1]);
    else 
        REQUIRE(tree_paths[6][1] == tree_paths[5][1]);
    REQUIRE(tree_paths[6][3] == 6);

    // Path from 0 to 7: should be either the path from 0 to 6 plus 7, or 
    // the path from 0 to 3 plus 7
    REQUIRE(tree_paths[7].size() == 5);
    REQUIRE(tree_paths[7][0] == 0);
    REQUIRE((tree_paths[7][3] == tree_paths[3][3] ^ tree_paths[7][3] == tree_paths[6][3]));
    if (tree_paths[7][3] == tree_paths[3][3])
    {
        REQUIRE(tree_paths[7][1] == tree_paths[3][1]); 
        REQUIRE(tree_paths[7][2] == tree_paths[3][2]); 
    }
    else 
    {
        REQUIRE(tree_paths[7][1] == tree_paths[6][1]); 
        REQUIRE(tree_paths[7][2] == tree_paths[6][2]); 
    }
    REQUIRE(tree_paths[7][4] == 7);

    // Check that the tree contains all requisite edges 
    REQUIRE(boost::edge(0, 1, tree).second);
    REQUIRE(boost::edge(1, 2, tree).second);
    REQUIRE(boost::edge(2, 3, tree).second); 
    REQUIRE(boost::edge(0, 4, tree).second);  
    REQUIRE((boost::edge(1, 5, tree).second ^ boost::edge(4, 5, tree).second)); 
    REQUIRE((boost::edge(5, 6, tree).second ^ boost::edge(2, 6, tree).second));
    REQUIRE((boost::edge(6, 7, tree).second ^ boost::edge(3, 7, tree).second));  
}

TEST_CASE("Tests for getPathInMinimumWeightPathTree()", "[getPathInMinimumWeightPathTree()]")
{
    Graph graph; 

    // ------------------------------------------------------------ // 
    // Graph with three consecutive 4-cycles 
    // ------------------------------------------------------------ //
    graph = graph1();
    auto result = getMinimumWeightPathTree(graph, 0);
    std::vector<std::vector<int> > tree_paths = result.first; 
    Graph tree = result.second;

    // Define the parent and depth of each vertex with respect to the root
    int nv = boost::num_vertices(graph);  
    std::vector<int> parents(nv), depths(nv);
    parents[0] = -1;    // The root has no parent 
    depths[0] = 0;      // The root has depth zero
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

    // Check the path between each pair of vertices 
    for (int i = 0; i < nv; ++i)
    {
        for (int j = i + 1; j < nv; ++j)
        {
            std::vector<int> path = getPathInMinimumWeightPathTree(
                tree_paths, parents, depths, i, j
            ); 
            REQUIRE(path[0] == i); 
            REQUIRE(path[path.size() - 1] == j);

            // Check that each edge in the path lies in the tree  
            for (auto it = path.begin() + 1; it != path.end(); ++it)
            {
                int u = *std::prev(it); 
                int v = *it; 
                REQUIRE(boost::edge(u, v, tree).second); 
            }

            // Check that there are no repeated vertices along the path 
            REQUIRE(std::unordered_set<int>(path.begin(), path.end()).size() == path.size()); 
        }
    }
}

TEST_CASE("Tests for getMinimumWeightPaths()", "[getMinimumWeightPaths()]")
{
    Graph graph; 
    std::vector<int> path; 

    // ------------------------------------------------------------ // 
    // Graph with three consecutive 4-cycles 
    // ------------------------------------------------------------ //
    graph = graph1();
    REQUIRE(boost::num_vertices(graph) == 8); 
    REQUIRE(boost::num_edges(graph) == 10);

    // Get all minimum-weight paths
    std::map<std::pair<int, int>, std::vector<int> > min_paths = getMinimumWeightPaths(graph);  

    // Path from 0 to 1
    path = min_paths[std::make_pair(0, 1)]; 
    REQUIRE(path.size() == 2);
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1);

    // Path from 0 to 2
    path = min_paths[std::make_pair(0, 2)];
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1); 
    REQUIRE(path[2] == 2);

    // Path from 0 to 3 
    path = min_paths[std::make_pair(0, 3)];
    REQUIRE(path.size() == 4); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1); 
    REQUIRE(path[2] == 2); 
    REQUIRE(path[3] == 3);

    // Path from 0 to 4
    path = min_paths[std::make_pair(0, 4)];
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 4);

    // Path from 0 to 5: either 0, 1, 5 or 0, 4, 5
    path = min_paths[std::make_pair(0, 5)]; 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 0); 
    REQUIRE((path[1] == 1 || path[1] == 4));
    REQUIRE(path[2] == 5);

    // Path from 0 to 6: either 0, 1, 2, 6 or 0, 1, 5, 6 or 0, 4, 5, 6
    path = min_paths[std::make_pair(0, 6)];
    REQUIRE(path.size() == 4);
    REQUIRE(path[0] == 0); 
    REQUIRE((path[1] == 1 || path[1] == 4)); 
    if (path[1] == 1)
        REQUIRE((path[2] == 2 || path[2] == 5));
    else 
        REQUIRE(path[2] == 5); 
    REQUIRE(path[3] == 6);

    // Path from 0 to 7: either 0, 1, 2, 3, 7 or 0, 1, 2, 6, 7 or 0, 1, 5, 6, 7
    // or 0, 4, 5, 6, 7
    path = min_paths[std::make_pair(0, 7)];
    REQUIRE(path.size() == 5);
    REQUIRE(path[0] == 0); 
    REQUIRE((path[1] == 1 || path[1] == 4)); 
    if (path[1] == 1)
    {
        REQUIRE((path[2] == 2 || path[2] == 5));
        if (path[2] == 2)
            REQUIRE((path[3] == 3 || path[3] == 6));
        else 
            REQUIRE(path[3] == 6);  
    }
    else
    { 
        REQUIRE(path[2] == 5);
        REQUIRE(path[3] == 6); 
    } 
    REQUIRE(path[4] == 7);

    // Path from 1 to 2 
    path = min_paths[std::make_pair(1, 2)];
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 2);

    // Path from 1 to 3 
    path = min_paths[std::make_pair(1, 3)];
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 2); 
    REQUIRE(path[2] == 3);

    // Path from 1 to 4
    path = min_paths[std::make_pair(1, 4)];
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1);
    REQUIRE((path[1] == 0 || path[1] == 5)); 
    REQUIRE(path[2] == 4); 

    // Path from 1 to 5
    path = min_paths[std::make_pair(1, 5)];
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 5); 

    // Path from 1 to 6 
    path = min_paths[std::make_pair(1, 6)];
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1); 
    REQUIRE((path[1] == 2 || path[1] == 5)); 
    REQUIRE(path[2] == 6); 

    // Path from 1 to 7 
    path = min_paths[std::make_pair(1, 7)];
    REQUIRE(path.size() == 4); 
    REQUIRE(path[0] == 1); 
    REQUIRE((path[1] == 2 || path[1] == 5)); 
    if (path[1] == 2)
        REQUIRE((path[2] == 3 || path[2] == 6)); 
    else 
        REQUIRE(path[2] == 6); 
    REQUIRE(path[3] == 7);  
}

TEST_CASE("Tests for getMinimumCycleBasis()", "[getMinimumCycleBasis()]")
{
    Graph graph; 

    // ------------------------------------------------------------ // 
    // Graph with three consecutive 4-cycles 
    // ------------------------------------------------------------ //
    graph = graph1();
    Matrix<int, Dynamic, Dynamic> basis = getMinimumCycleBasis<int>(graph);

    // The edges are ordered lexicographically as: 
    // (0,1), (0,4), (1,2), (1,5), (2,3), (2,6), (3,7), (4,5), (5,6), (6,7)
    Matrix<int, Dynamic, Dynamic> cycles(boost::num_edges(graph), 3);
    cycles << 1, 0, 0,
              1, 0, 0,
              0, 1, 0,
              1, 1, 0,
              0, 0, 1,
              0, 1, 1,
              0, 0, 1, 
              1, 0, 0,
              0, 1, 0,
              0, 0, 1;
    REQUIRE(matchArraysUpToPermutedCols(basis.array(), cycles.array())); 
}

