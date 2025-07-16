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

TEST_CASE("Tests for getMinimumWeightPath()", "[getMinimumWeightPath()]")
{
    Graph graph; 
    std::vector<int> path; 

    // ------------------------------------------------------------ // 
    // Graph with three consecutive 4-cycles 
    // ------------------------------------------------------------ //
    graph = graph1();
    REQUIRE(boost::num_vertices(graph) == 8); 
    REQUIRE(boost::num_edges(graph) == 10); 

    // Path from 0 to itself 
    path = getMinimumWeightPath(graph, 0, 0); 
    REQUIRE(path.size() == 1);
    REQUIRE(path[0] == 0);

    // Path from 0 to 1
    path = getMinimumWeightPath(graph, 0, 1); 
    REQUIRE(path.size() == 2);
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1);

    // Path from 0 to 2
    path = getMinimumWeightPath(graph, 0, 2); 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1); 
    REQUIRE(path[2] == 2);

    // Path from 0 to 3 
    path = getMinimumWeightPath(graph, 0, 3); 
    REQUIRE(path.size() == 4); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 1); 
    REQUIRE(path[2] == 2); 
    REQUIRE(path[3] == 3);

    // Path from 0 to 4
    path = getMinimumWeightPath(graph, 0, 4); 
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 0); 
    REQUIRE(path[1] == 4);

    // Path from 0 to 5: either 0, 1, 5 or 0, 4, 5
    path = getMinimumWeightPath(graph, 0, 5); 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 0); 
    REQUIRE((path[1] == 1 || path[1] == 4));
    REQUIRE(path[2] == 5);

    // Path from 0 to 6: either 0, 1, 2, 6 or 0, 1, 5, 6 or 0, 4, 5, 6
    path = getMinimumWeightPath(graph, 0, 6); 
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
    path = getMinimumWeightPath(graph, 0, 7); 
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

    // Path from 1 to 0 
    path = getMinimumWeightPath(graph, 1, 0); 
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 0);  

    // Path from 1 to 2 
    path = getMinimumWeightPath(graph, 1, 2); 
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 2);

    // Path from 1 to 3 
    path = getMinimumWeightPath(graph, 1, 3); 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 2); 
    REQUIRE(path[2] == 3);

    // Path from 1 to 4
    path = getMinimumWeightPath(graph, 1, 4); 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1);
    REQUIRE((path[1] == 0 || path[1] == 5)); 
    REQUIRE(path[2] == 4); 

    // Path from 1 to 5
    path = getMinimumWeightPath(graph, 1, 5); 
    REQUIRE(path.size() == 2); 
    REQUIRE(path[0] == 1); 
    REQUIRE(path[1] == 5); 

    // Path from 1 to 6 
    path = getMinimumWeightPath(graph, 1, 6); 
    REQUIRE(path.size() == 3); 
    REQUIRE(path[0] == 1); 
    REQUIRE((path[1] == 2 || path[1] == 5)); 
    REQUIRE(path[2] == 6); 

    // Path from 1 to 7 
    path = getMinimumWeightPath(graph, 1, 7); 
    REQUIRE(path.size() == 4); 
    REQUIRE(path[0] == 1); 
    REQUIRE((path[1] == 2 || path[1] == 5)); 
    if (path[1] == 2)
        REQUIRE((path[2] == 3 || path[2] == 6)); 
    else 
        REQUIRE(path[2] == 6); 
    REQUIRE(path[3] == 7);  
}

TEST_CASE("Tests for getMinimumWeightPathTree()", "[getMinimumWeightPathTree()]")
{
}
