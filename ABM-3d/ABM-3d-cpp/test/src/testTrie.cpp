/**
 * Test module for the `Trie` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/26/2025
 */
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/topology.hpp"

using namespace Eigen;

bool compareVectors(std::vector<int>& v, std::vector<int>& w)
{
    if (v.size() != w.size())
        return false; 

    for (int i = 0; i < v.size(); ++i)
    {
        if (v[i] != w[i])
            return false; 
    }

    return true; 
}

TEST_CASE("Tests for Trie methods", "[Trie]")
{
    // Generate a trie with the same structure as in the Wikipedia article:
    // https://en.wikipedia.org/wiki/Trie
    //
    // Here, we map letters to numbers in the standard way, as t = 20,
    // o = 15, e = 5, a = 1, A = 27, d = 4, n = 14, i = 9
    Trie trie;
    trie.insert({20, 15});    // "to"
    trie.insert({27});        // "A"
    trie.insert({9, 14});     // "in"

    // The trie should now have height 2 (excluding the root) and 6 nodes
    REQUIRE(trie.getHeight() == 2);
    REQUIRE(trie.getNumNodes() == 6);

    // Insert "tea" and "ted" 
    trie.insert({20, 5, 1});
    trie.insert({20, 5, 4});

    // The trie should now have height 3 (excluding the root) and 9 nodes 
    REQUIRE(trie.getHeight() == 3);
    REQUIRE(trie.getNumNodes() == 9);

    // Insert "ten" and "inn" 
    trie.insert({20, 5, 14});
    trie.insert({9, 14, 14});

    // The trie should now have height 3 (excluding the root) and 11 nodes 
    REQUIRE(trie.getHeight() == 3);
    REQUIRE(trie.getNumNodes() == 11); 

    // The trie should contain all the input strings 
    REQUIRE(trie.containsString({20, 15}));       // "to"
    REQUIRE(trie.containsString({27}));           // "A"
    REQUIRE(trie.containsString({20, 5, 1}));     // "tea"
    REQUIRE(trie.containsString({20, 5, 4}));     // "ted"
    REQUIRE(trie.containsString({20, 5, 14}));    // "ten"
    REQUIRE(trie.containsString({9, 14, 14}));    // "inn"

    // The trie should also contain substrings 
    REQUIRE(trie.containsString({20}));        // "t"
    REQUIRE(trie.containsString({9}));         // "i"
    REQUIRE(trie.containsString({20, 5}));     // "te"
    REQUIRE(trie.containsString({9, 14}));     // "in"
    REQUIRE(!trie.containsString({2})); 
    REQUIRE(!trie.containsString({17})); 
    REQUIRE(!trie.containsString({500})); 
    REQUIRE(!trie.containsString({9, 14, 5407})); 
    REQUIRE(!trie.containsString({1, 2, 3, 4, 5}));

    // Get all strings stored within the trie 
    //
    // Test that the strings are returned in lexicographical order 
    std::vector<std::vector<int> > strings = trie.getStrings();
    std::vector<std::vector<int> > strings_target = {
        {9}, {9, 14}, {9, 14, 14},
        {20}, {20, 5}, {20, 5, 1}, {20, 5, 4}, {20, 5, 14}, {20, 15},
        {27}
    }; 
    REQUIRE(strings.size() == strings_target.size()); 
    for (int i = 0; i < strings.size(); ++i)
        REQUIRE(compareVectors(strings[i], strings_target[i]));

    // Get all strings of length 3 within the trie 
    strings = trie.getStrings(true, true, 3); 
    strings_target = {{9, 14, 14}, {20, 5, 1}, {20, 5, 4}, {20, 5, 14}};
    REQUIRE(strings.size() == strings_target.size()); 
    for (int i = 0; i < strings.size(); ++i)
        REQUIRE(compareVectors(strings[i], strings_target[i]));

    // Get all substrings of the strings stored within the trie 
    //
    // Test that the substrings are returned in order of length, then in 
    // lexicographic order
    std::vector<std::vector<int> > substrings = trie.getSubstrings();
    std::vector<std::vector<int> > substrings_target = {
        {1}, {4}, {5}, {9}, {14}, {15}, {20}, {27},
        {5, 1}, {5, 4}, {5, 14}, {9, 14}, {14, 14},
        {20, 1}, {20, 4}, {20, 5}, {20, 14}, {20, 15}, 
        {9, 14, 14}, {20, 5, 1}, {20, 5, 4}, {20, 5, 14}
    };
    std::sort(substrings_target.begin(), substrings_target.end());  
    REQUIRE(substrings.size() == substrings_target.size()); 
    for (int i = 0; i < substrings.size(); ++i)
        REQUIRE(compareVectors(substrings[i], substrings_target[i]));

    // Get all substrings of length 2
    substrings = trie.getSubstrings(true, 2);
    substrings_target = {
        {5, 1}, {5, 4}, {5, 14}, {9, 14}, {14, 14},
        {20, 1}, {20, 4}, {20, 5}, {20, 14}, {20, 15} 
    };
    std::sort(substrings_target.begin(), substrings_target.end());  
    REQUIRE(substrings.size() == substrings_target.size()); 
    for (int i = 0; i < substrings.size(); ++i)
        REQUIRE(compareVectors(substrings[i], substrings_target[i]));

    // Get all superstrings of {20}
    //
    // Test that the superstrings are returned in lexicographic order
    std::vector<std::vector<int> > superstrings = trie.getSuperstrings({20});
    std::vector<std::vector<int> > superstrings_target = {
        {20}, {20, 5}, {20, 5, 1}, {20, 5, 4}, {20, 5, 14}, {20, 15}
    };
    std::sort(superstrings_target.begin(), superstrings_target.end()); 
    REQUIRE(superstrings.size() == superstrings_target.size()); 
    for (int i = 0; i < superstrings.size(); ++i)
        REQUIRE(compareVectors(superstrings[i], superstrings_target[i])); 

    // Get all superstrings of {20, 5} of length 3
    superstrings = trie.getSuperstrings({20, 5}, 3);
    superstrings_target = {
        {20, 5, 1}, {20, 5, 4}, {20, 5, 14}
    };
    std::sort(superstrings_target.begin(), superstrings_target.end()); 
    REQUIRE(superstrings.size() == superstrings_target.size()); 
    for (int i = 0; i < superstrings.size(); ++i)
        REQUIRE(compareVectors(superstrings[i], superstrings_target[i]));

    // Clear the trie and check that it only has one node 
    trie.clear();
    REQUIRE(trie.getHeight() == 0);  
    REQUIRE(trie.getNumNodes() == 1); 
} 
