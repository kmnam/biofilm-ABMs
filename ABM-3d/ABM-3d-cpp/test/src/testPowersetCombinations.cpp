/**
 * Test module for the `getPowerset()` and `getCombinations()` functions. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/25/2025
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"

TEST_CASE("Tests for powerset function", "[getPowerset()]")
{
    for (int i = 1; i <= 10; ++i)
    {
        // Define the set {1, ..., i}
        std::vector<int> set;
        for (int j = 1; j <= i; ++j)
            set.push_back(j); 

        // Get the powerset (excluding the empty set)
        std::vector<std::vector<int> > powerset = getPowerset(set);
        REQUIRE(powerset.size() == std::pow(2, i) - 1);
        
        // Manually generate the powerset 
        std::vector<std::vector<int> > powerset_true;
        powerset_true.push_back({}); 
        for (int j = 1; j <= i; ++j)
        {
            int n_subsets = powerset_true.size(); 
            for (int k = 0; k < n_subsets; ++k)
            {
                std::vector<int> subset2(powerset_true[k]);
                subset2.push_back(j);
                powerset_true.push_back(subset2); 
            }
        }

        // Check that the two powersets match  
        REQUIRE(powerset.size() == powerset_true.size() - 1); 
        for (auto&& subset : powerset_true)
        {
            if (subset.size() > 0)
            {
                auto it = std::find(powerset.begin(), powerset.end(), subset);
                REQUIRE(it != powerset.end());
            } 
        }
    }
}

TEST_CASE("Tests for combinations function", "[getCombinations()]")
{
    for (int n = 1; n <= 10; ++n)
    {
        // Define the set {1, ..., n}
        std::vector<int> set;
        for (int i = 1; i <= n; ++i)
            set.push_back(i);

        // Get all k-combinations, for k = 1, ..., n
        std::vector<std::vector<std::vector<int> > > combinations_all;
        for (int k = 1; k <= n; ++k)
            combinations_all.push_back(getCombinations(set, k));

        // Check the size of each set of k-combinations 
        for (int k = 1; k <= n; ++k)
        {
            int num = combinations_all[k - 1].size(); 
            REQUIRE(num == binom(n, k));  
        } 

        // Get the powerset (excluding the empty set)
        std::vector<std::vector<int> > powerset = getPowerset(set);
        
        // Check that the two collections match 
        for (auto&& subset : powerset)
        {
            int size = subset.size(); 
            auto it = std::find(
                combinations_all[size - 1].begin(),
                combinations_all[size - 1].end(), subset
            ); 
            REQUIRE(it != powerset.end());
        }
    }
}
