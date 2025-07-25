/**
 * Functions for pre-patterning populations of cells. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/23/2025
 */

#ifndef BIOFILM_PREPATTERN_HPP 
#define BIOFILM_PREPATTERN_HPP

#include <Eigen/Dense>
#include <boost/random.hpp>
#include "indices.hpp"

using namespace Eigen; 

enum PrepatternMode
{
    RANDOM, 
    INNERMOST,
    OUTERMOST
}; 

/**
 * @param cells
 * @param mode
 * @param fraction
 * @param group_attributes
 * @param growth_dists
 * @param attribute_dists
 * @param rng
 */
template <typename T>
void prepattern(Ref<Array<T, Dynamic, Dynamic> > cells, const PrepatternMode mode,
                const double fraction, std::vector<int>& group_attributes,
                std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                std::map<std::pair<int, int>, std::function<T(boost::random::mt19937&)> >& attribute_dists,
                boost::random::mt19937& rng)
{
    // Fix all cells as group 1
    const int n = cells.rows(); 
    for (int i = 0; i < n; ++i)
        cells(i, __colidx_group) = 1;

    // Fix fraction of cells to switch to group 2
    const int m = static_cast<int>(fraction * n);
    if (m >= n)
        throw std::runtime_error(
            "Fraction of cells to be switched while pre-patterning must be less than 1"
        );
    else if (m < 0)
        throw std::runtime_error(
            "Fraction of cells to be switched while pre-patterning must be positive"
        );

    // Number of attributes to switch 
    const int n_attributes = group_attributes.size();

    // Indices to switch
    std::vector<int> idx; 

    if (mode == PrepatternMode::RANDOM)
    {
        std::vector<int> sample = sampleWithoutReplacement(n, m, rng);
        for (auto it = sample.begin(); it != sample.end(); ++it)
        {
            int k = *it;
            idx.push_back(k); 
        }
    }
    else    // mode == INNERMOST or mode == OUTERMOST
    {
        // Get the center of mass of the population 
        T rx_mean = cells.col(__colidx_rx).sum() / n; 
        T ry_mean = cells.col(__colidx_ry).sum() / n;
        Matrix<T, 1, 2> r_mean; 
        r_mean << rx_mean, ry_mean; 

        // Get the distance of each cell center to the center of mass 
        std::vector<T> dists; 
        for (int i = 0; i < n; ++i)
            dists.push_back((cells(i, __colseq_r).matrix() - r_mean).norm());

        // Sort the distances and identify the cutoff distance by which 
        // to change the innermost/outermost cells to group 2
        if (mode == PrepatternMode::INNERMOST)
        {
            // Sort the distances in ascending order
            std::vector<T> dists2(dists); 
            std::sort(dists2.begin(), dists2.end());

            // Get the cutoff distance
            T cutoff = dists2[m];

            // Find all cells with distance less than the cutoff 
            for (int i = 0; i < n; ++i)
            {
                if (dists[i] < cutoff)
                    idx.push_back(i); 
            }
        }
        else    // mode == PrepatternMode::OUTERMOST
        {
            // Sort the distances in descending order
            std::vector<T> dists2(dists); 
            std::sort(dists2.begin(), dists2.end(), [](T a, T b){ return a > b; });

            // Get the cutoff distance
            T cutoff = dists2[m];

            // Find all cells with distance greater than the cutoff
            for (int i = 0; i < n; ++i)
            {
                if (dists[i] > cutoff)
                    idx.push_back(i);
            }
        }
    }

    // Switch the chosen cells to group 2
    for (const int k : idx)
        cells(k, __colidx_group) = 2;
    
    // Reset all attributes according to each group 
    for (int i = 0; i < n; ++i)
    {
        int group = cells(i, __colidx_group); 
        T growth_rate = growth_dists[group - 1](rng); 
        cells(i, __colidx_growth) = growth_rate; 
        for (int j = 0; j < n_attributes; ++j)
        {
            auto pair = std::make_pair(group - 1, j);
            T attribute = attribute_dists[pair](rng); 
            cells(i, group_attributes[j]) = attribute;
        }
    }
}

#endif 
