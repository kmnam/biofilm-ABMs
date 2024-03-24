/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/16/2024
 */

#ifndef CARTESIAN_PRODUCTS_HPP
#define CARTESIAN_PRODUCTS_HPP

#include <vector>

/**
 * Generate the Cartesian product of a given vector of vectors.
 */
std::vector<std::vector<int> > getProduct(const std::vector<std::vector<int> >& ranges)
{
    std::vector<std::vector<int> > result; 

    // Check the base case 
    if (ranges.empty())
        return result;

    // Check the case where there is only one range
    if (ranges.size() == 1)
    {
        for (const int x : ranges[0])
        {
            std::vector<int> v { x };
            result.push_back(v); 
        }
        return result; 
    }

    // Now skip over the first range and recurse through the others
    std::vector<std::vector<int> > ranges_(ranges.begin() + 1, ranges.end());
    std::vector<std::vector<int> > subproduct = getProduct(ranges_);

    // Combine the resulting subproduct with the first range
    for (const int x : ranges[0])
    {
        for (auto&& v : subproduct)
        {
            std::vector<int> w; 
            w.push_back(x); 
            for (const int y : v)
                w.push_back(y);
            result.push_back(w);
        }
    }

    return result;
}

#endif
