/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/16/2024
 */

#include <iostream>
#include <vector>
#include "../../include/products.hpp"

int main()
{
    std::vector<std::vector<int> > ranges; 
    std::vector<int> x { 1, 2, 4, 5 };
    std::vector<int> y { 2, 7, 9 };
    std::vector<int> z { 0, 4, 6 };
    ranges.push_back(x);
    ranges.push_back(y);
    ranges.push_back(z);
    std::vector<std::vector<int> > product = getProduct(ranges);
    for (auto&& v : product)
    {
        for (const int x : v)
            std::cout << x << " ";
        std::cout << std::endl;
    } 
}
