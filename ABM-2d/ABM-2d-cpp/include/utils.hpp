/**
 * Various utility functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/16/2024
 */

#ifndef BIOFILM_UTILS_HPP
#define BIOFILM_UTILS_HPP

#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <boost/json/src.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>

using std::sin;
using boost::multiprecision::sin;
using std::cos;
using boost::multiprecision::cos;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::log;
using boost::multiprecision::log;

/**
 * Parse a JSON file specifying simulation parameters.
 *
 * @param filename Input JSON configurations file.
 * @returns `boost::json::value` instance containing the JSON data.  
 */
boost::json::value parseConfigFile(const std::string filename)
{
    std::string line;
    std::ifstream infile(filename);
    boost::json::stream_parser p; 
    boost::json::error_code ec;  
    while (std::getline(infile, line))
    {
        p.write(line, ec); 
        if (ec)
            return nullptr; 
    }
    p.finish(ec); 
    if (ec)
        return nullptr;
    
    return p.release(); 
}

/**
 * Write a population of cells, with the corresponding simulation parameters,
 * to the given path. 
 *
 * @param cells Existing population of cells.
 * @param params `std::map<std::string, std::string>` instance containing the
 *               simulation parameters.
 * @param filename Output file. 
 */
template <typename T>
void writeCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                std::map<std::string, std::string>& params, 
                const std::string filename)
{
    // Open output file 
    std::ofstream outfile(filename);
    
    // Run through the simulation parameters ... 
    for (auto&& param : params)
        outfile << "# " << param.first << " = " << param.second << std::endl;

    // Write each cell in the population ...
    outfile << std::setprecision(10);  
    for (int i = 0; i < cells.rows(); ++i)
    {
        for (int j = 0; j < cells.cols(); ++j)
        {
            if (j == 13)         // If the entry is a group identifier
                outfile << static_cast<int>(cells(i, j)) << '\t'; 
            else                 // Otherwise 
                outfile << cells(i, j) << '\t'; 
        }
        outfile.seekp(-1, std::ios_base::cur);
        outfile << std::endl;
    }

    // Close output file 
    outfile.close();  
}

/**
 * Write a population of cells, with the corresponding simulation parameters,
 * to the given path. 
 *
 * @param cells Existing population of cells.
 * @param params `boost::json::object` instance containing the JSON data. 
 * @param filename Output file. 
 */
template <typename T>
void writeCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                boost::json::object& params, const std::string filename)
{
    // Open output file 
    std::ofstream outfile(filename);
    
    // Run through the simulation parameters ... 
    for (auto&& param : params)
    {
        //std::string key(std::string_view(param.key().data(), param.key().size());
        std::string key(param.key().data()); 
        outfile << "# " << key << " = " << params.at(key) << std::endl;
    }

    // Write each cell in the population ...
    outfile << std::setprecision(10);  
    for (int i = 0; i < cells.rows(); ++i)
    {
        for (int j = 0; j < cells.cols(); ++j)
        {
            if (j == 10)         // If the entry is a group identifier
                outfile << static_cast<int>(cells(i, j)) << '\t'; 
            else                 // Otherwise 
                outfile << cells(i, j) << '\t'; 
        }
        outfile.seekp(-1, std::ios_base::cur);
        outfile << std::endl;
    }

    // Close output file 
    outfile.close();  
}

/**
 * Sample a value from the standard normal distribution with the Box-Muller
 * method.
 *
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled value. 
 */
template <typename T>
T standardNormal(boost::random::mt19937& rng, boost::random::uniform_01<>& uniform_dist)
{
    T u = static_cast<T>(uniform_dist(rng)); 
    T v = static_cast<T>(uniform_dist(rng)); 
    T c = sqrt(-2 * log(u));
    return c * cos(boost::math::constants::two_pi<T>() * v); 
}

/**
 * Sample a value from the von Mises distribution with the given mean and
 * concentration parameter.
 *
 * This is an implementation of Best & Fisher's algorithm. 
 *
 * @param mu Mean. 
 * @param kappa Concentration parameter.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns A sampled value from the von Mises distribution.
 */
template <typename T>
T vonMises(const T mu, const T kappa, boost::random::mt19937& rng,
           boost::random::uniform_01<>& uniform_dist)
{
    T tau = 1 + std::sqrt(1 + 4 * kappa * kappa);
    T rho = (tau - std::sqrt(2 * tau)) / (2 * kappa);
    T r = (1 + rho * rho) / (2 * rho);
    T z, f, c; 
    bool reject = true; 
    while (reject)
    {
        T u1 = uniform_dist(rng); 
        T u2 = uniform_dist(rng);
        z = std::cos(boost::math::constants::pi<T>() * u1);
        f = (1 + r * z) / (r + z); 
        c = kappa * (r - f); 
        if (c * (2 - c) - u2 > 0)
            reject = false;
        else if (std::log(c / u2) + 1 - c >= 0)
            reject = false;
    }

    T u3 = uniform_dist(rng); 
    if (u3 > 0.5)
        return std::fmod(std::acos(f) + mu, boost::math::constants::two_pi<T>());
    else if (u3 < 0.5)
        return std::fmod(-std::acos(f) + mu, boost::math::constants::two_pi<T>()); 
    else 
        return mu;
}

/**
 * Sample `k` items from the range from `0` to `n - 1` without replacement. 
 *
 * It is assumed that `k <= n`. 
 */
std::vector<int> sampleWithoutReplacement(const int n, const int k,
                                          boost::random::mt19937& rng)
{
    if (k < 0)
    {
        throw std::invalid_argument("Cannot sample k items from list of n if k < 0"); 
    }
    else if (n < 1) 
    {
        throw std::invalid_argument("Cannot sample k items from list of n if n < 1"); 
    }
    if (k > n)
    {
        throw std::invalid_argument("Cannot sample k items from list of n if k > n");
    }
    else if (k == n)
    {
        std::vector<int> sample; 
        for (int i = 0; i < n; ++i)
            sample.push_back(i); 
        return sample; 
    }
    else    // 0 <= k < n and n >= 1
    {
        // Initialize an array with 0, ..., n - 1
        std::vector<int> array; 
        for (int i = 0; i < n; ++i)
            array.push_back(i);

        // Perform a Fisher-Yates shuffle
        for (int i = n - 1; i > 0; --i)
        {
            boost::random::uniform_int_distribution<> dist(0, i); 
            int j = dist(rng);
            int arr_i = array[i]; 
            array[i] = array[j]; 
            array[j] = arr_i; 
        }

        // Return the first k items in the shuffled array 
        return std::vector<int>(array.begin(), array.begin() + k); 
    }
}

/**
 * Rotate the given orientation vector by the given angle counterclockwise
 * in the xy-plane.
 *
 * @param n Input orientation vector.
 * @param theta Input angle. 
 * @returns Rotated orientation vector.
 */
template <typename T>
Array<T, 2, 1> rotate(const Ref<const Array<T, 2, 1> >& n, const T theta)
{
    Matrix<T, 2, 2> rot; 
    rot << std::cos(theta), -std::sin(theta),
           std::sin(theta), std::cos(theta); 
    return (rot * n.matrix()).array();
}

/**
 * Check that all cell-cell distances in the given array of neighboring 
 * cells are greater than some threshold.
 *
 * @param neighbors Array of neighboring pairs of cells.
 * @param threshold Distance threshold.
 * @returns True if the cell-cell distances exceed the given threshold, 
 *          false otherwise. 
 */
template <typename T>
bool distancesExceedThreshold(const Ref<const Array<T, Dynamic, 6> >& neighbors,
                              const T threshold)
{
    return (neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array() < threshold).any();
}

/**
 * Check that the given cell coordinates contain a NaN or infinity. 
 *
 * @param cells Existing population of cells.
 * @returns True if the cell coordinates contain a NaN or infinity, false
 *          otherwise.
 */
template <typename T>
bool isNaNOrInf(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    return (
        cells(Eigen::all, Eigen::seq(0, 3)).isNaN().any() ||
        cells(Eigen::all, Eigen::seq(0, 3)).isInf().any()
    );
}

#endif
