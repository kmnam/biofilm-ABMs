/**
 * Various utility functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/5/2024
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

// Expose math functions for both standard and boost MPFR types
using std::sin;
using boost::multiprecision::sin;
using std::cos;
using boost::multiprecision::cos;

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
 * Sample a value from the von Mises distribution with the given mean and
 * concentration parameter.
 *
 * This is an implementation of Best & Fisher's algorithm.
 *
 * This function takes double parameters and returns a double.
 *
 * @param mu Mean. 
 * @param kappa Concentration parameter.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns A sampled value from the von Mises distribution.
 */
double vonMises(const double mu, const double kappa, boost::random::mt19937& rng,
                boost::random::uniform_01<>& uniform_dist)
{
    double tau = 1 + std::sqrt(1 + 4 * kappa * kappa);
    double rho = (tau - std::sqrt(2 * tau)) / (2 * kappa);
    double r = (1 + rho * rho) / (2 * rho);
    double z, f, c; 
    bool reject = true; 
    while (reject)
    {
        double u1 = uniform_dist(rng); 
        double u2 = uniform_dist(rng);
        z = cos(boost::math::constants::pi<double>() * u1);
        f = (1 + r * z) / (r + z); 
        c = kappa * (r - f); 
        if (c * (2 - c) - u2 > 0)
            reject = false;
        else if (std::log(c / u2) + 1 - c >= 0)
            reject = false;
    }

    double u3 = uniform_dist(rng); 
    if (u3 > 0.5)
        return std::fmod(std::acos(f) + mu, boost::math::constants::two_pi<double>());
    else if (u3 < 0.5)
        return std::fmod(-std::acos(f) + mu, boost::math::constants::two_pi<double>()); 
    else 
        return mu;
}

/**
 * Rotate the given orientation vector by the given Tait-Bryan angles. 
 *
 * @param n Input orientation vector.
 * @param alpha Angle to rotate about z-axis.
 * @param beta Angle to rotate about y-axis.
 * @param gamma Angle to rotate about x-axis.
 * @returns Rotated orientation vector.
 */
template <typename T>
Array<T, 3, 1> rotate(const Ref<const Array<T, 3, 1> >& n, const T alpha, 
                      const T beta, const T gamma)
{
    // Define rotation matrices about each axis and apply each rotation
    // in sequence 
    Matrix<T, 3, 3> Rx, Ry, Rz;
    T sin_alpha = sin(alpha);
    T cos_alpha = cos(alpha);
    T sin_beta = sin(beta);
    T cos_beta = cos(beta);
    T sin_gamma = sin(gamma);
    T cos_gamma = cos(gamma);
    Rx << 1, 0, 0,
          0, cos_gamma, -sin_gamma,
          0, sin_gamma, cos_gamma;
    Ry << cos_beta, 0, sin_beta,
          0, 1, 0,
          -sin_beta, 0, cos_beta;
    Rz << cos_alpha, -sin_alpha, 0,
          sin_alpha, cos_alpha, 0,
          0, 0, 1;
    return (Rz * Ry * Rx * n.matrix()).array();
}

/**
 * Rotate the given orientation vector by the given angle in the xy-plane.
 *
 * @param n Input orientation vector.
 * @param theta Angle to rotate (about z-axis). 
 * @returns Rotated orientation vector.
 */
template <typename T>
Array<T, 3, 1> rotateXY(const Ref<const Array<T, 3, 1> >& n, const T theta)
{
    Matrix<T, 3, 3> rot;
    T sin_theta = sin(theta);
    T cos_theta = cos(theta);
    rot << cos_theta, -sin_theta, 0,
           sin_theta, cos_theta, 0,
           0, 0, 1;
    return (rot * n.matrix()).array();
}

/**
 * Rotate the given orientation vector by the given angle out of the xy-plane,
 * maintaining the x- and y-orientations.
 *
 * This function rotates the given vector within the plane spanned by itself
 * and the z-unit vector by the given angle. 
 *
 * @param n Input orientation vector.
 * @param theta Angle to rotate (out of xy-plane). 
 * @returns Rotated orientation vector.
 */
template <typename T>
Array<T, 3, 1> rotateOutOfXY(const Ref<const Array<T, 3, 1> >& n, const T theta)
{
    // Get the unit vector along the axis of rotation
    Matrix<T, 3, 1> z; 
    z << 0, 0, 1;
    Matrix<T, 3, 1> v = n.matrix().cross(z);
    T norm = v.norm();
    v /= norm;

    // Use the Rodrigues' rotation formula to rotate the input vector
    T sin_theta = sin(theta);
    T cos_theta = cos(theta);
    Matrix<T, 3, 1> w = v.cross(n.matrix()); 
    return (n.matrix() * cos_theta + w * sin_theta).array();   // Third term is zero
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
bool distancesExceedThreshold(const Ref<const Array<T, Dynamic, 7> >& neighbors,
                              const T threshold)
{
    return (neighbors(Eigen::all, Eigen::seq(3, 5)).matrix().rowwise().norm().array() < threshold).any();
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
        cells(Eigen::all, Eigen::seq(0, 5)).isNaN().any() ||
        cells(Eigen::all, Eigen::seq(0, 5)).isInf().any()
    );
}

#endif
