/**
 * Generate a pair of slightly overlapping cells that are nearly parallel
 * with each other. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/24/2024
 */

#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include "../include/mechanics.hpp"
#include "../include/growth.hpp"

using namespace Eigen;

int main()
{
    boost::random::mt19937 rng(1234567890);
    double L0 = 1.0;
    double R = 0.8;

    // Randomly sample a direction in 2-D space
    boost::random::normal_distribution<> gaussian_dist(0.0, 1.0);
    Array<double, 2, 1> dir = Array<double, 2, 1>::Zero();
    dir << gaussian_dist(rng), gaussian_dist(rng);
    dir /= dir.matrix().norm();

    // Rotate the direction by 178 degrees counterclockwise
    double theta = (178. / 180.) * boost::math::constants::pi<double>();
    Matrix<double, 2, 2> rot; 
    rot << std::cos(theta), -std::sin(theta),
           std::sin(theta),  std::cos(theta);

    // Determine x- and y-coordinates of the two cell centers by projecting
    // along each direction by distance 0.9 * (R + L0 / 2)
    Array<double, 2, 1> dir2 = (rot * dir.matrix()).array();
    double delta = 0.9 * (R + L0 / 2);
    Array<double, 2, 1> r1xy = delta * dir;
    Array<double, 2, 1> r2xy = delta * dir2;
    Array<double, 3, 1> r1 = Array<double, 3, 1>::Zero();
    r1(0) = r1xy(0);
    r1(1) = r1xy(1);
    Array<double, 3, 1> r2 = Array<double, 3, 1>::Zero();
    r2(0) = r2xy(0);
    r2(1) = r2xy(1);

    // Set z-coordinates to intersect slightly with the surface
    r1(2) = 0.8 * R;
    r2(2) = 0.9 * R;

    // Determine x- and y-coordinates of the (unnormalized) two cell
    // orientation vectors
    Array<double, 3, 1> n1 = Array<double, 3, 1>::Zero();
    n1(0) = dir(0); 
    n1(1) = dir(1);
    Array<double, 3, 1> n2 = Array<double, 3, 1>::Zero();
    n2(0) = dir2(0); 
    n2(1) = dir2(1);

    // Set z-coordinates to small values 
    n1(2) = -std::asin(boost::math::constants::pi<double>() / 60); 
    n2(2) = -std::asin(boost::math::constants::pi<double>() / 90);
    n1 /= n1.matrix().norm();
    n2 /= n2.matrix().norm();

    std::cout << "r1 = " << r1.transpose() << std::endl;
    std::cout << "r2 = " << r2.transpose() << std::endl;
    std::cout << "n1 = " << n1.transpose() << std::endl;
    std::cout << "n2 = " << n2.transpose() << std::endl;
}
