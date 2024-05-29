/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/29/2024
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Segment_3 Segment_3;

/**
 * Randomly sample a set of spherocylinders that contact the spherocylinder 
 * centered at the origin with orientation vector (0, 0, 1) and the given 
 * length and radius, and lie within the yz-plane (x == 0).
 *
 * The sampled spherocylinders can be distance between min_dist and max_dist
 * from the spherocylinder at the origin ("spherocylinder 1"), and have
 * lengths between min_l2 and max_l2, but have fixed radius R (which is the
 * radius of spherocylinder 1).
 *
 * Each spherocylinder vector is constrained to lie in the first quadrant or
 * orthant (x, y, z >= 0), and each orientation vector is constrained to have
 * positive z-coordinate.
 */
std::tuple<Matrix<double, Dynamic, 3>,
           Matrix<double, Dynamic, 3>,
           Matrix<double, Dynamic, 1>,
           Matrix<double, Dynamic, 1> >
    sampleSpherocylinders2D(const int nsample, const double l1, const double R,
                            const double min_dist, const double max_dist,
                            const double min_l2, const double max_l2,
                            const double min_theta, const double max_theta,
                            boost::random::mt19937& rng, const int burnin,
                            const int iter_per_sample)
{
    K kernel;
    boost::random::uniform_01<> uniform_dist;

    // Instantiate the spherocylinder at the origin
    Matrix<double, 3, 1> r1 = Matrix<double, 3, 1>::Zero(); 
    Matrix<double, 3, 1> n1; 
    n1 << 0, 0, 1;
    Spherocylinder<double> sc1(r1, n1, l1, R);
    Segment_3 seg1 = sc1.segment();
   
    // Generate an initial sample ... 
    //
    // Generate a center, orientation vector, and length such that the 
    // resulting spherocylinder has the desired distance to spherocylinder 1
    Matrix<double, 3, 1> r2, n2;
    double l2; 
    double dist = std::numeric_limits<double>::infinity();
    while (dist < min_dist || dist > max_dist)
    {
        // Generate a random orientation vector
        double theta = min_theta + (max_theta - min_theta) * uniform_dist(rng);
        double sin_theta = std::sin(theta); 
        double cos_theta = std::cos(theta);
        Matrix<double, 2, 2> rot; 
        rot << cos_theta, -sin_theta,
               sin_theta, cos_theta; 
        Matrix<double, 2, 1> v2, w2; 
        v2 << 0, 1; 
        w2 = rot * v2;
        n2 << 0, w2(0), w2(1); 

        // Generate a length 
        l2 = min_l2 + (max_l2 - min_l2) * uniform_dist(rng);

        // Generate a center, which should lie in the first quadrant/orthant
        // in the ball at the origin with radius l1 / 2 + max_dist + l2 / 2
        double min_rdist = min_dist;
        double max_rdist = l1 / 2 + max_dist + l2 / 2;
        double rtheta = boost::math::constants::two_pi<double>() * uniform_dist(rng);
        double rdist = min_rdist + (max_rdist - min_rdist) * uniform_dist(rng);
        r2 << 0, rdist * std::cos(rtheta), rdist * std::sin(rtheta);

        // Get the distance between spherocylinder 1 and the generated 
        // spherocylinder
        Spherocylinder<double> sc2(r2, n2, l2, R);  
        Segment_3 seg2 = sc2.segment();
        auto result = distBetweenSpherocylinders<double>(
            seg1, seg2, r1, n1, l1 / 2, r2, n2, l2 / 2, kernel
        );
        dist = std::get<0>(result).norm();
    }

    // Use Gibbs sampling to generate the full sample ... 
    const int ntotal = burnin + nsample * iter_per_sample; 
    Matrix<double, Dynamic, 3> sample_r(ntotal, 3);
    Matrix<double, Dynamic, 3> sample_n(ntotal, 3);
    Matrix<double, Dynamic, 1> sample_l(ntotal);
    Matrix<double, Dynamic, 1> sample_d(ntotal);
    sample_r.row(0) = r2.transpose(); 
    sample_n.row(0) = n2.transpose();
    sample_l(0) = l2;
    sample_d(0) = dist; 
    for (int i = 1; i < ntotal; ++i)
    {
        // Generate the orientation vector for the i-th sample based on
        // the (i-1)-th sample
        dist = std::numeric_limits<double>::infinity();
        while (dist < min_dist || dist > max_dist)
        {
            // Generate a random orientation vector
            double theta = min_theta + (max_theta - min_theta) * uniform_dist(rng);
            double sin_theta = std::sin(theta); 
            double cos_theta = std::cos(theta);
            Matrix<double, 2, 2> rot; 
            rot << cos_theta, -sin_theta,
                   sin_theta, cos_theta; 
            Matrix<double, 2, 1> v2, w2; 
            v2 << 0, 1; 
            w2 = rot * v2;
            n2 << 0, w2(0), w2(1); 

            // Update the orientation of spherocylinder 2 and calculate the
            // distance between the spherocylinders 
            Spherocylinder<double> sc2(r2, n2, l2, R);  
            Segment_3 seg2 = sc2.segment();
            auto result = distBetweenSpherocylinders<double>(
                seg1, seg2, r1, n1, l1 / 2, r2, n2, l2 / 2, kernel
            );
            dist = std::get<0>(result).norm();
        }

        // Generate a new length based on the newly sampled orientation and
        // the previously sampled center 
        dist = std::numeric_limits<double>::infinity();
        while (dist < min_dist || dist > max_dist)
        {
            l2 = min_l2 + (max_l2 - min_l2) * uniform_dist(rng);
            Spherocylinder<double> sc2(r2, n2, l2, R);  
            Segment_3 seg2 = sc2.segment();
            auto result = distBetweenSpherocylinders<double>(
                seg1, seg2, r1, n1, l1 / 2, r2, n2, l2 / 2, kernel
            );
            dist = std::get<0>(result).norm();
        }

        // Generate a new cell center based on the newly sampled orientation
        // and length
        dist = std::numeric_limits<double>::infinity();
        while (dist < min_dist || dist > max_dist)
        {
            double min_rdist = min_dist;
            double max_rdist = l1 / 2 + max_dist + l2 / 2;
            double rtheta = boost::math::constants::two_pi<double>() * uniform_dist(rng);
            double rdist = min_rdist + (max_rdist - min_rdist) * uniform_dist(rng);
            r2 << 0, rdist * std::cos(rtheta), rdist * std::sin(rtheta);
            Spherocylinder<double> sc2(r2, n2, l2, R);  
            Segment_3 seg2 = sc2.segment();
            auto result = distBetweenSpherocylinders<double>(
                seg1, seg2, r1, n1, l1 / 2, r2, n2, l2 / 2, kernel
            );
            dist = std::get<0>(result).norm();
        }

        // Update the sample 
        sample_r.row(i) = r2.transpose(); 
        sample_n.row(i) = n2.transpose(); 
        sample_l(i) = l2;
        sample_d(i) = dist;
    }

    // Get the desired subset of sampled coordinates 
    Matrix<double, Dynamic, 3> sample_r_final(nsample, 3); 
    Matrix<double, Dynamic, 3> sample_n_final(nsample, 3); 
    Matrix<double, Dynamic, 1> sample_l_final(nsample);
    Matrix<double, Dynamic, 1> sample_d_final(nsample); 
    for (int i = 0; i < nsample; ++i)
    {
        int j = burnin + (i + 1) * iter_per_sample - 1;
        sample_r_final.row(i) = sample_r.row(j); 
        sample_n_final.row(i) = sample_n.row(j); 
        sample_l_final(i) = sample_l(j);
        sample_d_final(i) = sample_d(j);
    }

    return std::make_tuple(sample_r_final, sample_n_final, sample_l_final, sample_d_final);
}


