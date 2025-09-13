/**
 * Various utility functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/25/2025
 */

#ifndef BIOFILM_UTILS_3D_HPP
#define BIOFILM_UTILS_3D_HPP

#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <vector>
#include <stack>
#include <regex>
#include <algorithm>
#include <filesystem>
#include <boost/json/src.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/Gmpz.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include "indices.hpp"
#include "distances.hpp"

using std::abs;
using boost::multiprecision::abs;
using std::sin;
using boost::multiprecision::sin;
using std::cos;
using boost::multiprecision::cos;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::log;
using boost::multiprecision::log;
using std::pow; 
using boost::multiprecision::pow;
using std::ceil; 
using boost::multiprecision::ceil;
using std::acos; 
using boost::multiprecision::acos; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3; 
typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef Polyhedron_3::HalfedgeDS HalfedgeDS;

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
 * Read a file containing data for a population of cells, with the corresponding
 * simulation parameters.
 *
 * @param filename Input file.
 * @returns Array of cell data, together with a dictionary of simulation 
 *          parameters.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::map<std::string, std::string> > readCells(const std::string filename)
{
    // Open input file 
    std::ifstream infile(filename);
    std::map<std::string, std::string> params;
    Array<T, Dynamic, Dynamic> cells(0, __ncols_required);
    int nrows = 0;
    int ncols = __ncols_required;  

    // For each line in the file ... 
    std::string line;
    while (std::getline(infile, line))
    {
        // Check if the line specifies a simulation parameter 
        if (line[0] == '#')
        {
            std::string token = line.substr(2, line.find(" = ") - 2);   // Remove leading "# "
            line.erase(0, line.find(" = ") + 3);
            params[token] = line;  
        }
        // Otherwise, read in the cell array coordinates 
        else 
        {
            std::stringstream ss; 
            std::string token;
            ss << line;
            nrows++; 
            cells.conservativeResize(nrows, ncols);
            int j = 0;  
            while (std::getline(ss, token, '\t'))
            {
                if (j >= ncols)
                {
                    ncols++; 
                    cells.conservativeResize(nrows, ncols); 
                }
                cells(nrows - 1, j) = static_cast<T>(std::stod(token)); 
                j++; 
            } 
        }
    }

    return std::make_pair(cells, params);  
}

/**
 * Write a population of cells, with the corresponding simulation parameters,
 * to the given path. 
 *
 * @param cells Existing population of cells.
 * @param params `std::map<std::string, std::string>` instance containing the
 *               simulation parameters.
 * @param filename Output file.
 * @param write_other_cols A map of extra column indices and their corresponding
 *                         types. 
 */
template <typename T>
void writeCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                std::map<std::string, std::string>& params, 
                const std::string filename,
                std::unordered_map<int, int>& write_other_cols)
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
            // If the entry is a group or cell identifier, write as an integer
            if (j == __colidx_id || j == __colidx_group)
            {
                outfile << static_cast<int>(cells(i, j)) << '\t';
            }
            // If the entry is another required entry, write as a double 
            else if (j > __colidx_id && j < __colidx_group)
            {
                outfile << cells(i, j) << '\t'; 
            }
            // If the entry is an extra column, write as desired type 
            else if (write_other_cols.count(j))
            {
                int type = write_other_cols[j]; 
                if (type == 0)
                    outfile << static_cast<int>(cells(i, j)) << '\t';
                else 
                    outfile << cells(i, j) << '\t'; 
            }
        }
        outfile.seekp(-1, std::ios_base::cur);   // Remove last tab
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
 * @param write_other_cols A map of extra column indices and their corresponding
 *                         types. 
 */
template <typename T>
void writeCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                boost::json::object& params, const std::string filename,
                std::unordered_map<int, int>& write_other_cols)
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
            // If the entry is a group or cell identifier, write as an integer
            if (j == __colidx_id || j == __colidx_group)
            {
                outfile << static_cast<int>(cells(i, j)) << '\t';
            }
            // If the entry is another required entry, write as a double 
            else if (j > __colidx_id && j < __colidx_group)
            {
                outfile << cells(i, j) << '\t'; 
            }
            // If the entry is an extra column, write as desired type 
            else if (write_other_cols.count(j))
            {
                int type = write_other_cols[j]; 
                if (type == 0)
                    outfile << static_cast<int>(cells(i, j)) << '\t';
                else 
                    outfile << cells(i, j) << '\t'; 
            }
        }
        outfile.seekp(-1, std::ios_base::cur);   // Remove last tab
        outfile << std::endl;
    }

    // Close output file 
    outfile.close();  
}

/**
 * Read the given lineage file. 
 *
 * @param filename Input lineage filename. 
 * @returns A dictionary containing the parent cell ID for each cell. 
 */
std::unordered_map<int, int> readLineage(const std::string filename)
{
    // Parse the lineage file 
    std::unordered_map<int, int> parents; 
    std::ifstream infile(filename); 
    std::string line, token;
    std::stringstream ss; 
    while (std::getline(infile, line))
    {
        // Each line contains the cell ID and the corresponding parent ID
        // (with parent ID = -1 for the initial cell)
        ss << line; 
        std::getline(ss, token, '\t');
        int child = std::stoi(token); 
        std::getline(ss, token); 
        int parent = std::stoi(token);
        parents[child] = parent; 
        ss.str(std::string()); 
        ss.clear(); 
    }

    return parents; 
}

/**
 * Write a lineage of cells to the given path. 
 *
 * @param parents Vector of parent IDs for each cell generated throughout the
 *                simulation.
 * @param filename Output file. 
 */
template <typename T>
void writeLineage(const std::vector<int>& parents, const std::string filename)
{
    // Open output file 
    std::ofstream outfile(filename);
    
    // Run through the parent IDs and write them one by one
    int i = 0; 
    for (auto&& parent : parents)
    {
        outfile << i << "\t" << parent << std::endl;
        i++;
    }
    
    // Close output file 
    outfile.close();  
}

/**
 * Parse a collection of simulation frames in the given directory, in order
 * of iteration number.
 *
 * @param dir Input directory.
 * @param nmax If given as a positive number, return a subcollection of the
 *             given number of filenames that are equally spaced in time. 
 * @param tmin Filter out all files with timepoints earlier than this time.
 * @param tmax Filter out all files with timepoints later than this time. 
 * @returns List of simulation filenames, in order of iteration number.
 */
std::vector<std::string> parseDir(const std::string dir, const int nmax = 0, 
                                  const double tmin = 0,
                                  const double tmax = std::numeric_limits<double>::max())
{
    // Store a vector of filenames and their timepoints 
    std::vector<std::pair<std::string, double> > filenames;

    // For each file in the input directory ... 
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        std::string filename = entry.path(); 

        // Check that the filename ends with '.txt'
        const int fsize = filename.size(); 
        if (fsize >= 4 && filename.compare(fsize - 4, fsize, ".txt") == 0)
        {
            // Skip over the lineage file 
            if (filename.compare(fsize - 12, fsize, "_lineage.txt") == 0)
                continue;

            const std::regex re(R"(iter(\d+)\.txt$)");
            std::smatch m;  

            // If the file contains the initial simulation frame, collect
            // it with time 0
            if (filename.compare(fsize - 9, fsize, "_init.txt") == 0)
            {
                if (tmin == 0)
                    filenames.push_back(std::make_pair(filename, 0));
            }
            // If the file contains the final simulation frame, collect it 
            // with its timepoint 
            else if (filename.compare(fsize - 10, fsize, "_final.txt") == 0)
            {
                auto result = readCells<double>(filename);
                double time = std::stod(result.second["t_curr"]);
                if (time >= tmin && time <= tmax) 
                    filenames.push_back(std::make_pair(filename, time)); 
            }
            // If the file contains an intermediate simulation frame, collect
            // it with its timepoint 
            else if (std::regex_search(filename, m, re))
            {
                auto result = readCells<double>(filename); 
                double time = std::stod(result.second["t_curr"]);
                if (time >= tmin && time <= tmax) 
                    filenames.push_back(std::make_pair(filename, time));
            } 
        }
    }

    // Raise an exception if there were no files 
    if (filenames.size() == 0)
        throw std::runtime_error("No simulation frames found in given directory"); 

    // Sort the filenames by iteration number 
    std::sort(
        filenames.begin(), filenames.end(),
        [](const std::pair<std::string, double>& x, const std::pair<std::string, double>& y)
        {
            return x.second < y.second; 
        }
    );

    // If nmax > 0, determine a uniform mesh of timepoints and their
    // corresponding frames
    std::vector<std::string> filenames_sorted; 
    if (nmax > 0)
    {
        double tfinal = filenames[filenames.size() - 1].second; 
        Matrix<double, Dynamic, 1> times = Matrix<double, Dynamic, 1>::LinSpaced(
            nmax, tmin, tfinal
        );

        // Start with the initial filename
        filenames_sorted.push_back(filenames[0].first);

        // Then find the files corresponding to the intermediate timepoints  
        for (int i = 1; i < nmax - 1; ++i)
        {
            double time = times(i);
            auto it = filenames.begin(); 
            while (it->second < time)
                it++;
            auto prev = std::prev(it); 
            double delta1 = abs(prev->second - time); 
            double delta2 = abs(it->second - time); 
            if (delta1 < delta2) 
                filenames_sorted.push_back(prev->first); 
            else 
                filenames_sorted.push_back(it->first);  
        }

        // Then add the final filename 
        filenames_sorted.push_back(filenames[filenames.size() - 1].first); 
    }
    else    // Otherwise, just return the filenames in sorted order 
    {
        for (auto it = filenames.begin(); it != filenames.end(); ++it)
            filenames_sorted.push_back(it->first);
    }

    return filenames_sorted;  
}

/**
 * A safe version of acos() that accounts for (slightly) out-of-range input
 * values. 
 *
 * @param x Input value. 
 * @returns Arccosine of input value.  
 */
template <typename T>
T acosSafe(const T& x)
{
    if (x >= 1)
        return 0; 
    else if (x <= -1)
        return boost::math::constants::pi<T>(); 
    else 
        return acos(x);  
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
        return std::fmod(acosSafe<T>(f) + mu, boost::math::constants::two_pi<T>());
    else if (u3 < 0.5)
        return std::fmod(-acosSafe<T>(f) + mu, boost::math::constants::two_pi<T>()); 
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
 * Rotate the given orientation vector by the given angle counterclockwise
 * in the xy-plane.
 *
 * @param n Input orientation vector.
 * @param theta Input angle (about z-axis).  
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
 * Generate a uniform mesh of points on the unit circle.
 *
 * @param n Number of points to be included in the mesh. 
 * @param restrict_ypos If true, restrict the y-coordinate to be positive.
 * @returns Uniform mesh of points on the unit circle.
 */
template <typename T>
Matrix<T, Dynamic, 2> uniformMeshCircle(const int n, const bool restrict_ypos = false)
{
    // Create the mesh in terms of polar coordinates
    Matrix<T, Dynamic, 1> theta = Matrix<T, Dynamic, 1>::LinSpaced(
        n, 0,
        (
            restrict_ypos ? boost::math::constants::pi<T>() * (n - 1) / n :
            boost::math::constants::two_pi<T>() * (n - 1) / n
        )
    );
    Matrix<T, Dynamic, 2> mesh(n, 2); 
    mesh.col(0) = theta.array().cos().matrix();
    mesh.col(1) = theta.array().sin().matrix(); 

    return mesh; 
}

/**
 * Generate a uniform mesh of points on the unit sphere, obtained by iteratively
 * subdividing a regular icosahedral mesh. 
 *
 * @param n Minimum number of points to be included in the final mesh. 
 * @param restrict_zpos If true, restrict the z-coordinate to be positive.
 * @returns Nearly uniform mesh of points on the unit sphere. 
 */
template <typename T>
Matrix<T, Dynamic, 3> uniformMeshSphere(const int n, const bool restrict_zpos = false)
{
    // Create an initial icosahedral mesh
    //
    // These coordinates are taken from:
    // https://math.stackexchange.com/questions/2174594/
    // co-ordinates-of-the-vertices-an-icosahedron-relative-to-its-centroid
    std::vector<Point_3> vertices;
    double a = 1 / std::sqrt(5); 
    double b = (5 - std::sqrt(5)) / 10; 
    double c = (5 + std::sqrt(5)) / 10; 
    double d = -b;    // -5 + sqrt(5) / 10 
    double e = -c;    // -5 - sqrt(5) / 10
    vertices.push_back(Point_3(1, 0, 0)); 
    vertices.push_back(Point_3(a, 2 * a, 0)); 
    vertices.push_back(Point_3(a, b, sqrt(c))); 
    vertices.push_back(Point_3(a, e, sqrt(b))); 
    vertices.push_back(Point_3(a, e, -sqrt(b))); 
    vertices.push_back(Point_3(a, b, -sqrt(c))); 
    vertices.push_back(Point_3(-1, 0, 0)); 
    vertices.push_back(Point_3(-a, -2 * a, 0)); 
    vertices.push_back(Point_3(-a, d, -sqrt(c))); 
    vertices.push_back(Point_3(-a, c, -sqrt(b))); 
    vertices.push_back(Point_3(-a, c, sqrt(b))); 
    vertices.push_back(Point_3(-a, d, sqrt(c)));

    // Store the edges in an adjacency matrix 
    Matrix<int, Dynamic, Dynamic> edges = Matrix<int, Dynamic, Dynamic>(12, 12);
    for (int i = 1; i < 6; ++i)
    {
        edges(0, i) = 1;
        edges(i, 0) = 1; 
        edges(i, 2 + i % 5 - 1) = 1; 
        edges(i, 2 + (i + 3) % 5 - 1) = 1; 
        edges(i, 8 + (i + 1) % 5 - 1) = 1; 
        edges(i, 8 + (i + 2) % 5 - 1) = 1; 
    }
    for (int i = 6; i < 12; ++i)
    {
        edges(6, i) = 1; 
        edges(i, 6) = 1; 
        edges(i, 8 + (i - 1) % 5 - 1) = 1; 
        edges(i, 8 + (i + 2) % 5 - 1) = 1; 
        edges(i, 2 + i % 5 - 1) = 1; 
        edges(i, 2 + (i + 1) % 5 - 1) = 1; 
    }

    // Get the faces from the adjacency matrix
    std::vector<std::vector<int> > faces; 
    for (int i = 0; i < 12; ++i)
    {
        for (int j = i + 1; j < 12; ++j)
        {
            if (edges(i, j))
            {
                for (int k = j + 1; k < 12; ++k)
                {
                    if (edges(j, k) && edges(i, k))
                    {
                        // The face is (i, j, k)
                        //
                        // First find a normal vector to the face 
                        Vector_3 u(vertices[i], vertices[j]); 
                        Vector_3 v(vertices[i], vertices[k]); 
                        Vector_3 cross = CGAL::cross_product(u, v);

                        // Calculate the dot product of this normal vector
                        // with an inward-pointing vector (towards the origin)
                        Vector_3 center(
                            (vertices[i].x() + vertices[j].x() + vertices[k].x()) / 3,
                            (vertices[i].y() + vertices[j].y() + vertices[k].y()) / 3,
                            (vertices[i].z() + vertices[j].z() + vertices[k].z()) / 3
                        );  
                        double dot = CGAL::scalar_product(cross, center);
                        if (dot < 0)    // Normal points outward
                            faces.push_back({i, j, k});
                        else            // Normal points inward
                            faces.push_back({i, k, j}); 
                    }
                }
            }
        }
    }

    // Define the icosahedron 
    Polyhedron_3 poly;
    CGAL::Polyhedron_incremental_builder_3<HalfedgeDS> builder(poly.hds(), true);
    builder.begin_surface(vertices.size(), faces.size()); 
    for (const Point_3& v : vertices)
        builder.add_vertex(v); 
    for (int i = 0; i < faces.size(); ++i)
    {
        builder.begin_facet(); 
        builder.add_vertex_to_facet(faces[i][0]);
        builder.add_vertex_to_facet(faces[i][1]);
        builder.add_vertex_to_facet(faces[i][2]); 
        builder.end_facet(); 
    }
    builder.end_surface();

    // Subdivide the polyhedron using the Loop subdivision method, checking
    // after each iteration if the minimum number of vertices has been
    // reached 
    int n_vertices = 0; 
    if (restrict_zpos)
    {
        for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
        {
            if (it->point().z() >= 0)
                n_vertices++;  
        }
    }
    else
    {
        n_vertices = poly.size_of_vertices();
    } 
    while (n_vertices < n)
    {
        CGAL::Subdivision_method_3::Loop_subdivision(poly, 1);
        if (restrict_zpos)
        {
            n_vertices = 0; 
            for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
            {
                if (it->point().z() >= 0)
                    n_vertices++;  
            }
        }
        else
        {
            n_vertices = poly.size_of_vertices();
        } 
    }

    // Extract the vertices in the subdivided mesh and normalize each vertex
    // to be unit length
    Matrix<T, Dynamic, 3> mesh(n_vertices, 3); 
    int i = 0; 
    for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
    {
        Point_3 p = it->point();
        T x = static_cast<T>(p.x()); 
        T y = static_cast<T>(p.y()); 
        T z = static_cast<T>(p.z());
        if (!restrict_zpos || z >= 0)
        { 
            T norm = sqrt(x * x + y * y + z * z);
            mesh(i, 0) = x / norm; 
            mesh(i, 1) = y / norm; 
            mesh(i, 2) = z / norm; 
            i++;
        } 
    }

    return mesh; 
}

/**
 * Given a maximum (centerline) cell length and a range of acceptable cell-cell
 * distances, generate a uniform lattice of cell centers whose (approximate)
 * maximal cell-cell distances to a maximum-length cell at the origin are
 * within the given range.
 *
 * This function can generate this mesh in either 2 or 3 dimensions.  
 *
 * @param n Minimum number of cell centers to include in the lattice. 
 * @param dmin Minimum cell-cell distance. 
 * @param dmax Maximum cell-cell distance. 
 * @param lmax Maximum cell length. 
 * @returns Generated lattice of cell centers. 
 */
template <typename T, int Dim>
Matrix<T, Dynamic, Dim> uniformLattice(const int n, const T dmin, const T dmax,
                                       const T lmax)
{
    K kernel;
    const double half_lmax = static_cast<double>(lmax / 2);  

    // Infer the maximum coordinate per dimension
    const double rmax = static_cast<double>(dmax + lmax);

    // Iteratively create and refine the lattice until it contains at least
    // n points ...
    //
    // Start with n^{1/2} points or n^{1/3} points per dimension, depending
    // on the dimensionality
    int n_per_dim = static_cast<int>(ceil(pow(n, 1. / static_cast<double>(Dim))));
    int n_lattice = 0;
    Matrix<double, 3, 1> r1, n1, z;
    r1 << 0, 0, 0; 
    n1 << 1, 0, 0;
    z << 0, 0, 1;
    Segment_3 cell1 = generateSegment<double>(r1, n1, half_lmax);
    Matrix<T, Dynamic, Dim> lattice;  
    while (n_lattice < n)
    {
        // Generate a uniform lattice from 0 to rmax along each dimension 
        Matrix<double, Dynamic, 1> mesh_per_dim
            = Matrix<double, Dynamic, 1>::LinSpaced(n_per_dim, 0, rmax);
        lattice = Matrix<T, Dynamic, Dim>::Zero(pow(n_per_dim, Dim), Dim);
        int m = 0;
        if (Dim == 2)
        {
            Matrix<double, 2, 2> rot;    // Rotation by 90 degrees 
            rot << 0, -1, 
                   1,  0; 
            for (int i = 0; i < n_per_dim; ++i)
            {
                for (int j = 0; j < n_per_dim; ++j)
                {
                    Matrix<double, 3, 1> r2;
                    r2 << mesh_per_dim(i), mesh_per_dim(j), 0;

                    // Get the nearest point along the central, maximum-length
                    // cell to r2
                    double s = nearestCellBodyCoordToPoint<double>(
                        r1, n1, half_lmax, r2
                    ); 
                    Matrix<double, 3, 1> q = r1 + s * n1; 

                    // Get the distance between the central, maximum-length
                    // cell to the maximum-length cell centered at r2 with 
                    // some orientation orthogonal to the vector from q to r2
                    Matrix<double, 3, 1> n2 = Matrix<double, 3, 1>::Zero(); 
                    n2.head(2) = rot * (r2 - q).head(2);
                    n2 /= n2.norm();
                    Segment_3 cell2 = generateSegment<double>(r2, n2, half_lmax); 
                    auto result = distBetweenCells<double>(
                        cell1, cell2, 0, r1, n1, half_lmax, 1, r2, n2, half_lmax,
                        kernel
                    );
                    Matrix<double, 3, 1> d12 = std::get<0>(result);
                    
                    // If the distance is within the desired range, collect r2
                    double d = d12.norm(); 
                    if (d >= dmin && d <= dmax)
                    {
                        lattice(m, 0) = static_cast<T>(r2(0)); 
                        lattice(m, 1) = static_cast<T>(r2(1)); 
                        m++;
                    }
                }
            }
        }
        else    // Dim == 3
        {
            for (int i = 0; i < n_per_dim; ++i)
            {
                for (int j = 0; j < n_per_dim; ++j)
                {
                    for (int k = 0; k < n_per_dim; ++k)
                    {
                        Matrix<double, 3, 1> r2;
                        r2 << mesh_per_dim(i), mesh_per_dim(j), mesh_per_dim(k);

                        // Get the nearest point along the central, maximum-length
                        // cell to r2
                        double s = nearestCellBodyCoordToPoint<double>(
                            r1, n1, half_lmax, r2
                        ); 
                        Matrix<double, 3, 1> q = r1 + s * n1; 

                        // Get the distance between the central, maximum-length
                        // cell to the maximum-length cell centered at r2 with 
                        // some orientation orthogonal to the vector from q to r2
                        Matrix<double, 3, 1> n2 = (r2 - q).cross(z);
                        n2 /= n2.norm();
                        Segment_3 cell2 = generateSegment<double>(r2, n2, half_lmax); 
                        auto result = distBetweenCells<double>(
                            cell1, cell2, 0, r1, n1, half_lmax, 1, r2, n2,
                            half_lmax, kernel
                        );
                        Matrix<double, 3, 1> d12 = std::get<0>(result);
                        
                        // If the distance is within the desired range, collect r2
                        double d = d12.norm(); 
                        if (d >= dmin && d <= dmax)
                        {
                            lattice(m, 0) = static_cast<T>(r2(0)); 
                            lattice(m, 1) = static_cast<T>(r2(1)); 
                            lattice(m, 2) = static_cast<T>(r2(2)); 
                            m++;
                        }
                    }
                }
            }
        }
        n_lattice = m;
        n_per_dim++;  
    }

    return lattice(Eigen::seq(0, n_lattice - 1), Eigen::all); 
}

/**
 * Identify, via binary search, the index of the nearest value to the given
 * query, x, in the given array.
 *
 * The values are assumed to be distinct and sorted in ascending order.
 *
 * @param values Input array.
 * @param x Query value. 
 * @returns Index of nearest value to x.  
 */
template <typename T>
int nearestValue(const Ref<const Matrix<T, Dynamic, 1> >& values, const T x)
{
    // Quickly check if x is less than the first value or greater than the
    // last value
    if (x <= values(0))
        return 0; 
    else if (x >= values(values.size() - 1))
        return values.size() - 1;  

    // Otherwise, do binary search 
    int low = 0; 
    int high = values.size() - 1;
    int nearest_idx = 0;
    while (low <= high)
    {
        int mid = (low + high) / 2;

        // If x falls between values(mid) and values(mid + 1), set mid 
        // as the nearest index
        if (values(mid) <= x && x < values(mid + 1))
        {
            nearest_idx = mid; 
            break;
        }
        // If x is greater than values(mid + 1), then increase low 
        else if (x >= values(mid + 1))
        {
            low = mid + 1;
        }
        // If x is less than values(mid), then decrease high  
        else
        {
            high = mid - 1;
        }
    }

    // Note that this loop cannot have exited due to low > high
    if (low > high)
        throw std::runtime_error("Unexpected error during binary search");

    // Check which of the two endpoints of the interval is nearest
    if (x - values(nearest_idx) < values(nearest_idx + 1) - x)
        return nearest_idx; 
    else 
        return nearest_idx + 1;  
}

/**
 * Get the binomial coefficient, n choose k.
 *
 * @param n Total number of items.
 * @param k Number of items to choose. 
 * @returns n choose k. 
 */
int binom(const int n, const int k)
{
    if (n < k)
        return 0; 
    else if (k == 0 || n == k)
        return 1; 
    else 
        return binom(n - 1, k - 1) + binom(n - 1, k); 
}

/**
 * A simple quasi-recursive function for getting the power set of an ordered
 * set (i.e., a vector).
 *
 * @param vec Input ordered set. 
 * @param nonempty If true, skip the empty set. 
 * @returns Power set of input set. 
 */
std::vector<std::vector<int> > getPowerset(const std::vector<int>& vec, 
                                           const bool nonempty = true)
{
    std::vector<std::vector<int> > powerset;
    const int n = vec.size();  

    // Maintain a stack of sub-vectors 
    std::stack<std::pair<int, std::vector<int> > > stack;
    stack.push(std::make_pair(0, std::vector<int>({})));
    while (!stack.empty())
    {
        auto next = stack.top();
        stack.pop(); 
        int start = next.first; 
        std::vector<int> subset = next.second;
        if (!nonempty || subset.size() > 0)
            powerset.push_back(subset); 
        for (int i = start; i < n; ++i)
        {
            std::vector<int> new_subset(subset);
            new_subset.push_back(vec[i]);  
            stack.push(std::make_pair(i + 1, new_subset)); 
        } 
    }

    return powerset; 
}

/**
 * A simple quasi-recursive function for getting all k-combinations of an
 * ordered set (i.e., a vector). 
 *
 * @param vec Input ordered set. 
 * @param k Number of items to choose per combination. 
 * @returns All k-combinations of input set. 
 */
std::vector<std::vector<int> > getCombinations(const std::vector<int>& vec, 
                                               const int k)
{
    std::vector<std::vector<int> > combinations;
    const int n = vec.size();

    // Maintain a stack of sub-vectors 
    std::stack<std::pair<int, std::vector<int> > > stack;
    stack.push(std::make_pair(0, std::vector<int>({})));  
    while (!stack.empty())
    {
        auto next = stack.top();
        stack.pop(); 
        int start = next.first; 
        std::vector<int> path = next.second; 
        if (path.size() == k)
        {
            combinations.push_back(path); 
            continue; 
        }
        for (int i = start; i < n; ++i)
        {
            std::vector<int> newpath(path); 
            newpath.push_back(vec[i]); 
            stack.push(std::make_pair(i + 1, newpath)); 
        } 
    }

    return combinations;  
}

/**
 * Get a row echelon form for the given matrix with exact numerical types
 * via Gaussian elimination. 
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> rowEchelonForm(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                           const bool rref = false)
{
    // Initialize pivot row index 
    const int nrows = A.rows(); 
    const int ncols = A.cols();
    Matrix<T, Dynamic, Dynamic> A_reduced(A);  
    int pivot_row = 0;
    int pivot_col = 0;

    while (pivot_row < nrows && pivot_col < ncols)
    {
        // Find the pivot, which is the entry A(i, pivot_col), for
        // i >= pivot_row, with the largest absolute value
        int max_i = pivot_row;
        for (int i = pivot_row + 1; i < nrows; ++i)
        {
            if (abs(A_reduced(i, pivot_col)) > abs(A_reduced(max_i, pivot_col)))
                max_i = i;
        }

        // If the maximal entry is zero, then move onto the next column
        if (A_reduced(max_i, pivot_col) == 0)
        {
            pivot_col++; 
            continue; 
        }

        // Otherwise, swap the chosen row with the pivot row 
        Matrix<T, 1, Dynamic> row(A_reduced.row(max_i)); 
        A_reduced.row(max_i) = A_reduced.row(pivot_row); 
        A_reduced.row(pivot_row) = row;

        // For all rows below the (new) pivot row, subtract the appropriate
        // multiple of the pivot row 
        for (int i = pivot_row + 1; i < nrows; ++i)
        {
            T mult = A_reduced(i, pivot_col) / A_reduced(pivot_row, pivot_col);
            A_reduced(i, pivot_col) = 0;
            for (int j = pivot_col + 1; j < ncols; ++j)
                A_reduced(i, j) -= mult * A_reduced(pivot_row, j);  
        }

        // Move onto the next row and column 
        pivot_row++;
        pivot_col++;  
    }

    // Get the reduced row echelon form, if desired 
    if (rref)
    {
        // Get the pivots in each row 
        Matrix<int, Dynamic, 1> pivots = -Matrix<int, Dynamic, 1>::Ones(nrows);
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                // Identify the first nonzero entry in each row
                if (A_reduced(i, j) != 0)
                {
                    pivots(i) = j;
                    break; 
                }
            }
        }

        // For each row, starting from the bottom ... 
        for (int i = nrows - 1; i >= 0; --i)
        {
            // If that row contains a pivot ...
            if (pivots(i) != -1)
            {
                // For each row above the i-th row, subtract a multiple 
                // of the i-th row
                for (int j = i - 1; j >= 0; --j)
                {
                    T mult = A_reduced(j, pivots(i)) / A_reduced(i, pivots(i)); 
                    A_reduced.row(j) -= mult * A_reduced.row(i); 
                }

                // Divide the i-th row by the pivot entry 
                A_reduced.row(i) /= A_reduced(i, pivots(i)); 
            }
        }
    }

    return A_reduced; 
}

/**
 * Given a matrix in row echelon form, return the pivots in each row.
 *
 * This function returns an array of coordinates, pivots(i) = j, where j is
 * the column containing the pivot in the i-th row. 
 *
 * If the i-th row does not have a nonzero entry (and therefore has no pivot),
 * then pivots(i) == -1. 
 */
template <typename T>
Matrix<int, Dynamic, 1> getPivots(const Ref<const Matrix<T, Dynamic, Dynamic> >& A_reduced)
{
    const int nrows = A_reduced.rows();
    const int ncols = A_reduced.cols();  
    Matrix<int, Dynamic, 1> pivots = -Matrix<int, Dynamic, 1>::Ones(nrows);
    for (int i = 0; i < nrows; ++i)
    {
        // Identify the first nonzero entry in each row
        for (int j = 0; j < ncols; ++j)
        {
            if (A_reduced(i, j) != 0)
            {
                pivots(i) = j;
                break; 
            }
        }
    }

    return pivots; 
}

/**
 * Given a matrix in row echelon form, return the pivot columns in each row.
 *
 * This function returns an array of coordinates, pivots(i) = j, where j is
 * the row containing the pivot in the i-th column. 
 *
 * If the i-th column does not have a nonzero entry (and therefore has no
 * pivot), then pivots(i) == -1. 
 */
template <typename T>
Matrix<int, Dynamic, 1> getPivotCols(const Ref<const Matrix<T, Dynamic, Dynamic> >& A_reduced)
{
    const int nrows = A_reduced.rows();
    const int ncols = A_reduced.cols();  
    Matrix<int, Dynamic, 1> pivots = -Matrix<int, Dynamic, 1>::Ones(ncols);
    for (int i = 0; i < ncols; ++i)
    {
        // Identify the final nonzero entry in each column 
        for (int j = nrows - 1; j >= 0; --j)
        {
            if (A_reduced(j, i) != 0)
            {
                // If the first nonzero entry is in a *lower* row than any 
                // preceding pivot entry, then this counts as a pivot
                if (i == 0 || (pivots.head(i).array() < j).all())
                    pivots(i) = j; 
                break; 
            }
        }
    }

    return pivots; 
}

/**
 * Get a basis for the image (column space) of a matrix with exact numerical
 * types via Gaussian elimination.
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 *
 * Each *column* in the returned matrix is a basis vector for the kernel. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> columnSpace(const Ref<const Matrix<T, Dynamic, Dynamic> >& A)
{
    // Row-reduce the given matrix 
    Matrix<T, Dynamic, Dynamic> A_reduced = rowEchelonForm<T>(A); 

    // Now that the matrix is in row-reduced form, find the pivots ... 
    Matrix<int, Dynamic, 1> pivots = getPivots<T>(A_reduced);

    // ... and the basic variables 
    std::vector<int> basic_vars; 
    basic_vars.push_back(pivots(0));
    const int nrows = A.rows();  
    for (int i = 1; i < nrows; ++i)
    {
        if (!(pivots(i) == -1 || pivots(i) == pivots(i - 1)))
            basic_vars.push_back(pivots(i));
    }

    // Return the columns corresponding to the basic variables 
    Matrix<T, Dynamic, Dynamic> colspace(nrows, basic_vars.size());
    for (int i = 0; i < basic_vars.size(); ++i)
        colspace.col(i) = A.col(basic_vars[i]); 

    return colspace; 
}

/**
 * Get a basis for the kernel of a matrix with exact numerical types via
 * Gaussian elimination.
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 *
 * Each *column* in the returned matrix is a basis vector for the kernel. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> kernel(const Ref<const Matrix<T, Dynamic, Dynamic> >& A)
{
    // Row-reduce the given matrix 
    Matrix<T, Dynamic, Dynamic> A_reduced = rowEchelonForm<T>(A); 

    // Now that the matrix is in row-reduced form, find the pivots ...
    Matrix<int, Dynamic, 1> pivots = getPivots<T>(A_reduced); 

    // ... and the basic and free variables
    //
    // Store, for each row i, the tuple (pivots(i), i), depending on whether
    // pivots(i) represents a basic or free variable
    const int nrows = A.rows(); 
    const int ncols = A.cols(); 
    Matrix<int, Dynamic, 1> basic_vars = Matrix<int, Dynamic, 1>::Zero(ncols);
    std::unordered_map<int, int> basic_constraints; 
    basic_vars(pivots(0)) = 1;  
    basic_constraints[pivots(0)] = 0;
    for (int i = 1; i < nrows; ++i)
    {
        if (!(pivots(i) == -1 || pivots(i) == pivots(i - 1)))
        {
            basic_vars(pivots(i)) = 1;
            basic_constraints[pivots(i)] = i;
        } 
    }
    Matrix<int, Dynamic, 1> free_vars = Matrix<int, Dynamic, 1>::Ones(ncols) - basic_vars;

    // Maintain index lookups for the free and basic variables
    //
    // That is, if x_k is a free variable, then free_idx[k] = index of k
    // among the free variables  
    std::unordered_map<int, int> free_idx, basic_idx;
    int basic_i = 0; 
    int free_i = 0; 
    for (int i = 0; i < ncols; ++i)
    {
        if (basic_vars(i))
        {
            basic_idx[i] = basic_i;
            basic_i++; 
        }
        else 
        {
            free_idx[i] = free_i; 
            free_i++; 
        }
    } 

    // Now run through the basic variables in reverse and express them in 
    // terms of the free variables
    const int nbasic = basic_vars.sum();
    const int nfree = free_vars.sum();
    Matrix<T, Dynamic, Dynamic> basic_in_terms_of_free(nbasic, nfree);
    for (int i = ncols - 1; i >= 0; --i)
    {
        if (basic_vars(i)) 
        {
            int j = basic_idx[i];           // Index of basic variable  
            int k = basic_constraints[i];   // Index of determining equation 
            
            // For each subsequent variable in the k-th equation ... 
            for (int m = j + 1; m < ncols; ++m)
            {
                // Is it a basic variable or a free variable? 
                if (free_idx.find(m) != free_idx.end())
                {
                    // If free, then set the coefficient accordingly
                    //
                    // xi = j-th basic variable
                    // xm = free variable
                    // k = constraint in which xi is the pivot  
                    basic_in_terms_of_free(j, free_idx[m])
                        -= A_reduced(k, m) / A_reduced(k, i); 
                }
                else 
                {
                    // Otherwise, the m-th variable must be expressed as 
                    // a linear combination of free variables that must 
                    // have already been processed
                    //
                    // xi = j-th basic variable
                    // xm = current basic variable 
                    // k = constraint in which xi is the pivot  
                    int m_idx = basic_idx[m];
                    basic_in_terms_of_free.row(j)
                        -= A_reduced(k, m) * basic_in_terms_of_free.row(m_idx) / A_reduced(k, i);  
                }
            }
        }
    }

    // Now generate the kernel basis vectors ... 
    Matrix<T, Dynamic, Dynamic> kernel = Matrix<T, Dynamic, Dynamic>::Zero(nfree, ncols);
    for (int i = 0; i < ncols; ++i)
    {
        if (free_vars(i))
        {
            kernel(free_idx[i], i) = 1;
            for (int j = 0; j < ncols; ++j)
            {
                if (basic_vars(j))
                    kernel(free_idx[i], j) = basic_in_terms_of_free(basic_idx[j], free_idx[i]); 
            }
        }
    }

    return kernel.transpose(); 
}

/**
 * Solve the linear system, A * x = b, given that A is full rank, for 
 * each column b in the matrix B, via Gaussian elimination.
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 *
 * Each *column* is a solution to the linear system A * x = b, where b is
 * a *column* of B. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> solve(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                  const Ref<const Matrix<T, Dynamic, Dynamic> >& B,
                                  const bool check_all_basic = false,
                                  const T free_var_value = 0)
{
    // Row-reduce the given linear system
    const int nrows = A.rows(); 
    const int ncols = A.cols();
    const int nvecs = B.cols(); 
    Matrix<T, Dynamic, Dynamic> system(nrows, ncols + nvecs);
    system(Eigen::all, Eigen::seq(0, ncols - 1)) = A;
    system(Eigen::all, Eigen::lastN(nvecs)) = B; 
    Matrix<T, Dynamic, Dynamic> reduced = rowEchelonForm<T>(system);
    Matrix<T, Dynamic, Dynamic> A_reduced = reduced(Eigen::all, Eigen::seq(0, ncols - 1)); 
    Matrix<T, Dynamic, Dynamic> B_reduced = reduced(Eigen::all, Eigen::lastN(nvecs)); 

    // Now that the matrix is in row-reduced form, find the pivots ...
    Matrix<int, Dynamic, 1> pivots = getPivots<T>(A_reduced); 

    // Get the basic variables 
    //
    // Store, for each row i, the tuple (pivots(i), i), depending on whether
    // pivots(i) represents a basic or free variable
    Matrix<int, Dynamic, 1> basic_vars = Matrix<int, Dynamic, 1>::Zero(ncols);
    std::unordered_map<int, int> basic_constraints; 
    basic_vars(pivots(0)) = 1;  
    basic_constraints[pivots(0)] = 0;
    for (int i = 1; i < nrows; ++i)
    {
        if (!(pivots(i) == -1 || pivots(i) == pivots(i - 1)))
        {
            basic_vars(pivots(i)) = 1;
            basic_constraints[pivots(i)] = i;
        } 
    }

    // Check that every variable is basic, if desired  
    if (check_all_basic && basic_vars.sum() < ncols)
        throw std::runtime_error(
            "Found non-basic variables, meaning that input matrix is not of "
            "full rank"
        );

    Matrix<T, Dynamic, Dynamic> x(ncols, nvecs);

    // First, if there are any free variables, set their values arbitrarily
    if (basic_vars.sum() < ncols)
    {
        for (int idx = 0; idx < nvecs; ++idx)
        {
            // For each right-hand vector, check whether there is an inconsistency  
            for (int i = 0; i < nrows; ++i)
            {
                if ((A_reduced.row(i).array() == 0).all() && B_reduced(i, idx) != 0)
                    throw std::runtime_error(
                        "Found inconsistency in non-full-rank linear system" 
                    ); 
            }

            // If not, set the free variables arbitrarily
            for (int j = 0; j < ncols; ++j)
            {
                if (basic_vars(j) == 0)
                    x(j, idx) = free_var_value; 
            } 
        }
    }

    // Now back-substitute to get the entries of the solution vector, x,
    // for each column in B
    for (int idx = 0; idx < nvecs; ++idx)
    { 
        for (int j = ncols - 1; j >= 0; --j)
        {
            if (basic_vars(j) == 1)
            {
                // Get the corresponding constraint
                int i = basic_constraints[j];
                
                // Solve for the value of the j-th variable
                x(j, idx) = B_reduced(i, idx); 
                for (int k = j + 1; k < ncols; ++k)
                    x(j, idx) -= A_reduced(i, k) * x(k, idx);
                x(j, idx) /= A_reduced(i, j);
            } 
        }
    }
    
    return x;
}

/**
 * Get a basis for the complement of the span of the given basis, with respect
 * to the ambient vector space. 
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 *
 * Each *column* of the given matrix is assumed to be a basis vector, and 
 * each *column* of the output matrix is a basis vector for the complement.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> complement(const Ref<const Matrix<T, Dynamic, Dynamic> >& basis)
{
    const int dim = basis.rows();      // Note that dim is the ambient dimension
    const int nvecs = basis.cols();    // In contrast, nvecs is the dimension of the basis
    
    // Construct linear system, one for each standard basis vector
    Matrix<T, Dynamic, Dynamic> system(dim, nvecs + dim); 
    system(Eigen::all, Eigen::seq(0, nvecs - 1)) = basis; 
    system(Eigen::all, Eigen::lastN(dim)) = Matrix<T, Dynamic, Dynamic>::Identity(dim, dim); 

    // Get row echelon form 
    Matrix<T, Dynamic, Dynamic> system_reduced = rowEchelonForm<T>(system);

    // Get the pivot columns 
    Matrix<int, Dynamic, 1> pivots = getPivotCols<T>(system_reduced); 

    // Return the columns corresponding to the pivots among the last dim
    // columns
    std::vector<int> pivot_cols; 
    for (int i = nvecs; i < nvecs + dim; ++i)
    {
        if (pivots(i) != -1)
            pivot_cols.push_back(i); 
    }
    Matrix<T, Dynamic, Dynamic> complement
        = Matrix<T, Dynamic, Dynamic>::Zero(dim, pivot_cols.size());
    for (int i = 0; i < pivot_cols.size(); ++i)
        complement.col(i) = system_reduced.col(pivot_cols[i]); 

    return complement;  
}

/**
 * Get a basis for the quotient space, ker(A) / im(B), given that im(B) is 
 * a subspace of ker(A), using Gaussian elimination. 
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> quotientSpace(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                          const Ref<const Matrix<T, Dynamic, Dynamic> >& B)
{
    // Get bases for the kernel of A and the image of B
    Matrix<T, Dynamic, Dynamic> kerA = ::kernel<T>(A);
    Matrix<T, Dynamic, Dynamic> imB = ::columnSpace<T>(B);
    
    // Form the augmented matrix 
    Matrix<T, Dynamic, Dynamic> system(kerA.rows(), kerA.cols() + imB.cols());
    system(Eigen::all, Eigen::seq(0, imB.cols() - 1)) = imB; 
    system(Eigen::all, Eigen::lastN(kerA.cols())) = kerA;

    // Get the row echelon form and pivot columns  
    Matrix<T, Dynamic, Dynamic> system_reduced = ::rowEchelonForm<T>(system);
    Matrix<int, Dynamic, 1> pivots = ::getPivotCols<T>(system_reduced);

    // Return the vectors that do not correspond to the pivot columns
    int n_quotient_basis = 0;
    for (int i = imB.cols(); i < imB.cols() + kerA.cols(); ++i)
    {
        if (pivots(i) != -1)
            n_quotient_basis++; 
    }
    Matrix<T, Dynamic, Dynamic> quotient_basis(kerA.rows(), n_quotient_basis);
    int j = 0;  
    for (int i = imB.cols(); i < imB.cols() + kerA.cols(); ++i)
    {
        if (pivots(i) != -1)
        {
            quotient_basis.col(j) = kerA.col(i - imB.cols());
            j++; 
        }
    }

    return quotient_basis; 
}

/**
 * Get a basis for the quotient space, T^d / im(B), where T is the underlying
 * field and d is the dimension (number of rows in B), using Gaussian
 * elimination. 
 *
 * We assume that the underlying field is the rationals or a field of finite
 * characteristic.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> quotientSpace(const Ref<const Matrix<T, Dynamic, Dynamic> >& B)
{
    // Get a basis for the image of B
    Matrix<T, Dynamic, Dynamic> imB = ::columnSpace<T>(B);
    
    // Get the complement of im(B) with respect to the standard basis 
    Matrix<T, Dynamic, Dynamic> imB_complement = ::complement<T>(imB);

    return imB_complement;
}

/**
 * Convert a linear program that minimizes the 1-norm of a vector, z, 
 * subject to A * x = b, where x includes the variables in z, into
 * standard form.
 *
 * We assume that the first nvars_obj variables comprise the vector, z,
 * whose 1-norm we are seeking to minimize in the objective.
 */
template <typename T>
CGAL::Quadratic_program<T> defineL1LinearProgram(const int nvars_obj,
                                                 const Ref<const Matrix<double, Dynamic, Dynamic> >& A, 
                                                 const Ref<const Matrix<double, Dynamic, 1> >& b,
                                                 const int inequality_mode = -1)
{
    const int nconstraints = A.rows(); 
    const int nvars_total = A.cols();

    // The standardization procedure is as follows:
    //
    // - For each variable zk in the objective, we first define an auxiliary
    //   variable uk >= 0 such that -uk <= zk <= uk
    //
    // - Each such pair of inequalities can be rewritten as:
    //   1) zk - uk <= 0    [right]
    //   2) -zk - uk <= 0   [left]
    //
    // - If equality constraints are desired:
    //   - To convert each pair of inequalities into equalities, we define 
    //     slack variables sk1 and sk2 such that 
    //     1) zk - uk + sk1 = 0
    //     2) -zk - uk + sk2 = 0
    // - If less-than constraints are desired:
    //   - Convert A * x = b into inequalities A * x <= b and -A * x <= -b.
    //
    // So, in all, if d = nvars_obj is the number of variables in the objective,
    // we seek to minimize 
    //
    // u1 + ... + ud
    //
    // such that, if equality constraints are desired: 
    //
    // zk - uk + sk1 = 0 for all k = 1, ..., d
    // -zk - uk + sk2 = 0 for all k = 1, ..., d
    // A * x = b
    //
    // or, if less-than inequality constraints are desired:
    //
    // zk - uk <= 0 for all k = 1, ..., d
    // -zk - uk <= 0 for all k = 1, ..., d
    // A * x <= b
    // -A * x <= -b
    //
    // or, if greater-than inequality constraints are desired:
    //
    // -zk + uk >= 0 for all k = 1, ..., d
    // zk + uk >= 0 for all k = 1, ..., d
    // A * x >= b
    // -A * x >= -b
    //
    // uk, sk1, sk2 are all non-negative, but the input variables x, which
    // include the zk, may not be
    CGAL::Comparison_result inequality; 
    if (inequality_mode == -1)
        inequality = CGAL::SMALLER; 
    else if (inequality_mode == 0)
        inequality = CGAL::EQUAL; 
    else 
        inequality = CGAL::LARGER;
    CGAL::Quadratic_program<T> lp(
        inequality,
        false,        // No lower bounds by default 
        0,            // Meaningless
        false,        // No upper bounds by default
        0             // Meaningless
    );

    // If equality constraints are desired ... 
    if (inequality_mode == 0)
    {
        // First incorporate the linear constraints A * x = b
        for (int i = 0; i < nconstraints; ++i)
        {
            for (int j = 0; j < nvars_total; ++j)
            {
                lp.set_a(j, i, static_cast<T>(A(i, j))); 
            }
            lp.set_b(i, static_cast<T>(b(i))); 
        }

        // Then incorporate the additional constraints  
        for (int k = 0; k < nvars_obj; ++k)
        {
            int zk_idx = k; 
            int uk_idx = nvars_total + k; 
            int sk1_idx = nvars_total + nvars_obj + k; 
            int sk2_idx = nvars_total + 2 * nvars_obj + k;
            int con1_idx = nconstraints + k; 
            int con2_idx = nconstraints + nvars_obj + k;
            lp.set_a(zk_idx, con1_idx, 1); 
            lp.set_a(uk_idx, con1_idx, -1); 
            lp.set_a(sk1_idx, con1_idx, 1);
            lp.set_b(con1_idx, 0);
            lp.set_a(zk_idx, con2_idx, -1); 
            lp.set_a(uk_idx, con2_idx, -1); 
            lp.set_a(sk2_idx, con2_idx, 1);
            lp.set_b(con2_idx, 0);
            lp.set_l(uk_idx, true, 0);
            lp.set_l(sk1_idx, true, 0); 
            lp.set_l(sk2_idx, true, 0);
        }
    }
    // If less-than constraints are desired ... 
    else if (inequality_mode == -1)
    {
        // First incorporate the linear constraints A * x <= b
        for (int i = 0; i < nconstraints; ++i)
        {
            for (int j = 0; j < nvars_total; ++j)
            {
                lp.set_a(j, i, static_cast<T>(A(i, j))); 
            }
            lp.set_b(i, static_cast<T>(b(i))); 
        }
        // ... as well as the constraints -A * x <= -b
        for (int i = 0; i < nconstraints; ++i)
        {
            for (int j = 0; j < nvars_total; ++j)
            {
                lp.set_a(j, nconstraints + i, -static_cast<T>(A(i, j))); 
            }
            lp.set_b(nconstraints + i, -static_cast<T>(b(i))); 
        }

        // Then incorporate the additional constraints  
        for (int k = 0; k < nvars_obj; ++k)
        {
            int zk_idx = k; 
            int uk_idx = nvars_total + k;
            int con1_idx = 2 * nconstraints + k; 
            int con2_idx = 2 * nconstraints + nvars_obj + k; 
            lp.set_a(zk_idx, con1_idx, 1); 
            lp.set_a(uk_idx, con1_idx, -1); 
            lp.set_b(con1_idx, 0);
            lp.set_a(zk_idx, con2_idx, -1); 
            lp.set_a(uk_idx, con2_idx, -1); 
            lp.set_b(con2_idx, 0);
            lp.set_l(uk_idx, true, 0);
        }
    }
    // If greater-than constraints are desired ... 
    else    // inequality_mode == 1
    {
        // First incorporate the linear constraints A * x >= b
        for (int i = 0; i < nconstraints; ++i)
        {
            for (int j = 0; j < nvars_total; ++j)
            {
                lp.set_a(j, i, static_cast<T>(A(i, j))); 
            }
            lp.set_b(i, static_cast<T>(b(i))); 
        }
        // ... as well as the constraints -A * x >= -b
        for (int i = 0; i < nconstraints; ++i)
        {
            for (int j = 0; j < nvars_total; ++j)
            {
                lp.set_a(j, nconstraints + i, -static_cast<T>(A(i, j))); 
            }
            lp.set_b(nconstraints + i, -static_cast<T>(b(i))); 
        }

        // Then incorporate the additional constraints  
        for (int k = 0; k < nvars_obj; ++k)
        {
            int zk_idx = k; 
            int uk_idx = nvars_total + k;
            int con1_idx = 2 * nconstraints + k; 
            int con2_idx = 2 * nconstraints + nvars_obj + k; 
            lp.set_a(zk_idx, con1_idx, -1); 
            lp.set_a(uk_idx, con1_idx, 1); 
            lp.set_b(con1_idx, 0);
            lp.set_a(zk_idx, con2_idx, 1); 
            lp.set_a(uk_idx, con2_idx, 1); 
            lp.set_b(con2_idx, 0);
            lp.set_l(uk_idx, true, 0);
        }
    }

    // Finally define the objective function
    for (int k = 0; k < nvars_obj; ++k)
    {
        int uk_idx = nvars_total + k; 
        lp.set_c(uk_idx, 1);
    } 

    return lp;  
} 

#endif
