/**
 * Various utility functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/12/2025
 */

#ifndef BIOFILM_UTILS_3D_HPP
#define BIOFILM_UTILS_3D_HPP

#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <boost/json/src.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Subdivision_method_3.h>
#include <CGAL/Kd_tree.h>
#include "indices.hpp"

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
 * Generate a uniform mesh of points on the unit sphere, obtained by iteratively
 * subdividing a regular icosahedral mesh. 
 *
 * @param n Minimum number of points to be included in the final mesh. 
 * @returns Nearly uniform mesh of points on the unit sphere. 
 */
template <typename T>
Matrix<T, Dynamic, 3> uniformMeshSphere(const int n)
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
                        std::cout << i << " " << j << " " << k << std::endl;  
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

    // Subdivide the polyhedron using the Loop subdivision method
    int n_vertices = poly.size_of_vertices(); 
    while (n_vertices < n)
    {
        CGAL::Subdivision_method_3::Loop_subdivision(poly, 1);
        n_vertices = poly.size_of_vertices(); 
    }

    // Extract the vertices in the subdivided mesh and normalize each 
    Matrix<T, Dynamic, 3> mesh(poly.size_of_vertices(), 3);
    int i = 0; 
    for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
    {
        Point_3 p = it->point();
        T x = static_cast<T>(p.x()); 
        T y = static_cast<T>(p.y()); 
        T z = static_cast<T>(p.z());  
        T norm = sqrt(x * x + y * y + z * z);
        mesh(i, 0) = x / norm; 
        mesh(i, 1) = y / norm; 
        mesh(i, 2) = z / norm; 
        i++; 
    }

    return mesh; 
}

/**
 * Given a maximum (centerline) cell length and a range of acceptable cell-cell
 * distances, generate a uniform lattice of cell centers whose (approximate)
 * maximal cell-cell distances to a maximum-length cell at the origin are
 * within the given range. 
 *
 * @param n Minimum number of cell centers to include in the lattice. 
 * @param dmin Minimum cell-cell distance. 
 * @param dmax Maximum cell-cell distance. 
 * @param lmax Maximum cell length. 
 * @returns Generated lattice of cell centers. 
 */
template <typename T>
Matrix<T, Dynamic, 3> uniformLattice(const int n, const T dmin, const T dmax,
                                     const T lmax)
{
    K kernel; 

    // Infer the maximum coordinate per dimension
    const T rmax = dmax + lmax;  

    // Iteratively create and refine the lattice until it contains at least
    // n points ...
    //
    // Start with n^{1/3} points per dimension
    int n_per_dim = static_cast<int>(ceil(pow(n, 1. / 3.));
    int n_lattice = 0;
    Matrix<T, 3, 1> r1, n1, z;
    r1 << 0, 0, 0; 
    n1 << 1, 0 ,0;
    z << 0, 0, 1;
    Segment_3 cell1 = generateSegment<T>(r1, n1, lmax / 2.0); 
    while (n_lattice < n)
    {
        // Generate a uniform lattice from 0 to rmax along each dimension 
        Matrix<T, Dynamic, 1> mesh_per_dim = Matrix<T, Dynamic, 1>::LinSpaced(n_per_dim, 0, rmax);
        Matrix<T, Dynamic, 3> lattice = Matrix<T, Dynamic, 3>::Zero(pow(n_per_dim, 3), 3);
        int m = 0; 
        for (int i = 0; i < n_per_dim; ++i)
        {
            for (int j = 0; j < n_per_dim; ++j)
            {
                for (int k = 0; k < n_per_dim; ++k)
                {
                    Matrix<T, 3, 1> r2;
                    r2 << mesh_per_dim(i), mesh_per_dim(j), mesh_per_dim(k);

                    // Get the nearest point along the central, maximum-length
                    // cell to r2
                    T s = nearestCellBodyCoordToPoint<T>(r1, n1, lmax / 2.0, r2); 
                    Matrix<T, 3, 1> q = r1 + s * n1; 

                    // Get the distance between the central, maximum-length
                    // cell to the maximum-length cell centered at r2 with 
                    // some orientation orthogonal to the vector from q to r2
                    Matrix<T, 3, 1> n2 = (r2 - q).cross(z);
                    n2 /= n2.norm();
                    Segment_3 cell2 = generateSegment<T>(r2, n2, lmax / 2.0); 
                    auto result = distBetweenCells<T>(
                        cell1, cell2, 0, r1, n1, lmax / 2.0, 1, r2, n2,
                        lmax / 2.0, kernel
                    );
                    Matrix<T, 3, 1> d12 = std::get<0>(result);
                    
                    // If the distance is within the desired range, collect r2
                    T d = d12.norm(); 
                    if (d >= dmin && d <= dmax)
                    {
                        lattice.row(m) = r2;
                        m++;
                    }
                }
            }
        }
        n_lattice = m;
        n_per_dim++;  
    }

    return lattice.conservativeResize(n_lattice, 3);  
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
    int low = 0; 
    int high = values.size() - 1;
    T nearest = std::numeric_limits<T>::infinity();
    int nearest_idx;  
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (values[mid] == x)
            return mid;
        else if (values[mid] < x)
            low = mid + 1; 
        else    // values[mid] > x
            high = mid - 1;
        
        if (abs(values[mid] - x) < nearest)
        {
            nearest = values[mid];
            nearest_idx = mid; 
        }
    }

    return nearest_idx; 
}

#endif
