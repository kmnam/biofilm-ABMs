/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/29/2025
 */

#include <iostream>
#include <string>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "../include/boundaries.hpp"
#include "../include/indices.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

using std::sqrt; 
using std::sin; 
using std::cos; 

/**
 * Project the given cells in 3-D into the r-z plane.
 *
 * The origin is set to the center of mass of the cells. 
 *
 * The output array has 5 columns, which give the r- and z-coordinates, as 
 * well as the projected orientation vector. 
 *
 * @param cells Input population of cells. 
 * @returns Projections of cells into the r-z plane. 
 */
template <typename T>
Array<T, Dynamic, 5> projectRZ(const Ref<const Array<T, Dynamic, Dynamic> >& cells)
{
    // Get the radial direction for each cell from the origin (which we 
    // take as (0, 0, 0) for consistency)
    Array<T, Dynamic, 3> rdists = cells(Eigen::all, __colseq_r);
    Array<T, Dynamic, 1> rnorms = rdists.matrix().rowwise().norm().array();
    Array<T, Dynamic, 3> rdirs = rdists.colwise() / rnorms;

    // For each cell, prepare the projected coordinates 
    Array<T, Dynamic, 5> coords(cells.rows(), 5); 
    for (int i = 0; i < cells.rows(); ++i)
    {
        coords(i, 0) = sqrt(
            cells(i, __colidx_rx) * cells(i, __colidx_rx) +
            cells(i, __colidx_ry) * cells(i, __colidx_ry)
        );
        coords(i, 1) = cells(i, __colidx_rz);
        Matrix<T, 3, 1> v = rdirs.row(i).matrix();
        Matrix<T, 3, 1> n = cells(i, __colseq_n).matrix(); 
        Matrix<T, 3, 1> projection = n.dot(v) * v;
        coords(i, 2) = projection(0); 
        coords(i, 3) = projection(1); 
        coords(i, 4) = projection(2); 
    } 

    return coords; 
}

/**
 * Get the peripheral subset of the given cells from a simply connected
 * alpha-shape built from 2-D cross-sectional outlines of the cells that have
 * been projected onto the r-z plane.
 *
 * It is assumed that there are 3 or more cells.  
 *
 * @param cells            Input population of cells. 
 * @param R                Cell radius. 
 * @param outline_meshsize Approximate meshsize with which to obtain points
 *                         from each cell outline.
 * @returns An object containing the alpha-shape built from the cell outlines,
 *          together with a vector of indices that assigns each outline point
 *          to the cell from which it originates. 
 */
template <typename T>
std::pair<AlphaShape2DProperties, std::vector<int> >
    getBoundaryFromOutlines(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                            const T R, const T outline_meshsize) 
{
    std::vector<double> x, y;
    std::vector<int> idx;

    // Project the cells into the r-z plane 
    Array<T, Dynamic, 5> rz_coords = projectRZ<T>(cells); 

    // For each cell ... 
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Get the projected coordinates 
        Array<T, 3, 1> ri, ni;
        ri << rz_coords(i, 0), 0.0, rz_coords(i, 1); 
        ni << rz_coords(i, 2), rz_coords(i, 3), rz_coords(i, 4); 
        T li = cells(i, __colidx_l);
        T half_li = cells(i, __colidx_half_l);

        // Generate a 2-D outline with approximately the given meshsize
        //
        // First generate the cylinder ...
        int m = static_cast<int>(li / outline_meshsize) + 1;
        Array<T, Dynamic, 1> mesh1 = Array<T, Dynamic, 1>::LinSpaced(m, -half_li, half_li);
        for (int j = 0; j < m; ++j)
        {
            // Get the corresponding point along the centerline 
            Array<T, 3, 1> p = ri + mesh1(j) * ni;

            // Get the point on the spherocylinder surface obtained by 
            // going up in z by R 
            Array<T, 3, 1> q; 
            q << p(0), p(1), p(2) + R;

            // Get the point on the spherocylinder surface obtained by 
            // going down in z by R
            Array<T, 3, 1> s; 
            s << p(0), p(1), p(2) - R; 

            // For each point, keep track of the r- and z-coordinates and the 
            // cell from which it originates 
            x.push_back(static_cast<double>(p(0)));
            y.push_back(static_cast<double>(p(2))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(q(0))); 
            y.push_back(static_cast<double>(q(2))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(s(0))); 
            y.push_back(static_cast<double>(s(2))); 
            idx.push_back(i); 
        }

        // ... then generate the hemispherical caps
        m = static_cast<int>(boost::math::constants::pi<T>() * R / outline_meshsize) + 1;
        Array<T, Dynamic, 1> mesh2 = Array<T, Dynamic, 1>::LinSpaced(
            m, -boost::math::constants::half_pi<T>(), boost::math::constants::half_pi<T>()
        );
        Array<T, 3, 1> pi = ri - half_li * ni;
        Array<T, 3, 1> qi = ri + half_li * ni;
        Array<T, 2, 1> pi_proj, qi_proj, ni_proj; 
        pi_proj << pi(0), pi(2); 
        qi_proj << qi(0), qi(2);
        ni_proj << ni(0), ni(2); 
        for (int j = 0; j < m; ++j)
        {
            Matrix<T, 2, 2> rot; 
            T cos_theta = cos(mesh2(j)); 
            T sin_theta = sin(mesh2(j)); 
            rot << cos_theta, -sin_theta,
                   sin_theta,  cos_theta; 
            Matrix<T, 2, 1> v = pi_proj.matrix() + R * rot * (-ni_proj).matrix();
            Matrix<T, 2, 1> w = qi_proj.matrix() + R * rot * ni_proj.matrix();

            // For each point, keep track of the x- and y-coordinates and the 
            // cell from which it originates 
            x.push_back(static_cast<double>(v(0))); 
            y.push_back(static_cast<double>(v(1))); 
            idx.push_back(i); 
            x.push_back(static_cast<double>(w(0))); 
            y.push_back(static_cast<double>(w(1))); 
            idx.push_back(i);
        }
    }
   
    return std::make_pair(Boundary2D(x, y).getSimplyConnectedBoundary(), idx);
}

typedef double T; 

int main(int argc, char** argv)
{
    std::string dir = argv[1];
    std::string shape_dir = std::filesystem::path(dir) / "boundaries";

    // For each file in the input directory ... 
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        std::string filename = entry.path();
        if (filename.size() >= 4 && filename.compare(filename.size() - 4, filename.size(), ".txt") == 0)
        {
            // Skip over the lineage file 
            if (filename.compare(filename.size() - 12, filename.size(), "_lineage.txt") == 0)
                continue; 

            // Define the output filename
            std::string basename = std::filesystem::path(filename).stem();  
            std::stringstream ss;
            ss << basename << "_boundary.txt";
            std::string outfilename = std::filesystem::path(shape_dir) / ss.str();  

            // Parse the data file 
            auto result = readCells<T>(filename);
            Array<T, Dynamic, Dynamic> cells = result.first;
            Array<T, Dynamic, 5> rz_coords = projectRZ<T>(cells); 
            std::map<std::string, std::string> params = result.second;
            const T R = static_cast<T>(std::stod(params["R"]));

            // Get the boundary cells
            if (cells.rows() >= 20)
            {
                std::cout << "Computing boundary: " << filename << std::endl;  
                auto boundary = getBoundaryFromOutlines<T>(cells, R, 50);
                AlphaShape2DProperties shape = boundary.first; 
                std::vector<int> idx = boundary.second;            
                std::cout << "... found simply connected boundary: "
                          << shape.is_simple_cycle << std::endl;

                // Get the boundary cell indices
                std::unordered_set<int> boundary_cells; 
                for (const int& v : shape.vertices)
                    boundary_cells.insert(idx[v]);
                std::vector<int> boundary_cells_sorted(boundary_cells.begin(), boundary_cells.end()); 
                std::sort(boundary_cells_sorted.begin(), boundary_cells_sorted.end()); 

                // Output the boundary cell indices
                std::ofstream outfile(outfilename);
                outfile << std::setprecision(10);  
                for (const int& i : boundary_cells_sorted)
                    outfile << "BOUNDARY_CELL\t" << i << '\t'
                            << cells(i, __colidx_id) << '\t'
                            << cells(i, __colidx_rx) << '\t'
                            << cells(i, __colidx_ry) << '\t'
                            << cells(i, __colidx_nx) << '\t'
                            << cells(i, __colidx_ny) << '\t'
                            << cells(i, __colidx_l) << '\t'
                            << rz_coords(i, 0) << '\t'
                            << rz_coords(i, 1) << '\t'
                            << rz_coords(i, 2) << '\t'
                            << rz_coords(i, 3) << '\t'
                            << rz_coords(i, 4) << std::endl; 
                
                // Output each boundary point and edge 
                for (const int& v : shape.vertices)
                    outfile << "BOUNDARY_VERTEX\t"
                            << v << '\t' << shape.x[v] << '\t' << shape.y[v] << std::endl;
                for (const std::pair<int, int>& e : shape.edges)
                    outfile << "BOUNDARY_EDGE\t"
                            << e.first << '\t' << e.second << std::endl;  
            } 
        }
    }

    return 0; 
}
