/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/5/2026
 */

#include <iostream>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/distances.hpp"
#include "../include/utils.hpp"
#include "../include/fields.hpp"
#include "../include/topology.hpp"

std::vector<int> findParentSimplex(std::unordered_map<int, bool>& is_new, 
                                   std::unordered_set<std::string>& prev_simplices, 
                                   std::unordered_map<int, int>& parents,
                                   std::vector<int>& simplex) 
{
    // Infer the putative parent simplex in the previous frame
    const int n = simplex.size();  
    std::vector<int> parent_simplex; 
    for (int i = 0; i < n; ++i)
        parent_simplex.push_back(is_new[simplex[i]] ? parents[simplex[i]] : simplex[i]); 

    // Are there any repeated vertices in the simplex?
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (parent_simplex[i] == parent_simplex[j])
            {
                // If so, then the simplex must be new
                return std::vector<int>();
            }
        }
    }

    // Otherwise, it is a possible simplex; check whether it existed in the
    // previous frame
    std::sort(parent_simplex.begin(), parent_simplex.end()); 
    std::stringstream ss; 
    for (int i = 0; i < n - 1; ++i)
        ss << parent_simplex[i] << ","; 
    ss << parent_simplex[n - 1]; 
    std::string encoding = ss.str();

    // Did the parent simplex exist in the previous frame?
    if (prev_simplices.find(encoding) != prev_simplices.end())
        return parent_simplex;
    else 
        return std::vector<int>();
}

int main(int argc, char** argv)
{
    // Parse input arguments
    std::vector<std::string> cells_filenames, complex_filenames;
    int nframes = 0;        // Impose no maximum number of frames by default
    double tmin = 2.0; 
    double tmax = std::numeric_limits<double>::max(); 
    bool verbose = false; 

    // Parse additional arguments
    if (argc > 4)
    {
        std::vector<std::string> args; 
        for (int i = 4; i < argc; ++i)
            args.push_back(argv[i]);

        // -n: Number of frames 
        auto it = std::find(args.begin(), args.end(), "-n"); 
        if (it != args.end())
        {
            auto next = std::next(it);
            if (next == args.end())
                throw std::runtime_error("Invalid format for input option -n"); 
            nframes = std::stoi(*next); 
        }

        // --tmin: Minimum time 
        it = std::find(args.begin(), args.end(), "--tmin"); 
        if (it != args.end())
        {
            auto next = std::next(it); 
            if (next == args.end())
                throw std::runtime_error("Invalid format for input option --tmin"); 
            tmin = std::stod(*next); 
        } 

        // --tmax: Maximum time 
        it = std::find(args.begin(), args.end(), "--tmax"); 
        if (it != args.end())
        {
            auto next = std::next(it); 
            if (next == args.end())
                throw std::runtime_error("Invalid format for input option --tmax"); 
            tmax = std::stod(*next); 
        }

        // --verbose: Verbosity 
        if (std::find(args.begin(), args.end(), "--verbose") != args.end())
            verbose = true;  
    }

    // Gather the filenames in the given directories 
    std::string cells_dir = argv[1]; 
    std::string complex_dir = argv[2];
    cells_filenames = parseDir(cells_dir, nframes, tmin, tmax).first;
    for (const std::string& filename : cells_filenames)
    {
        std::string basename = std::filesystem::path(filename).stem();  
        std::stringstream ss; 
        ss << basename << "_graph.txt";
        std::string complex_filename = std::filesystem::path(complex_dir) / ss.str(); 
        complex_filenames.push_back(complex_filename);  
    }

    // Locate the lineage file
    nframes = cells_filenames.size(); 
    std::stringstream ss; 
    std::string basename = std::filesystem::path(cells_filenames[0]).stem();
    ss << basename; 
    std::string token;
    std::vector<std::string> tokens; 
    while (std::getline(ss, token, '_'))
        tokens.push_back(token);
    ss.str(std::string()); 
    ss.clear();
    ss << tokens[0]; 
    for (int i = 1; i < tokens.size() - 1; ++i)
        ss << "_" << tokens[i];  
    ss << "_lineage.txt";
    std::string lineage_filename = std::filesystem::path(cells_dir) / ss.str(); 

    // Parse the lineage file 
    std::unordered_map<int, int> parents = readLineage(lineage_filename);  

    // Keep track of the current population of cells 
    std::unordered_set<int> prev_cells; 

    // Keep track of the simplices from the previous frame 
    std::unordered_set<std::string> prev_edges, prev_triangles, prev_tetrahedra; 

    // Keep track of simplex lifetimes 
    std::unordered_map<std::pair<int, int>,
                       std::vector<std::pair<double, double> >,
                       boost::hash<std::pair<int, int> > > edge_lifetimes;
    std::unordered_map<std::tuple<int, int, int>,
                       std::vector<std::pair<double, double> >,
                       boost::hash<std::tuple<int, int, int> > > triangle_lifetimes; 
    std::unordered_map<std::tuple<int, int, int, int>,
                       std::vector<std::pair<double, double> >, 
                       boost::hash<std::tuple<int, int, int, int> > > tetrahedron_lifetimes; 
   
    // Keep track of the timepoint of the current frame 
    double t_curr = tmin; 

    // Parse the cells and the simplicial complex at each frame ... 
    for (int i = 0 ; i < nframes; ++i)
    {
        std::string cells_filename = cells_filenames[i]; 
        std::string complex_filename = complex_filenames[i];
        std::cout << "Parsing frame #" << i << ":\n"
                  << "- Frame filename: " << cells_filename << std::endl
                  << "- Complex filename: " << complex_filename << std::endl;

        // Parse simulation frame and simplicial complex 
        auto result = readCells<double>(cells_filename);
        Array<double, Dynamic, Dynamic> cells = result.first;
        std::map<std::string, std::string> params = result.second;
        double t_next = std::stod(params["t_curr"]); 
        SimplicialComplex3D<double> cplex;
        Array<double, Dynamic, 3> coords(cells.rows(), 3);
        coords.col(0) = cells.col(__colidx_rx); 
        coords.col(1) = cells.col(__colidx_ry); 
        coords.col(2) = cells.col(__colidx_rz);
        cplex.read(complex_filename, coords);

        // Get the simplices in the complex 
        Array<int, Dynamic, 2> edges = cplex.getSimplices<1>();
        Array<int, Dynamic, 3> triangles = cplex.getSimplices<2>();
        Array<int, Dynamic, 4> tetrahedra = cplex.getSimplices<3>(); 

        // If we are not processing the very first frame ... 
        if (i > 0)
        {
            // Which cells are new?
            std::unordered_map<int, bool> is_new; 
            for (int j = 0; j < cells.rows(); ++j)
            {
                int cell_id = static_cast<int>(cells(j, __colidx_id)); 
                is_new[cell_id] = (prev_cells.find(cell_id) == prev_cells.end()); 
            }

            // Keep track of parent simplices whose lifetimes have been inherited 
            std::unordered_set<std::pair<int, int>,
                               boost::hash<std::pair<int, int> > > edges_to_remove; 
            std::unordered_set<std::tuple<int, int, int>,
                               boost::hash<std::tuple<int, int, int> > > triangles_to_remove;
            std::unordered_set<std::tuple<int, int, int, int>,
                               boost::hash<std::tuple<int, int, int, int> > > tetrahedra_to_remove;  

            // Run through the edges in the complex ... 
            for (int j = 0; j < edges.rows(); ++j)
            {
                int k = edges(j, 0); 
                int m = edges(j, 1); 
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                if (cell_id1 > cell_id2)
                {
                    int tmp = cell_id1; 
                    cell_id1 = cell_id2;
                    cell_id2 = tmp; 
                }
                std::pair<int, int> edge = std::make_pair(cell_id1, cell_id2);

                // Define string encoding 
                std::stringstream ss; 
                ss << cell_id1 << "," << cell_id2;
                std::string encoding = ss.str();

                // Were both cells present in the previous population?
                if (!is_new[cell_id1] && !is_new[cell_id2])
                { 
                    // Was the edge present in the previous population?
                    if (prev_edges.find(encoding) != prev_edges.end()) 
                    {
                        // If so, update the lifetime interval for this edge
                        std::vector<std::pair<double, double> >& lifetimes = edge_lifetimes[edge];  
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        lifetimes.pop_back(); 
                        lifetimes.push_back(new_lifetime); 
                    }
                    else    // Otherwise, this is a new edge 
                    {
                        // Add a new lifetime interval for this edge
                        if (edge_lifetimes.find(edge) == edge_lifetimes.end())
                            edge_lifetimes[edge] = std::vector<std::pair<double, double> >();
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        edge_lifetimes[edge].push_back(new_lifetime);
                    }
                }
                // If not, look for a parent edge 
                else 
                {
                    // The edge must be new, so add an entry 
                    edge_lifetimes[edge] = std::vector<std::pair<double, double> >(); 

                    // Look for a parent edge 
                    std::vector<int> edge_ {cell_id1, cell_id2}; 
                    std::vector<int> parent_ = findParentSimplex(
                        is_new, prev_edges, parents, edge_
                    );
                    
                    // If the parent edge did not exist, then the edge is new
                    if (parent_.size() == 0)
                    {
                        // Add a new lifetime interval for this edge 
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        edge_lifetimes[edge].push_back(new_lifetime);
                    }
                    // Otherwise, inherit the parent edge lifetime and mark 
                    // the parent edge for removal 
                    else 
                    {
                        // Extract the lifetime intervals for the parent edge
                        // and remove the current interval
                        std::pair<int, int> parent = std::make_pair(parent_[0], parent_[1]);
                        std::vector<std::pair<double, double> >& lifetimes = edge_lifetimes[parent];
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        edges_to_remove.insert(parent); 

                        // Copy over the updated lifetime interval for the 
                        // child edge  
                        edge_lifetimes[edge].push_back(new_lifetime); 
                    }
                } 
            }

            // Remove the last lifetime interval for each parent edge whose 
            // lifetime interval was inherited 
            for (auto&& edge : edges_to_remove)
                edge_lifetimes[edge].pop_back(); 

            // Run through the triangles in the complex ... 
            for (int j = 0; j < triangles.rows(); ++j)
            {
                int k = triangles(j, 0); 
                int m = triangles(j, 1);
                int p = triangles(j, 2);  
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                int cell_id3 = static_cast<int>(cells(p, __colidx_id));
                std::vector<int> triangle_ {cell_id1, cell_id2, cell_id3}; 
                std::sort(triangle_.begin(), triangle_.end()); 
                auto triangle = std::make_tuple(triangle_[0], triangle_[1], triangle_[2]);

                // Define string encoding 
                std::stringstream ss; 
                ss << triangle_[0] << "," << triangle_[1] << "," << triangle_[2]; 
                std::string encoding = ss.str();  

                // Were all three cells present in the previous population?
                if (!is_new[cell_id1] && !is_new[cell_id2] && !is_new[cell_id3])
                { 
                    // Was the triangle present in the previous population?
                    if (prev_triangles.find(encoding) != prev_triangles.end()) 
                    {
                        // If so, update the lifetime interval for this triangle
                        std::vector<std::pair<double, double> >& lifetimes
                            = triangle_lifetimes[triangle]; 
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        lifetimes.pop_back(); 
                        lifetimes.push_back(new_lifetime); 
                    }
                    else    // Otherwise, this is a new triangle 
                    {
                        // Add a new lifetime interval for this triangle
                        if (triangle_lifetimes.find(triangle) == triangle_lifetimes.end())
                        {
                            triangle_lifetimes[triangle]
                                = std::vector<std::pair<double, double> >();
                        }
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        triangle_lifetimes[triangle].push_back(new_lifetime);
                    }
                }
                // If not, look for a parent triangle 
                else 
                {
                    // The triangle must be new, so add an entry 
                    triangle_lifetimes[triangle] = std::vector<std::pair<double, double> >(); 

                    // Look for a parent triangle
                    std::vector<int> parent_ = findParentSimplex(
                        is_new, prev_triangles, parents, triangle_
                    );
                    
                    // If the parent triangle did not exist, then the triangle
                    // is new
                    if (parent_.size() == 0)
                    {
                        // Add a new lifetime interval for this triangle 
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        triangle_lifetimes[triangle].push_back(new_lifetime);
                    }
                    // Otherwise, inherit the parent triangle lifetime and mark
                    // the parent triangle for removal 
                    else 
                    {
                        // Extract the lifetime intervals for the parent triangle
                        // and remove the current interval
                        auto parent = std::make_tuple(parent_[0], parent_[1], parent_[2]);
                        std::vector<std::pair<double, double> >& lifetimes
                            = triangle_lifetimes[parent];
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        triangles_to_remove.insert(parent); 

                        // Copy over the updated lifetime interval for the 
                        // child triangle  
                        triangle_lifetimes[triangle].push_back(new_lifetime); 
                    }
                }
            }

            // Remove the last lifetime interval for each parent triangle whose 
            // lifetime interval was inherited 
            for (auto&& triangle : triangles_to_remove)
                triangle_lifetimes[triangle].pop_back();

            // Run through the tetrahedra in the complex ... 
            for (int j = 0; j < tetrahedra.rows(); ++j)
            {
                int k = tetrahedra(j, 0); 
                int m = tetrahedra(j, 1);
                int p = tetrahedra(j, 2);
                int q = tetrahedra(j, 3);  
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                int cell_id3 = static_cast<int>(cells(p, __colidx_id));
                int cell_id4 = static_cast<int>(cells(q, __colidx_id)); 
                std::vector<int> tetrahedron_ {cell_id1, cell_id2, cell_id3, cell_id4}; 
                std::sort(tetrahedron_.begin(), tetrahedron_.end()); 
                auto tetrahedron = std::make_tuple(
                    tetrahedron_[0], tetrahedron_[1], tetrahedron_[2], tetrahedron_[3]
                );

                // Define string encoding 
                std::stringstream ss; 
                ss << tetrahedron_[0] << "," << tetrahedron_[1] << ","
                   << tetrahedron_[2] << "," << tetrahedron_[3]; 
                std::string encoding = ss.str();  

                // Were all four cells present in the previous population?
                if (!is_new[cell_id1] && !is_new[cell_id2] && !is_new[cell_id3] && !is_new[cell_id4])
                { 
                    // Was the tetrahedron present in the previous population?
                    if (prev_tetrahedra.find(encoding) != prev_tetrahedra.end()) 
                    {
                        // If so, update the lifetime interval for this tetrahedron
                        std::vector<std::pair<double, double> >& lifetimes
                            = tetrahedron_lifetimes[tetrahedron]; 
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        lifetimes.pop_back(); 
                        lifetimes.push_back(new_lifetime); 
                    }
                    else    // Otherwise, this is a new tetrahedron 
                    {
                        // Add a new lifetime interval for this tetrahedron
                        if (tetrahedron_lifetimes.find(tetrahedron) == tetrahedron_lifetimes.end())
                        {
                            tetrahedron_lifetimes[tetrahedron]
                                = std::vector<std::pair<double, double> >();
                        }
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        tetrahedron_lifetimes[tetrahedron].push_back(new_lifetime);
                    }
                }
                // If not, look for a parent tetrahedron 
                else 
                {
                    // The tetrahedron must be new, so add an entry 
                    tetrahedron_lifetimes[tetrahedron]
                        = std::vector<std::pair<double, double> >(); 

                    // Look for a parent tetrahedron
                    std::vector<int> parent_ = findParentSimplex(
                        is_new, prev_tetrahedra, parents, tetrahedron_
                    );
                    
                    // If the parent tetrahedron did not exist, then the
                    // tetrahedron is new
                    if (parent_.size() == 0)
                    {
                        // Add a new lifetime interval for this tetrahedron 
                        auto new_lifetime = std::make_pair(t_curr, t_next);
                        tetrahedron_lifetimes[tetrahedron].push_back(new_lifetime);
                    }
                    // Otherwise, inherit the parent tetrahedron lifetime and mark
                    // the parent tetrahedron for removal 
                    else 
                    {
                        // Extract the lifetime intervals for the parent
                        // tetrahedron and remove the current interval
                        auto parent = std::make_tuple(
                            parent_[0], parent_[1], parent_[2], parent_[3]
                        );
                        std::vector<std::pair<double, double> >& lifetimes
                            = tetrahedron_lifetimes[parent];
                        auto lifetime = lifetimes.back();
                        auto new_lifetime = std::make_pair(lifetime.first, t_curr);
                        tetrahedra_to_remove.insert(parent); 

                        // Copy over the updated lifetime interval for the 
                        // child tetrahedron  
                        tetrahedron_lifetimes[tetrahedron].push_back(new_lifetime); 
                    }
                }
            }

            // Remove the last lifetime interval for each parent tetrahedron whose 
            // lifetime interval was inherited 
            for (auto&& tetrahedron : tetrahedra_to_remove)
                tetrahedron_lifetimes[tetrahedron].pop_back(); 
        }
        else     // Otherwise, initialize lifetimes of all simplices 
        {
            for (int j = 0; j < edges.rows(); ++j)
            {
                int k = edges(j, 0); 
                int m = edges(j, 1); 
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                if (cell_id1 > cell_id2)
                {
                    int tmp = cell_id1; 
                    cell_id1 = cell_id2;
                    cell_id2 = tmp; 
                }
                std::pair<int, int> edge = std::make_pair(cell_id1, cell_id2);
                edge_lifetimes[edge] = std::vector<std::pair<double, double> >();
                edge_lifetimes[edge].push_back(std::make_pair(t_curr, t_next));  
            }
            for (int j = 0; j < triangles.rows(); ++j)
            {
                int k = triangles(j, 0); 
                int m = triangles(j, 1);
                int p = triangles(j, 2);  
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                int cell_id3 = static_cast<int>(cells(p, __colidx_id));
                std::vector<int> triangle_ {cell_id1, cell_id2, cell_id3}; 
                std::sort(triangle_.begin(), triangle_.end()); 
                auto triangle = std::make_tuple(triangle_[0], triangle_[1], triangle_[2]);
                triangle_lifetimes[triangle] = std::vector<std::pair<double, double> >();
                triangle_lifetimes[triangle].push_back(std::make_pair(t_curr, t_next));  
            }
            for (int j = 0; j < tetrahedra.rows(); ++j)
            {
                int k = tetrahedra(j, 0); 
                int m = tetrahedra(j, 1);
                int p = tetrahedra(j, 2); 
                int q = tetrahedra(j, 3);  
                int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
                int cell_id2 = static_cast<int>(cells(m, __colidx_id));
                int cell_id3 = static_cast<int>(cells(p, __colidx_id));
                int cell_id4 = static_cast<int>(cells(q, __colidx_id)); 
                std::vector<int> tetrahedron_ {cell_id1, cell_id2, cell_id3, cell_id4}; 
                std::sort(tetrahedron_.begin(), tetrahedron_.end()); 
                auto tetrahedron = std::make_tuple(
                    tetrahedron_[0], tetrahedron_[1], tetrahedron_[2], tetrahedron_[3]
                );
                tetrahedron_lifetimes[tetrahedron] = std::vector<std::pair<double, double> >();
                tetrahedron_lifetimes[tetrahedron].push_back(std::make_pair(t_curr, t_next));  
            }
        }
       
        // Update current population
        prev_cells.clear();
        for (int j = 0; j < cells.rows(); ++j)
            prev_cells.insert(static_cast<int>(cells(j, __colidx_id)));

        // Update current collections of simplices 
        prev_edges.clear(); 
        for (int j = 0; j < edges.rows(); ++j)
        {
            int k = edges(j, 0); 
            int m = edges(j, 1); 
            int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
            int cell_id2 = static_cast<int>(cells(m, __colidx_id));
            if (cell_id1 > cell_id2)
            {
                int tmp = cell_id1; 
                cell_id1 = cell_id2;
                cell_id2 = tmp; 
            }
            std::stringstream ss; 
            ss << cell_id1 << "," << cell_id2; 
            prev_edges.insert(ss.str()); 
        }
        prev_triangles.clear(); 
        for (int j = 0; j < triangles.rows(); ++j)
        {
            int k = triangles(j, 0); 
            int m = triangles(j, 1);
            int p = triangles(j, 2);  
            int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
            int cell_id2 = static_cast<int>(cells(m, __colidx_id));
            int cell_id3 = static_cast<int>(cells(p, __colidx_id));
            std::vector<int> triangle_ {cell_id1, cell_id2, cell_id3}; 
            std::sort(triangle_.begin(), triangle_.end());
            std::stringstream ss; 
            ss << triangle_[0] << "," << triangle_[1] << "," << triangle_[2]; 
            prev_triangles.insert(ss.str()); 
        }
        prev_tetrahedra.clear(); 
        for (int j = 0; j < tetrahedra.rows(); ++j)
        {
            int k = tetrahedra(j, 0); 
            int m = tetrahedra(j, 1);
            int p = tetrahedra(j, 2); 
            int q = tetrahedra(j, 3);  
            int cell_id1 = static_cast<int>(cells(k, __colidx_id)); 
            int cell_id2 = static_cast<int>(cells(m, __colidx_id));
            int cell_id3 = static_cast<int>(cells(p, __colidx_id));
            int cell_id4 = static_cast<int>(cells(q, __colidx_id)); 
            std::vector<int> tetrahedron_ {cell_id1, cell_id2, cell_id3, cell_id4}; 
            std::sort(tetrahedron_.begin(), tetrahedron_.end()); 
            std::stringstream ss; 
            ss << tetrahedron_[0] << "," << tetrahedron_[1] << ","
               << tetrahedron_[2] << "," << tetrahedron_[3]; 
            prev_tetrahedra.insert(ss.str()); 
        }
        
        // Update current time 
        t_curr = t_next; 
    }

    // Write lifetimes to file
    std::string outfilename = argv[3];
    std::ofstream outfile(outfilename);
    outfile << std::setprecision(10);  
    outfile << "# tmin = " << tmin << std::endl; 
    outfile << "# tmax = " << tmax << std::endl; 
    for (auto&& [edge, lifetimes] : edge_lifetimes)
    {
        if (lifetimes.size() > 0)
        {
            outfile << "EDGE\t" << edge.first << '\t' << edge.second << '\t';
            for (int i = 0; i < lifetimes.size() - 1; ++i)
            {
                outfile << lifetimes[i].first << ',' << lifetimes[i].second << '\t'; 
            }
            outfile << lifetimes[lifetimes.size() - 1].first << ','
                    << lifetimes[lifetimes.size() - 1].second << std::endl;
        } 
    }
    for (auto&& [triangle, lifetimes] : triangle_lifetimes)
    {
        if (lifetimes.size() > 0)
        {
            outfile << "TRIANGLE\t" << std::get<0>(triangle) << '\t'
                                    << std::get<1>(triangle) << '\t'
                                    << std::get<2>(triangle) << '\t'; 
            for (int i = 0; i < lifetimes.size() - 1; ++i)
            {
                outfile << lifetimes[i].first << ',' << lifetimes[i].second << '\t'; 
            }
            outfile << lifetimes[lifetimes.size() - 1].first << ','
                    << lifetimes[lifetimes.size() - 1].second << std::endl;
        }
    } 
    for (auto&& [tetrahedron, lifetimes] : tetrahedron_lifetimes)
    {
        if (lifetimes.size() > 0)
        {
            outfile << "TETRAHEDRON\t" << std::get<0>(tetrahedron) << '\t'
                                       << std::get<1>(tetrahedron) << '\t'
                                       << std::get<2>(tetrahedron) << '\t'
                                       << std::get<3>(tetrahedron) << '\t'; 
            for (int i = 0; i < lifetimes.size() - 1; ++i)
            {
                outfile << lifetimes[i].first << ',' << lifetimes[i].second << '\t'; 
            }
            outfile << lifetimes[lifetimes.size() - 1].first << ','
                    << lifetimes[lifetimes.size() - 1].second << std::endl;
        }
    } 
    outfile.close(); 

    return 0;  
}
