/**
 * Classes and functions for graphs and simplicial complexes. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/26/2025
 */

#ifndef SIMPLICIAL_COMPLEXES_3D_HPP
#define SIMPLICIAL_COMPLEXES_3D_HPP

#include <iostream>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include "distances.hpp"
#include "mechanics.hpp"
#include "utils.hpp"

using namespace Eigen;

typedef boost::adjacency_list<boost::hash_setS, boost::vecS, boost::undirectedS> Graph; 

using Simplex_tree = Gudhi::Simplex_tree<>; 
using Vertex_handle = Simplex_tree::Vertex_handle;
using Simplex_handle = Simplex_tree::Simplex_handle;
using Field_Zp = Gudhi::persistent_cohomology::Field_Zp; 
using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;

/** ------------------------------------------------------------------- //
 *                            GRAPH UTILITIES                           //
 *  ------------------------------------------------------------------- */
/**
 * Define and return a graph in which each cell is a vertex, and two vertices
 * are connected by an edge if (1) both cells are in group 1 and (2) the two
 * cells are within 2 * R distance of each other.
 *
 * @param cells Input population of cells.  
 * @param R Cell radius, including the EPS. 
 * @param Ldiv Cell division length. 
 * @returns Cell-cell neighbor graph. 
 */
template <typename T>
Graph getNeighborGraph(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                       const T R, const T Ldiv)
{
    // Get the cell-cell distance between each pair of neighboring cells in
    // the population
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, 2 * R, R, Ldiv);

    // Define a graph and add a vertex for each cell  
    Graph graph(cells.rows());

    // Add an edge between each pair of group 1 cells that are within 2 * R
    // distance 
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = neighbors(k, 0); 
        int j = neighbors(k, 1); 
        T dij = neighbors(k, Eigen::seq(2, 4)).matrix().norm();
        if (dij < 2 * R && cells(i, __colidx_group) == 1 && cells(j, __colidx_group) == 1)
            boost::add_edge(i, j, graph);
    }

    return graph; 
}

/**
 * Get the connected components in the given graph.
 *
 * The i-th entry in the returned std::vector is the index of the connected
 * component containing vertex i.
 *
 * @param graph Input graph. 
 * @returns Indices of connected components containing each vertex.  
 */
template <typename T>
std::vector<int> getConnectedComponents(const Graph& graph)
{
    std::vector<int> components(boost::num_vertices(graph));
    boost::connected_components(graph, &components[0]);

    return components;  
}

/**
 * Get the degrees of all vertices in the given graph. 
 *
 * The i-th entry in the returned array is the number of edges incident on 
 * the vertex i. 
 *
 * @param graph Input graph. 
 * @returns Degrees of vertices in the graph. 
 */
template <typename T>
Array<int, Dynamic, 1> getDegrees(const Graph& graph)
{
    Array<int, Dynamic, 1> degrees = Array<int, Dynamic, 1>::Zero(boost::num_vertices(graph));
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it; 
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        degrees(i) += 1; 
        degrees(j) += 1; 
    }

    return degrees; 
}

/**
 * Get the local clustering coefficients for all vertices in the given graph. 
 *
 * @param graph Input graph. 
 * @returns Local clustering coefficients for all vertices in the graph. 
 */
template <typename T>
Array<T, Dynamic, 1> getLocalClusteringCoefficients(const Graph& graph)
{
    Array<T, Dynamic, 1> coefs = Array<T, Dynamic, 1>::Zero(boost::num_vertices(graph));
    std::pair<boost::graph_traits<Graph>::out_edge_iterator, 
              boost::graph_traits<Graph>::out_edge_iterator> it;
    for (int i = 0; i < boost::num_vertices(graph); ++i)
    {
        // First identify the neighbors of i
        std::vector<int> neighbors;  
        for (it = boost::out_edges(i, graph); it.first != it.second; ++it.first)
        {
            // Rigorously check which vertex is the neighbor and which is i
            // (the former should always be the target) 
            boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
            int j = boost::source(edge, graph); 
            int k = boost::target(edge, graph);
            if (i == j) 
                neighbors.push_back(k); 
            else 
                neighbors.push_back(j); 
        }
        int n_neighbors = neighbors.size();

        // If degree = 0 or 1, then the local clustering coefficient is undefined
        if (n_neighbors == 0 || n_neighbors == 1)
        {
            coefs(i) = -1;
        }
        // Otherwise, check, for each pair of neighbors, if they are connected
        // by a third edge
        else
        {
            T n_cluster = 0.0; 
            for (int j = 0; j < neighbors.size(); ++j)
            {
                for (int k = j + 1; k < neighbors.size(); ++k)
                {
                    if (boost::edge(neighbors[j], neighbors[k], graph).second)
                        n_cluster += 1; 
                }
            } 
            coefs(i) = 2 * n_cluster / (n_neighbors * (n_neighbors - 1));
        }    
    }

    return coefs;
}

/**
 * Get all triangles (3-cliques) in the given graph. 
 *
 * Each row in the returned array is a combination of three vertices that 
 * form a triangle. 
 *
 * @param graph Input graph. 
 * @returns Array of triangle-forming vertices. 
 */
template <typename T>
Array<int, Dynamic, 3> getTriangles(const Graph& graph)
{
    const int n = boost::num_vertices(graph);
    int nt = 0;  
    Array<int, Dynamic, 3> triangles(nt, 3);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (boost::edge(i, j, graph).second)
            {
                for (int k = j + 1; k < n; ++k)
                {
                    if (boost::edge(j, k, graph).second && boost::edge(i, k, graph).second)
                    {
                        nt++; 
                        triangles.conservativeResize(nt, 3); 
                        triangles(nt - 1, 0) = i; 
                        triangles(nt - 1, 1) = j; 
                        triangles(nt - 1, 2) = k; 
                    }
                }
            }
        }
    }

    return triangles; 
}

/**
 * Get all tetrahedra (4-cliques) in the given graph. 
 *
 * Each row in the returned array is a combination of four vertices that 
 * form a tetrahedron.  
 *
 * @param graph Input graph. 
 * @returns Array of tetrahedron-forming vertices. 
 */
template <typename T>
Array<int, Dynamic, 4> getTetrahedra(const Graph& graph)
{
    const int n = boost::num_vertices(graph);
    int nt = 0;  
    Array<int, Dynamic, 4> tetrahedra(nt, 4);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (boost::edge(i, j, graph).second)    // Check that i-j
            {
                for (int k = j + 1; k < n; ++k)
                {
                    // Check that j-k and i-k
                    if (boost::edge(j, k, graph).second && boost::edge(i, k, graph).second)
                    {
                        for (int m = k + 1; m < n; ++m)
                        {
                            // Check that i-m, j-m, and k-m
                            if (boost::edge(i, m, graph).second && 
                                boost::edge(j, m, graph).second &&
                                boost::edge(k, m, graph).second)
                            {
                                nt++; 
                                tetrahedra.conservativeResize(nt, 4); 
                                tetrahedra(nt - 1, 0) = i; 
                                tetrahedra(nt - 1, 1) = j; 
                                tetrahedra(nt - 1, 2) = k;
                                tetrahedra(nt - 1, 3) = m;
                            }
                        } 
                    }
                }
            }
        }
    }

    return tetrahedra; 
}

/**
 * Write the given graph and its connectivity information to file. 
 *
 * @param graph Input graph. 
 * @param components Vector of connected component sizes, from largest to
 *                   smallest.
 * @param degrees Vector of vertex degrees. 
 * @param filename Path to output file. 
 * @param write_cluster_coefs If true, write local clustering coefficients
 *                            to file. 
 * @param cluster_coefs Vector of local clustering coefficients. 
 * @param write_triangles If true, write triangles to file. 
 * @param triangles Array of vertices that form triangles. 
 * @param write_tetrahedra If true, write tetrahedra to file. 
 * @param tetrahedra Array of vertices that form tetrahedra. 
 */
template <typename T>
void writeGraph(const Graph& graph, std::vector<int>& components,
                const Ref<const Array<int, Dynamic, 1> >& degrees,
                const std::string filename, const bool write_cluster_coefs,  
                const Ref<const Array<T, Dynamic, 1> >& cluster_coefs,         
                const bool write_triangles, 
                const Ref<const Array<int, Dynamic, 3> >& triangles,
                const bool write_tetrahedra,
                const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
{
    // Get the component sizes 
    int num_components = *std::max_element(components.begin(), components.end()) + 1;
    Array<int, Dynamic, 1> component_sizes = Array<int, Dynamic, 1>::Zero(num_components); 
    for (int i = 0; i < components.size(); ++i)
        component_sizes(components[i]) += 1;

    // Open output file and write header 
    std::ofstream outfile(filename);
    outfile << std::setprecision(10); 
    outfile << "NUM_VERTICES\t" << boost::num_vertices(graph) << std::endl; 
    outfile << "NUM_EDGES\t" << boost::num_edges(graph) << std::endl; 
    outfile << "NUM_COMPONENTS\t" << num_components << std::endl;
    if (write_triangles)
        outfile << "NUM_TRIANGLES\t" << triangles.rows() << std::endl;
    if (write_tetrahedra)
        outfile << "NUM_TETRAHEDRA\t" << tetrahedra.rows() << std::endl;

    // Compute the degree distribution
    Array<int, Dynamic, 1> degree_dist = Array<int, Dynamic, 1>::Zero(degrees.maxCoeff() + 1);
    for (int i = 0; i < boost::num_vertices(graph); ++i)
        degree_dist(degrees(i)) += 1;
    for (int i = 0; i <= degrees.maxCoeff(); ++i) 
        outfile << "DEGREE_DIST\t" << i << '\t' << degree_dist(i) << std::endl; 

    // Write the vertices to the output file 
    for (int i = 0; i < boost::num_vertices(graph); ++i)
    {
        if (write_cluster_coefs)
            outfile << "VERTEX\t" << i << "\tCOMPONENT:" << components[i]
                                       << "\tDEGREE:" << degrees(i)
                                       << "\tCLUSTER:" << cluster_coefs(i) << std::endl;
        else 
            outfile << "VERTEX\t" << i << "\tCOMPONENT:" << components[i]
                                       << "\tDEGREE:" << degrees(i) << std::endl;
    }        

    // Write the edges to the output file
    std::pair<boost::graph_traits<Graph>::edge_iterator, 
              boost::graph_traits<Graph>::edge_iterator> it; 
    for (it = boost::edges(graph); it.first != it.second; ++it.first)
    {
        boost::graph_traits<Graph>::edge_descriptor edge = *(it.first);
        int i = boost::source(edge, graph); 
        int j = boost::target(edge, graph);
        outfile << "EDGE\t" << i << '\t' << j << std::endl; 
    }

    // Write the component sizes to the output file 
    for (int i = 0; i < num_components; ++i)
        outfile << "COMPONENT\t" << i << '\t' << component_sizes(i) << std::endl;

    // Write triangles to the output file, if desired 
    if (write_triangles)
    {
        for (int i = 0; i < triangles.rows(); ++i)
        {
            outfile << "TRIANGLE\t" << triangles(i, 0) << '\t'
                                    << triangles(i, 1) << '\t'
                                    << triangles(i, 2) << std::endl;
        } 
    }
    
    // Write tetrahedra to the output file, if desired 
    if (write_tetrahedra)
    {
        for (int i = 0; i < tetrahedra.rows(); ++i)
        {
            outfile << "TETRAHEDRA\t" << tetrahedra(i, 0) << '\t'
                                      << tetrahedra(i, 1) << '\t'
                                      << tetrahedra(i, 2) << '\t'
                                      << tetrahedra(i, 3) << std::endl; 
        }
    }

    // Close output file 
    outfile.close();
}

/** ------------------------------------------------------------------- //
 *                         SIMPLICIAL COMPLEXES                         //
 *  ------------------------------------------------------------------- */
/**
 * A wrapper class around Gudhi::Simplex_tree<> for 3-D simplicial complexes. 
 */
template <typename T>
class SimplicialComplex3D
{
    private:
        Array<T, Dynamic, 3> points;     // Array of point coordinates 
        Simplex_tree tree;               // Simplex tree 

    public:
        /**
         * Default constructor, which takes in arrays of points, edges, 
         * triangles, and tetrahedra. 
         *
         * @param points Array of point coordinates. 
         * @param edges Array of vertices connected by edges. 
         * @param triangles Array of vertices that form triangles. 
         * @param tetrahedra Array of vertices that form tetrahedra. 
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points,
                            const Ref<const Array<int, Dynamic, 2> >& edges,
                            const Ref<const Array<int, Dynamic, 3> >& triangles, 
                            const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
        {
            this->points = points; 
            
            // Construct the simplex tree, one simplex at a time 
            for (int i = 0; i < points.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex { i }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < edges.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex { edges(i, 0), edges(i, 1) }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < triangles.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    triangles(i, 0), triangles(i, 1), triangles(i, 2)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < tetrahedra.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    tetrahedra(i, 0), tetrahedra(i, 1), tetrahedra(i, 2), tetrahedra(i, 3)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
        }

        /**
         * Alternative constructor, which takes in an array of points and an 
         * input graph, and forms the corresponding alpha-complex (i.e., 
         * each 3-clique is "filled in" to form a triangle, and each 4-clique
         * is "filled in" to form a tetrahedron).
         *
         * @param points Array of point coordinates. 
         * @param graph Input graph. 
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points, 
                            Graph& graph)
        {
            this->points = points;

            // Construct the simplex tree, starting with the 0-simplices 
            for (int i = 0; i < points.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex { i }; 
                this->tree.insert_simplex(simplex, 0);
            }
            
            // Parse the edges from the graph
            std::pair<boost::graph_traits<Graph>::edge_iterator, 
                      boost::graph_traits<Graph>::edge_iterator> it;
            for (it = boost::edges(graph); it.first != it.second; ++it.first)
            {
                boost::graph_traits<Graph>::edge_descriptor edge = *(it.first); 
                int j = boost::source(edge, graph); 
                int k = boost::target(edge, graph);
                std::vector<Vertex_handle> simplex { j, k };
                this->tree.insert_simplex(simplex, 0);
            }

            // Define the triangles and tetrahedra from the 3- and 4-cliques 
            // in the graph (this gives the alpha-complex of the graph)
            Array<int, Dynamic, 3> triangles = ::getTriangles<T>(graph); 
            Array<int, Dynamic, 4> tetrahedra = ::getTetrahedra<T>(graph);

            // Add them to the simplex tree 
            for (int i = 0; i < triangles.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    triangles(i, 0), triangles(i, 1), triangles(i, 2)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < tetrahedra.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    tetrahedra(i, 0), tetrahedra(i, 1), tetrahedra(i, 2), tetrahedra(i, 3)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
        }

        /**
         * Alternative constructor, which takes in an array of points and 
         * a pre-constructed simplex tree. 
         *
         * @param points Array of point coordinates. 
         * @param tree Input simplex tree.
         */
        SimplicialComplex3D(const Ref<const Array<T, Dynamic, 3> >& points, 
                            Simplex_tree& tree)
        {
            this->points = points;
            this->tree = tree; 
        }

        /**
         * Trivial destructor. 
         */
        ~SimplicialComplex3D()
        {
        }

        /**
         * Return the array of point coordinates. 
         *
         * @returns Array of point coordinates.  
         */
        Array<T, Dynamic, 3> getPoints() const
        {
            return this->points; 
        }

        /**
         * Return the simplices of the given dimension. 
         *
         * If `sort` is true, then the vertices specifying each simplex are 
         * sorted in ascending order. 
         *
         * @param sort If true, sort the vertices specifying each simplex 
         *             in ascending order. 
         * @returns Array of vertices specifying each simplex of the given 
         *          dimension. 
         */
        template <int Dim>
        Array<int, Dynamic, Dim + 1> getSimplices(const bool sort = true) const
        {
            // Run through the simplices in the complex
            std::vector<std::vector<int> > simplices_;
            for (auto& simplex : this->tree.skeleton_simplex_range(Dim))
            {
                // If the simplex has the right dimension ...
                if (this->tree.dimension(simplex) == Dim)
                {
                    std::vector<int> simplex_vertices;
                    for (auto vertex : this->tree.simplex_vertex_range(simplex))
                        simplex_vertices.push_back(vertex);
                    
                    // Ensure that the vertices are sorted, if desired 
                    if (sort)
                        std::sort(simplex_vertices.begin(), simplex_vertices.end());

                    simplices_.push_back(simplex_vertices);
                } 
            }
            
            // Re-organize as an array and return  
            Array<int, Dynamic, Dim + 1> simplices(simplices_.size(), Dim + 1);
            int i = 0; 
            for (auto it1 = simplices_.begin(); it1 != simplices_.end(); ++it1)
            {
                int j = 0;
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
                {
                    simplices(i, j) = *it2;
                    j++;
                }
                i++;  
            } 
            return simplices;
        }

        /**
         * Update the simplicial complex with the given points, edges, 
         * triangles, and tetrahedra. 
         *
         * @param points Array of point coordinates. 
         * @param edges Array of vertices connected by edges. 
         * @param triangles Array of vertices that form triangles. 
         * @param tetrahedra Array of vertices that form tetrahedra. 
         */
        void update(const Ref<const Array<T, Dynamic, 3> >& points,
                    const Ref<const Array<int, Dynamic, 2> >& edges,
                    const Ref<const Array<int, Dynamic, 3> >& triangles, 
                    const Ref<const Array<int, Dynamic, 4> >& tetrahedra)
        {
            this->points = points;
            this->tree.clear();  
            
            // Construct the simplex tree, one simplex at a time 
            for (int i = 0; i < points.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex { i }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < edges.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex { edges(i, 0), edges(i, 1) }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < triangles.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    triangles(i, 0), triangles(i, 1), triangles(i, 2)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
            for (int i = 0; i < tetrahedra.rows(); ++i)
            {
                std::vector<Vertex_handle> simplex {
                    tetrahedra(i, 0), tetrahedra(i, 1), tetrahedra(i, 2), tetrahedra(i, 3)
                }; 
                this->tree.insert_simplex(simplex, 0);
            }
        }

        int getNumFullDimCofaces(const Ref<const Array<int, Dynamic, 1> >& simplex) const
        {
            const int nvertices = simplex.size();
            const int dim = this->tree.dimension(); 
            const int codim = dim - (nvertices - 1);  
            std::vector<Vertex_handle> simplex_;
            for (int i = 0; i < nvertices; ++i)
                simplex_.push_back(simplex(i));
            int ncofaces = 0; 

            // Run through the codimension-1 cofaces of the given simplex ...  
            for (auto& coface : this->tree.cofaces_simplex_range(this->tree.find(simplex_), codim))
                ncofaces++; 

            return ncofaces; 
        }

        int getNumFullDimCofaces(std::vector<int>& simplex) const
        {
            const int nvertices = simplex.size();
            const int dim = this->tree.dimension(); 
            const int codim = dim - (nvertices - 1);  
            int ncofaces = 0; 

            // Run through the codimension-1 cofaces of the given simplex ...  
            for (auto& coface : this->tree.cofaces_simplex_range(this->tree.find(simplex), codim))
                ncofaces++; 

            return ncofaces; 
        }

        int getNumFullDimCofaces(const Simplex_handle& simplex) const
        {
            const int nvertices = this->tree.dimension(simplex) + 1;
            const int dim = this->tree.dimension(); 
            const int codim = dim - (nvertices - 1);  
            int ncofaces = 0; 

            // Run through the codimension-1 cofaces of the given simplex ... 
            for (auto& coface : this->tree.cofaces_simplex_range(simplex, codim))
                ncofaces++;

            return ncofaces; 
        }

        /**
         * TODO Is this necessary? 
         */
        Array<int, Dynamic, Dynamic> getFullDimCofaces(const Ref<const Array<int, Dynamic, 1> >& simplex,
                                                       const bool sort = true) const
        {
            const int nvertices = simplex.size();
            const int dim = this->tree.dimension(); 
            const int codim = dim - (nvertices - 1);  
            std::vector<Vertex_handle> simplex_; 
            for (int i = 0; i < nvertices; ++i)
                simplex_.push_back(simplex(i));
            std::vector<std::vector<int> > cofaces_; 

            // Run through the codimension-1 cofaces of the given simplex ...  
            for (auto& simplex : this->tree.cofaces_simplex_range(this->tree.find(simplex_), codim))
            {
                // ... and collect the vertices in the simplex
                std::vector<int> coface;  
                for (auto vertex : this->tree.simplex_vertex_range(simplex))
                    coface.push_back(vertex);

                // Sort the vertices
                std::sort(coface.begin(), coface.end()); 
                cofaces_.push_back(coface); 
            }

            // Return as an array 
            Array<int, Dynamic, Dynamic> cofaces(cofaces_.size(), dim + 1); 
            int i = 0; 
            for (auto it1 = cofaces_.begin(); it1 != cofaces_.end(); ++it1)
            {
                int j = 0; 
                for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
                {
                    cofaces(i, j) = *it2;
                    j++; 
                }
                i++; 
            }

            return cofaces; 
        }

        /**
         * Return the topological boundary of the simplicial complex.
         *
         * This is the subcomplex of faces that do not feature as a subface
         * of 2 or more full-dimensional faces.
         *
         * @returns The boundary simplicial complex.  
         */
        SimplicialComplex3D<T> getBoundary() const 
        {
            Array<int, Dynamic, 1> points_in_boundary = Array<int, Dynamic, 1>::Zero(this->points.rows());
            Simplex_tree tree_boundary;  

            // Run through all non-full-dimensional simplices in the tree ...
            for (auto& simplex : tree.skeleton_simplex_range(this->tree.dimension() - 1))
            {
                // ... and check the number of full-dimensional simplices that 
                // have this simplex as a face 
                int ncofaces = this->getNumFullDimCofaces(simplex);

                // If this number is 0 or 1, include as part of the boundary 
                if (ncofaces < 2)
                {
                    for (auto vertex : this->tree.simplex_vertex_range(simplex))
                        points_in_boundary(vertex) = 1;
                    tree_boundary.insert_simplex_and_subfaces(this->tree.simplex_vertex_range(simplex));
                } 
            } 
           
            // Get the subset of points that lie within the boundary 
            Array<T, Dynamic, 3> bpoints(points_in_boundary.sum(), 3);
            int i = 0;
            for (int j = 0; j < points.rows(); ++j)
            {
                if (points_in_boundary(j))
                {
                    bpoints.row(i) = points.row(j); 
                    i++;
                }
            } 

            return SimplicialComplex3D<T>(bpoints, tree_boundary); 
        }

        /**
         * Get the vector of Betti numbers for the simplicial complex. 
         *
         * @param p Characteristic of coefficient field. 
         * @returns Betti numbers of the simplicial complex. 
         */
        Array<int, Dynamic, 1> getBettiNumbers(const int p = 2)
        {
            // Compute the homology groups of the simplicial complex in the
            // given coefficient field  
            Persistent_cohomology pcoh(this->tree, true);
            pcoh.init_coefficients(p); 
            pcoh.compute_persistent_cohomology();

            // Collect the Betti numbers 
            Array<int, Dynamic, 1> betti(4);
            for (int i = 0; i < 4; ++i)
                betti(i) = pcoh.betti_number(i); 
            
            return betti;  
        }
};

#endif
