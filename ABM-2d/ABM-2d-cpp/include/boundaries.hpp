/**
 * Implementation of an alpha-shape-based algorithm for determining the 
 * boundary of a planar region approximated by a finite point-set.
 *
 * Adapted from github.com/kmnam/boundaries.git. 
 * 
 * Authors:
 *     Kee-Myoung Nam
 * 
 * Last updated:
 *     2/15/2023
 */

#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/algorithm.h>
#include <CGAL/exceptions.h>
#include <CGAL/tags.h>
#include <Eigen/Sparse>

using namespace Eigen;
const double TWO_PI = 2 * std::acos(-1);

typedef CGAL::Exact_predicates_inexact_constructions_kernel          K;
typedef K::FT                                                        FT;
typedef K::Point_2                                                   Point_2;
typedef K::Vector_2                                                  Vector_2;
typedef CGAL::Aff_transformation_2<K>                                Transformation;
typedef CGAL::Orientation                                            Orientation;
typedef CGAL::Polygon_2<K>                                           Polygon_2;
typedef CGAL::Polyline_simplification_2::Squared_distance_cost       Cost;
typedef CGAL::Polyline_simplification_2::Stop_below_count_threshold  Stop;

/**
 * Given three points `p`, `q`, and `r` and a choice of "outward side," return
 * the outward normal vector at the middle vertex, `q`.
 *
 * @param p First point.
 * @param q Second point.
 * @param r Third point.
 * @returns Outward normal vector at `q`.
 */
Vector_2 getOutwardVertexNormal(const double p_x, const double p_y, 
                                const double q_x, const double q_y,
                                const double r_x, const double r_y, 
                                const Orientation orientation)
{
    Vector_2 v, w, normal;

    // Get the vectors from q to p and from q to r 
    v = Vector_2(p_x - q_x, p_y - q_y); 
    w = Vector_2(r_x - q_x, r_y - q_y);

    // Get the angle between the two vectors
    double arg = CGAL::scalar_product(v, w) / std::sqrt(v.squared_length() * w.squared_length());
    if (arg > 1)         // Check that the argument passed to acos() is between -1 and 1
        arg = 1; 
    else if (arg < -1)
        arg = -1; 
    double angle = std::acos(arg);

    // Get the orientation of w with respect to -v (left or right turn)
    Orientation v_to_w = CGAL::orientation(-v, w);

    // Case 1: The outward side is on the left and -v and w form a right turn
    if (orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::RIGHT_TURN)
    {
        // Rotate w by (2*pi - angle) / 2 counterclockwise
        Transformation rotate(CGAL::ROTATION, std::sin((TWO_PI - angle) / 2.0), std::cos((TWO_PI - angle) / 2.0)); 
        normal = rotate(w);
    }
    // Case 2: The outward side is on the left and -v and w form a left turn
    else if (orientation == CGAL::RIGHT_TURN && v_to_w == CGAL::LEFT_TURN)
    {
        // Rotate w by angle / 2 counterclockwise
        Transformation rotate(CGAL::ROTATION, std::sin(angle / 2.0), std::cos(angle / 2.0));
        normal = rotate(w);
    }
    // Case 3: The outward side is on the right and -v and w form a right turn
    else if (orientation == CGAL::LEFT_TURN && v_to_w == CGAL::RIGHT_TURN)
    {
        // Rotate v by angle / 2 counterclockwise
        Transformation rotate(CGAL::ROTATION, std::sin(angle / 2.0), std::cos(angle / 2.0));
        normal = rotate(v);
    }
    // Case 4: The outward side is on the right and -v and w form a left turn
    else if (orientation == CGAL::LEFT_TURN && v_to_w == CGAL::LEFT_TURN)
    {
        // Rotate v by (2*pi - angle) / 2 counterclockwise
        Transformation rotate(CGAL::ROTATION, std::sin((TWO_PI - angle) / 2.0), std::cos((TWO_PI - angle) / 2.0));
        normal = rotate(v);
    }
    // Case 5: -v and w are collinear
    else
    {
        // If the outward side is on the left, rotate w by 90 degrees counterclockwise
        if (orientation == CGAL::RIGHT_TURN)
        {
            Transformation rotate(CGAL::ROTATION, 1.0, 0.0);
            normal = rotate(w);
        }
        // Otherwise, rotate v by 90 degrees counterclockwise
        else
        {
            Transformation rotate(CGAL::ROTATION, 1.0, 0.0);
            normal = rotate(v);
        }
    }

    // Normalize the vector by its length and return
    normal /= std::sqrt(normal.squared_length());
    return normal;
}

/**
 * A container that stores a planar point-set along with the indices of the 
 * points lying in an alpha shape of the point-set, together with the
 * corresponding value of alpha and the area of the enclosed region. 
 */
struct AlphaShape2DProperties
{
    private:
        /**
         * Given the indices of three *consecutive* vertices `p`, `q`, `r` in
         * the alpha shape, return the outward normal vector at the middle
         * vertex, `q`.
         *
         * `p`, `q`, `r` are assumed to be entries in `this->vertices`, and 
         * the pairs `(p, q)` and `(q, r)` are assumed to be entries in 
         * `this->edges`.
         *
         * This method is the private method used in `getOutwardVertexNormal(int)`,
         * which is public; the validity of `p`, `q`, `r` as indices of points 
         * in the alpha shape is not checked.  
         *
         * @param p Index of first vertex in alpha shape.
         * @param q Index of second vertex in alpha shape. 
         * @param r Index of third vertex in alpha shape. 
         * @returns Outward normal vector at `q`.
         */
        Vector_2 getOutwardVertexNormal(const int p, const int q, const int r)
        {
            return ::getOutwardVertexNormal(
                this->x[p], this->y[p], this->x[q], this->y[q], this->x[r], 
                this->y[r], this->orientation
            );  
        }

    public:
        /** x-coordinates of the points in the point-set. */
        std::vector<double> x;

        /** y-coordinates of the points in the point-set. */
        std::vector<double> y;

        /** Vertices in the alpha shape, indicated by their indices in `this->x`. */ 
        std::vector<int> vertices;

        /** Edges in the alpha shape. */ 
        std::vector<std::pair<int, int> > edges;
        
        /** Number of points in the point-set. */  
        int np;

        /** Number of vertices in the alpha shape. */ 
        int nv;

        /** Value of alpha. */ 
        double alpha;

        /** Area of the region enclosed by the alpha shape. */ 
        double area;

        /** Whether the alpha shape forms a simple cycle of edges. */  
        bool is_simple_cycle;

        /** Index of point with minimum y-coordinate. */  
        int min;

        /** Orientation of edges in the alpha shape. */ 
        Orientation orientation;

        /**
         * Trivial constructor. 
         */
        AlphaShape2DProperties()
        {
        }

        /**
         * Constructor with a non-empty point-set and alpha shape.
         *
         * Setting `is_simple_cycle = true` and `check_order = true` enforces
         * an explicit check that the boundary is a simple cycle and the
         * vertices and edges are specified "in order," as in `edges[0]` lies
         * between `vertices[0]` and `vertices[1]`, `edges[1]` between
         * `vertices[1]` and `vertices[2]`, and so on.
         *
         * @param x               x-coordinates of input point-set. 
         * @param y               y-coordinates of input point-set. 
         * @param vertices        Indices of vertices lying in input alpha shape.
         * @param edges           Pairs of indices of vertices connected by 
         *                        edges in input alpha shape.
         * @param alpha           Corresponding value of alpha. 
         * @param area            Area of region enclosed by the alpha shape. 
         * @param is_simple_cycle If true, the alpha shape consists of one 
         *                        simple cycle of edges.
         * @param check_order     If true (and `is_simple_cycle` is also true), 
         *                        this constructor checks whether the alpha
         *                        shape indeed consists of one simple cycle of 
         *                        edges, and whether the vertices and edges have
         *                        been specified in order, as described above.
         * @throws std::invalid_argument If `x`, `y`, and `vertices` do not all 
         *                               have the same size, or if `is_simple_cycle`
         *                               is true and yet `vertices` and `edges`
         *                               do not have the same size. 
         * @throws std::runtime_error    If `vertices` or `edges` contain invalid
         *                               vertices, or if `check_order` and
         *                               `is_simple_cycle` are true but the
         *                               vertices and edges do not form a simple
         *                               cycle or have not been specified in order. 
         */
        AlphaShape2DProperties(std::vector<double> x, std::vector<double> y,
                               std::vector<int> vertices,
                               std::vector<std::pair<int, int> > edges, 
                               double alpha, double area, bool is_simple_cycle,
                               bool check_order = false)
        {
            // The number of x- and y-coordinates should be the same  
            if (!(x.size() == y.size() && y.size() >= vertices.size()))
                throw std::invalid_argument("Invalid dimensions for input points");

            // Check that each vertex is valid
            this->np = x.size();
            this->nv = vertices.size();
            std::stringstream ss; 
            for (auto it = vertices.begin(); it != vertices.end(); ++it)
            {
                if (!(*it >= 0 && *it < this->np))
                {
                    ss << "Invalid vertex index specified: " << *it
                       << " (number of vertices = " << this->np << ")"; 
                    throw std::runtime_error(ss.str()); 
                }
            }
            for (auto it = edges.begin(); it != edges.end(); ++it)
            {
                if (!(it->first >= 0 && it->first < this->np))
                {
                    ss << "Invalid vertex index specified: " << it->first 
                       << " (number of vertices = " << this->np << ")";
                    throw std::runtime_error(ss.str()); 
                }
                if (!(it->second >= 0 && it->second < this->np))
                {
                    ss << "Invalid vertex index specified: " << it->second 
                       << " (number of vertices = " << this->np << ")";
                    throw std::runtime_error(ss.str()); 
                }
            }

            // If the boundary is assumed to be a simple cycle, then the 
            // number of vertices and edges should be the same
            if (is_simple_cycle && vertices.size() != edges.size())
                throw std::invalid_argument(
                    "Simple-cycle boundary should have the same number of vertices and edges"
                ); 
             
            this->x = x;
            this->y = y;
            this->vertices = vertices;
            this->edges = edges;
            this->alpha = alpha;
            this->area = area;
            this->is_simple_cycle = is_simple_cycle; 

            // Find the vertex with minimum y-coordinate, breaking any 
            // ties with whichever point has the smallest x-coordinate
            if (this->nv > 0)
            {
                this->min = 0;
                double xmin = this->x[this->vertices[0]];
                double ymin = this->y[this->vertices[0]];
                for (int i = 1; i < this->nv; ++i)
                {
                    if (this->y[this->vertices[i]] < ymin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                    else if (this->y[this->vertices[i]] == ymin && this->x[this->vertices[i]] < xmin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                }
            }
            else 
            {
                this->min = std::numeric_limits<double>::quiet_NaN();  
            }

            // Check that the edges and vertices were specified in order, 
            // given that the boundary is a simple cycle
            if (this->is_simple_cycle && check_order)
            {
                int i = 0;
                // Check the first vertex first 
                bool ordered = (vertices[0] == edges[0].first && vertices[0] == edges[this->nv-1].second);
                while (ordered && i < this->nv - 1)
                {
                    i++;
                    ordered = (vertices[i] == edges[i-1].second && vertices[i] == edges[i].first);
                }
                if (!ordered)
                    throw std::runtime_error(
                        "Vertices and edges were not specified in order in given simple-cycle boundary"
                    );
            }

            // Find the orientation of the edges (given there are at least
            // three vertices)
            if (this->is_simple_cycle && this->nv >= 3)
            {
                std::vector<Point_2> points_in_order;
                for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
                    points_in_order.emplace_back(Point_2(this->x[it->first], this->y[it->first])); 
                Polygon_2 polygon(points_in_order.begin(), points_in_order.end());
                if (polygon.orientation() == CGAL::CLOCKWISE)
                    this->orientation = CGAL::RIGHT_TURN;
                else
                    this->orientation = CGAL::LEFT_TURN; 
            }
            // If there are fewer than three vertices or the shape is not a
            // simple cycle, set to LEFT_TURN
            else 
            {
                this->orientation = CGAL::LEFT_TURN;
            }
        }
        
        /**
         * Trivial destructor. 
         */
        ~AlphaShape2DProperties()
        {
        }

        /**
         * Given the index `i` of a vertex *in `this->vertices`*, return the 
         * outward normal vector at `this->vertices[i]`.
         *
         * The boundary is assumed to a simple cycle. 
         *
         * @param i Index of first vertex in alpha shape *in `this->vertices`*.
         * @returns Outward normal vector at `this->vertices[i]`.
         * @throws std::invalid_argument If `i` is an invalid vertex index.
         * @throws std::runtime_error    If the boundary is not a simple cycle. 
         */
        Vector_2 getOutwardVertexNormal(const int i)
        {
            // Check that i is a valid vertex index 
            if (!(i >= 0 && i < this->nv))
            {
                std::stringstream ss;
                ss << "Invalid vertex index specified: " << i
                   << " (number of vertices = " << this->nv << ")";
                throw std::invalid_argument(ss.str());
            }

            // Check that the boundary is a simple cycle 
            if (!this->is_simple_cycle)
                throw std::runtime_error(
                    "Boundary is not simple cycle; outward normal vectors are not well-defined"
                );

            // Check that there are fewer than three vertices
            if (this->nv < 3)
                throw std::runtime_error(
                    "Outward normal vectors undefined for alpha shape with < 3 vertices"
                );  

            if (i == 0)
                return this->getOutwardVertexNormal(
                    this->vertices[this->nv - 1], this->vertices[0], this->vertices[1]
                );
            else if (i == this->nv - 1)
                return this->getOutwardVertexNormal(
                    this->vertices[this->nv - 2], this->vertices[this->nv - 1], this->vertices[0]
                );
            else 
                return this->getOutwardVertexNormal(
                    this->vertices[i - 1], this->vertices[i], this->vertices[i + 1]
                );
        }

        /**
         * Return the outward normal vectors from all vertices along the alpha
         * shape.
         *
         * @returns `std::vector` of outward normal vectors.
         * @throws std::runtime_error If `getOutwardVertexNormal()` throws such
         *                            an error.
         * @throws std::invalid_argument If `getOutwardVertexNormal()` throws
         *                               such an error. 
         */
        std::vector<Vector_2> getOutwardVertexNormals()
        {
            int p, q, r;
            std::vector<Vector_2> normals;
            Vector_2 normal; 

            // Obtain the outward normal vector at each vertex 
            for (int i = 0; i < this->nv; ++i)
            {
                try
                {
                    normal = this->getOutwardVertexNormal(i);
                }
                catch (const std::invalid_argument& e)
                {
                    throw; 
                }
                catch (const std::runtime_error& e)
                {
                    throw; 
                }
                normals.push_back(normal); 
            }

            return normals;
        }

        /**
         * Re-direct the edges (i.e., change the edge `(p, q)` to `(q, p)`) so
         * that every edge exhibits the given orientation.
         *
         * @param orientation            Desired orientation of edges.
         * @throws std::invalid_argument If the specified orientation is invalid 
         *                               (is not `CGAL::LEFT_TURN` or
         *                               `CGAL::RIGHT_TURN`).  
         */
        void orient(Orientation orientation)
        {
            if (orientation != CGAL::LEFT_TURN && orientation != CGAL::RIGHT_TURN)
                throw std::invalid_argument("Invalid orientation specified");

            // If the given orientation is the opposite of the current
            // orientation and there are three or more vertices ...
            if (orientation != this->orientation && this->nv >= 3)
            {
                std::vector<int> vertices;
                std::vector<std::pair<int, int> > edges; 

                vertices.push_back(this->vertices[0]);
                edges.push_back(std::make_pair(this->vertices[0], this->vertices[this->nv-1]));
                for (int i = this->nv - 1; i > 0; --i)
                {
                    vertices.push_back(this->vertices[i]);
                    edges.push_back(std::make_pair(this->vertices[i], this->vertices[i-1]));
                }
                this->vertices = vertices;
                this->edges = edges;
                this->orientation = orientation;
            }
        }

        /**
         * Delete the given points in the interior of the alpha shape.
         *
         * @param indices Indices of interior points to delete. 
         * @throws std::runtime_error If any specified index refers to a point 
         *                            that does not exist or lie in the interior,
         *                            or if any index was specified multiple times.
         */
        void deleteInteriorPoints(std::vector<int>& indices)
        {
            // Maintain a set of boundary point indices, to ensure that none 
            // of the points being deleted are boundary points 
            std::unordered_set<int> boundary_indices(
                this->vertices.begin(), this->vertices.end()
            ); 
            for (const int i : indices)
            {
                if (!(i >= 0 && i < this->np))
                    throw std::runtime_error(
                        "Invalid index specified for point to be removed"
                    );
                else if (boundary_indices.find(i) != boundary_indices.end())
                    throw std::runtime_error(
                        "Specified point lies in boundary, not in interior"
                    );  
            }

            // Maintain a vector of indices of points to be removed 
            std::vector<int> interior_indices(indices); 

            // Remove each point one by one ...  
            boost::random::uniform_01<double> dist;  
            for (int i = 0; i < interior_indices.size(); ++i)
            {
                std::vector<double>::iterator xit = this->x.begin() + interior_indices[i]; 
                this->x.erase(xit);
                std::vector<double>::iterator yit = this->y.begin() + interior_indices[i];
                this->y.erase(yit);
                this->np--;

                for (int j = 0; j < this->nv; ++j)
                {
                    // Note that this->vertices[j] should never equal index to remove
                    // (former is in boundary, latter is in interior)
                    if (this->vertices[j] > interior_indices[i])
                        this->vertices[j]--;
                }
                for (int j = 0; j < this->edges.size(); ++j)
                {
                    // Note that neither vertex of each edge should never equal index
                    // to remove (former is in boundary, latter is in interior)
                    if (this->edges[j].first > interior_indices[i])
                        this->edges[j].first--;
                    if (this->edges[j].second > interior_indices[i])
                        this->edges[j].second--;
                }
                for (int j = i + 1; j < interior_indices.size(); ++j)
                {
                    // Note that no other given interior point index (j) should 
                    // equal the current index (i), but this should be checked
                    if (interior_indices[j] == interior_indices[i])
                        throw std::runtime_error("Specified list of indices contains duplicates"); 
                    else if (interior_indices[j] > interior_indices[i])
                        interior_indices[j]--;
                }
            }

            // Find the vertex with minimum y-coordinate, breaking any 
            // ties with whichever point has the smallest x-coordinate
            if (this->nv > 0)
            {
                this->min = 0;
                double xmin = this->x[this->vertices[0]];
                double ymin = this->y[this->vertices[0]];
                for (int i = 1; i < this->nv; ++i)
                {
                    if (this->y[this->vertices[i]] < ymin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                    else if (this->y[this->vertices[i]] == ymin && this->x[this->vertices[i]] < xmin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                }
            }
            else 
            {
                this->min = std::numeric_limits<double>::quiet_NaN();  
            }
        }

        /**
         * Traverse the vertices in the stored alpha shape and determine 
         * whether they form a simple cycle. 
         *
         * @returns True the alpha shape is a simple cycle, false otherwise. 
         */
        bool isSimpleCycle()
        {
            // Return false if this->nv == 0, 1, or 2
            if (this->nv <= 2)
                return false; 

            // Re-index the vertices from 0 to this->nv - 1
            std::unordered_map<int, int> vertex_indices; 
            for (int i = 0; i < this->nv; ++i)
                vertex_indices[this->vertices[i]] = i;

            // Form a sparse adjacency matrix from the edges and check that 
            // each row (and thus each column) adds to 2
            SparseMatrix<int, RowMajor> adj(this->nv, this->nv);
            std::vector<Triplet<int> > triplets; 
            for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
            {
                int i = vertex_indices[it->first];
                int j = vertex_indices[it->second];
                triplets.emplace_back(Triplet<int>(i, j, 1));
                triplets.emplace_back(Triplet<int>(j, i, 1));  
            }
            adj.setFromTriplets(triplets.begin(), triplets.end());
            VectorXi degrees = adj * VectorXi::Ones(this->nv);
            for (int i = 0; i < this->nv; ++i)
            {
                if (degrees(i) != 2)
                    return false; 
            }
            
            // Now traverse each edge, starting from the zeroth vertex
            int i = 0; 
            VectorXi visited = VectorXi::Zero(this->nv);
            visited(i) = 1;
            bool returned = false;  
            while (!returned)
            {
                // Iterate over the nonzero entries in the current vertex's 
                // row -- there should *always* be two such entries 
                SparseMatrix<int, RowMajor>::InnerIterator row_it(adj, i);
                int first = row_it.col();
                ++row_it; 
                int second = row_it.col();
                ++row_it;

                // If both vertices have been visited *and* one of them is the zeroth
                // vertex, then we have returned
                if (visited(first) == 1 && visited(second) == 1 && (first == 0 || second == 0))
                {
                    returned = true;
                }
                // Otherwise, if both vertices have been visited, then the alpha shape
                // contains a more complicated structure
                else if (visited(first) == 1 && visited(second) == 1)
                {
                    return false;
                }
                // Otherwise, if only the first vertex has been visited, then jump to
                // the second vertex
                else if (visited(first) == 1 && visited(second) == 0)
                {
                    i = second;
                    visited(i) = 1; 
                } 
                // Otherwise, if only the second vertex has been visited, then jump to
                // the first vertex
                else if (visited(second) == 1 && visited(first) == 0)
                {
                    i = first;
                    visited(i) = 1; 
                }
                // Otherwise, if we are at the zeroth vertex (we are just starting our
                // traversal), then choose either vertex
                else if (visited(first) == 0 && visited(second) == 0 && i == 0)
                {
                    i = first;
                    visited(i) = 1;
                } 
                // Otherwise, if neither vertex has been visited, then there is
                // something wrong (this is supposed to be impossible)
                else 
                    throw std::runtime_error("This is not supposed to happen!");  
            }

            // Upon returning, finally check that every vertex was visited 
            return (returned && visited.sum() == this->nv); 
        }

        /**
         * Parse the data in a given boundary file, given in the same format 
         * as is written by `write()`:
         * - The first line contains the value of alpha. 
         * - The second line contains the area of the enclosed region.
         * - The next block of lines contains the coordinates of the points
         *   in the region.
         * - The next block of lines contains the indices of the vertices
         *   in the alpha shape.
         * - The final block of lines contains the indices of the endpoints
         *   of the edges in the alpha shape.
         *
         * This method overwrites all stored boundary data. 
         *
         * @param filename Input file name.
         * @param is_simple_cycle Whether or not the 
         * @throws std::runtime_error If the file is incorrectly formatted, 
         *                            or if any parsed vertex is invalid. 
         */
        void parse(std::string filename)
        {
            // Clear all stored boundary data
            this->np = 0; 
            this->nv = 0;
            this->x.clear(); 
            this->y.clear(); 
            this->vertices.clear(); 
            this->edges.clear();

            // Set alpha and area to infinity
            this->alpha = std::numeric_limits<double>::infinity(); 
            this->area = std::numeric_limits<double>::infinity();  

            // Parse each line in the file ... 
            std::ifstream infile; 
            infile.open(filename);
            std::string line, token;
            while (std::getline(infile, line))
            {
                std::vector<std::string> tokens;
                std::stringstream ss;
                ss << line; 

                // Each line in the file is tab-delimited 
                while (std::getline(ss, token, '\t'))
                    tokens.push_back(token);

                // Update the stored data according to the content in each line 
                if (tokens[0] == "ALPHA")
                {
                    if (tokens.size() != 2)
                        throw std::runtime_error("Unrecognized file format: anomalous ALPHA line"); 
                    this->alpha = std::stod(tokens[1]); 
                } 
                else if (tokens[0] == "AREA")
                {
                    if (tokens.size() != 2)
                        throw std::runtime_error("Unrecognized file format: anomalous AREA line"); 
                    this->area = std::stod(tokens[1]); 
                }
                else if (tokens[0] == "POINT")
                {
                    if (tokens.size() != 3)
                        throw std::runtime_error("Unrecognized file format: anomalous POINT line"); 
                    this->x.push_back(std::stod(tokens[1]));
                    this->y.push_back(std::stod(tokens[2])); 
                }
                else if (tokens[0] == "VERTEX")
                {
                    if (tokens.size() != 2)
                        throw std::runtime_error("Unrecognized file format: anomalous VERTEX line"); 
                    this->vertices.push_back(std::stoi(tokens[1]));
                }
                else if (tokens[0] == "EDGE")
                {
                    if (tokens.size() != 3)
                        throw std::runtime_error("Unrecognized file format: anomalous EDGE line"); 
                    this->edges.emplace_back(
                        std::make_pair(std::stoi(tokens[1]), std::stoi(tokens[2]))
                    ); 
                }
                else if (tokens[0] != "INPUT")    // The only other acceptable line heading is INPUT
                {
                    std::stringstream ss;
                    ss << "Unrecognized file heading: " << tokens[0];
                    throw std::runtime_error(ss.str()); 
                }
            }
            infile.close();
            this->np = this->x.size(); 
            this->nv = this->vertices.size();

            // Check that this->alpha and this->area have been specified 
            if (this->alpha == std::numeric_limits<double>::infinity())
                throw std::runtime_error("Input file does not specify alpha"); 
            if (this->area == std::numeric_limits<double>::infinity())
                throw std::runtime_error("Input file does not specify enclosed area");

            // Check that each vertex is valid
            std::stringstream ss; 
            for (auto it = this->vertices.begin(); it != this->vertices.end(); ++it)
            {
                if (!(*it >= 0 && *it < this->np))
                {
                    ss << "Invalid vertex index specified: " << *it
                       << " (number of vertices = " << this->np << ")"; 
                    throw std::runtime_error(ss.str()); 
                }
            }
            for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
            {
                if (!(it->first >= 0 && it->first < this->np))
                {
                    ss << "Invalid vertex index specified: " << it->first 
                       << " (number of vertices = " << this->np << ")";
                    throw std::runtime_error(ss.str()); 
                }
                if (!(it->second >= 0 && it->second < this->np))
                {
                    ss << "Invalid vertex index specified: " << it->second 
                       << " (number of vertices = " << this->np << ")";
                    throw std::runtime_error(ss.str()); 
                }
            }

            // Is the alpha shape a simple cycle? 
            this->is_simple_cycle = this->isSimpleCycle();

            // If it is a simple cycle, find the orientation by defining a 
            // polygon from the vertices 
            if (this->is_simple_cycle)
            {
                std::vector<Point_2> points_in_order;
                for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
                    points_in_order.emplace_back(Point_2(this->x[it->first], this->y[it->first])); 
                Polygon_2 polygon(points_in_order.begin(), points_in_order.end());
                if (polygon.orientation() == CGAL::CLOCKWISE)
                    this->orientation = CGAL::RIGHT_TURN;
                else
                    this->orientation = CGAL::LEFT_TURN; 
            } 
            // Otherwise, set it to LEFT_TURN by default 
            else 
            {
                this->orientation = CGAL::LEFT_TURN; 
            }

            // Find the vertex with minimum y-coordinate, breaking any 
            // ties with whichever point has the smallest x-coordinate
            if (this->nv > 0)
            {
                this->min = 0;
                double xmin = this->x[this->vertices[0]];
                double ymin = this->y[this->vertices[0]];
                for (int i = 1; i < this->nv; ++i)
                {
                    if (this->y[this->vertices[i]] < ymin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                    else if (this->y[this->vertices[i]] == ymin && this->x[this->vertices[i]] < xmin)
                    {
                        this->min = i;
                        xmin = this->x[this->vertices[i]];
                        ymin = this->y[this->vertices[i]];
                    }
                }
            }
            else 
            {
                this->min = std::numeric_limits<double>::quiet_NaN();  
            }
        }

        /**
         * Write the boundary data to an output file with the given name, as
         * follows: 
         * - The first line contains the value of alpha. 
         * - The second line contains the area of the enclosed region.
         * - The next block of lines contains the coordinates of the points
         *   in the region.
         * - The next block of lines contains the indices of the vertices
         *   in the alpha shape.
         * - The final block of lines contains the indices of the endpoints
         *   of the edges in the alpha shape.
         *
         * @param filename Output file name. 
         */
        void write(std::string filename)
        {
            std::ofstream outfile;
            outfile.open(filename);
            outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);

            // Write the value of alpha and the enclosed area 
            outfile << "ALPHA\t" << this->alpha << std::endl;
            outfile << "AREA\t" << this->area << std::endl;

            // Write each point in the full point-set 
            for (int i = 0; i < this->np; ++i)
                outfile << "POINT\t" << this->x[i] << "\t" << this->y[i] << std::endl;

            // Write each vertex and edge in the alpha shape 
            for (auto&& v : this->vertices)
                outfile << "VERTEX\t" << v << std::endl;
            for (auto&& e : this->edges)
                outfile << "EDGE\t" << e.first << "\t" << e.second << std::endl;

            outfile.close();
        }
};

/**
 * A class that, given a set of points in the plane, computes a suitable
 * "boundary" from its family of alpha shapes.  
 */
class Boundary2D 
{
    private:
        std::vector<double> x;    // x-coordinates
        std::vector<double> y;    // y-coordinates
        int n;                    // Number of stored points

        /**
         * Given a pre-computed alpha shape on the stored point-set, traverse 
         * the vertices in the alpha shape and determine whether they form a
         * simple cycle.
         *
         * Note that, if this method returns false as its second argument, then 
         * the entries in `traversal`, `vertices_to_indices`, `indices_to_vertices`,
         * `vertices_to_points` *should not* be trusted upon exiting the method.
         * On the other hand, `adj` is always updated completely and correctly,
         * regardless of whether the method returns true or false.  
         *
         * @param shape               Reference to pre-computed input `Alpha_shape_2`
         *                            instance.
         * @param points              Reference to `std::vector` of `Point_2` 
         *                            instances for all points in the alpha shape.  
         * @param traversal           Reference to `std::vector` to be updated
         *                            with the vertices of the alpha shape in
         *                            the traversed order. The updated contents
         *                            are to be trusted if the method returns
         *                            true as its second argument.  
         * @param vertices_to_indices Reference to `std::unordered_map` to be
         *                            updated with an index assignment to
         *                            vertices in the alpha shape. The updated 
         *                            contents are to be trusted if the method 
         *                            returns true as its second argument. 
         * @param indices_to_vertices Reference to `std::vector` to be updated 
         *                            with the same index assignment to vertices
         *                            in the alpha shape as in `vertices_to_indices`.
         *                            The updated contents are to be trusted if 
         *                            the method returns true as its second 
         *                            argument. 
         * @param vertices_to_points  Reference to `std::unordered_map` to be
         *                            updated with the nearest point to each
         *                            vertex in the alpha shape. The updated 
         *                            contents are to be trusted if the method
         *                            returns true as its second argument.  
         * @param adj                 Reference to adjacency matrix to be updated.
         * @param alpha_index         Index of the value of alpha of interest.
         * @param verbose             If true, output intermittent messages to
         *                            `stdout`.
         * @returns                   The number of vertices in the alpha shape, 
         *                            along with a boolean value that indicates 
         *                            whether the alpha shape is a simple cycle.  
         */
        template <typename Dt, typename ExactAlphaComparisonTag>
        bool traverseSimpleCycle(CGAL::Alpha_shape_2<Dt, ExactAlphaComparisonTag>& shape, 
                                 std::vector<Point_2>& points, 
                                 std::vector<int>& traversal,
                                 std::unordered_map<typename Dt::Vertex_handle, int>& vertices_to_indices,
                                 std::vector<typename Dt::Vertex_handle>& indices_to_vertices,
                                 std::unordered_map<typename Dt::Vertex_handle, std::pair<int, double> >& vertices_to_points, 
                                 SparseMatrix<int, RowMajor>& adj, 
                                 const int alpha_index, const bool verbose = false)
        {
            traversal.clear();
            vertices_to_indices.clear();
            indices_to_vertices.clear();
            vertices_to_points.clear(); 

            // Obtain the alpha shape for the given value of alpha  
            auto alpha = shape.get_nth_alpha(alpha_index);  
            shape.set_alpha(alpha);
            if (verbose)
                std::cout << "- Setting alpha = " << CGAL::to_double(alpha) << std::endl; 

            // Establish an ordering for the vertices in the alpha shape
            int nvertices = 0; 
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                // Find the point being pointed to by the vertex handle
                Point_2 point = (*it)->point();
                double xit = point.x(); 
                double yit = point.y(); 
                
                // Find the nearest point in the point-set to this vertex
                double nearest_sqdist = std::numeric_limits<double>::infinity(); 
                int nearest_index = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    double sqdist = std::pow(this->x[i] - xit, 2) + std::pow(this->y[i] - yit, 2);
                    if (sqdist < nearest_sqdist)
                    {
                        nearest_sqdist = sqdist;
                        nearest_index = i; 
                    }
                }
                vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
                vertices_to_indices[*it] = nvertices; 
                indices_to_vertices.push_back(*it); 
                nvertices++; 
            }

            // Check that there are at least three vertices
            if (nvertices < 3)
            {
                if (verbose)
                {
                    std::cout << "- Boundary is not simple cycle: contains < 3 vertices"
                              << std::endl;
                } 
                return false;
            } 
           
            // Iterate through the edges in the alpha shape and fill in the adjacency matrix 
            adj.resize(nvertices, nvertices); 
            std::vector<Triplet<int> > triplets;
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                // Add the two corresponding entries to the adjacency matrix 
                int j = it->second; 
                typename Dt::Vertex_handle source = (it->first)->vertex(Dt::cw(j));
                typename Dt::Vertex_handle target = (it->first)->vertex(Dt::ccw(j));
                int source_i = vertices_to_indices[source]; 
                int target_i = vertices_to_indices[target]; 
                triplets.emplace_back(Triplet<int>(source_i, target_i, 1)); 
                triplets.emplace_back(Triplet<int>(target_i, source_i, 1)); 
            }
            adj.setFromTriplets(triplets.begin(), triplets.end());

            // Check that every vertex has degree 2
            VectorXi ones = VectorXi::Ones(nvertices);
            VectorXi degrees = adj * ones;
            for (int i = 0; i < nvertices; ++i)
            {
                if (degrees(i) != 2)
                {
                    if (verbose)
                    {
                        std::cout << "- Boundary is not simple cycle: is not 2-regular"
                                  << std::endl;
                    } 
                    return false;
                } 
            }
            int nedges = (adj.triangularView<Upper>() * ones).sum(); 

            // Starting from an arbitrary vertex, identify the incident edges on the
            // vertex and "travel" along the boundary, checking that:
            // 1) Each vertex has exactly two incident edges
            // 2) Each vertex has one unvisited neighbor and one visited neighbor
            //    (except at the start and end of the traversal)
            // 3) Iteratively choosing the visited neighbor returns us to the
            //    starting vertex after *every* vertex has been visited
            // 
            // Start from the zeroth vertex ...
            int i = 0;
            VectorXi visited = VectorXi::Zero(nvertices);
            visited(i) = 1;
            int nvisited = 1;  
            traversal.push_back(i); 
            bool returned = false;  
            while (!returned)
            {
                // Iterate over the nonzero entries in the current vertex's 
                // row -- there should *always* be two such entries 
                SparseMatrix<int, RowMajor>::InnerIterator row_it(adj, i);
                int first = row_it.col();
                ++row_it; 
                int second = row_it.col();
                ++row_it;

                // If both vertices have been visited *and* one of them is the zeroth
                // vertex, then we have returned
                if (visited(first) == 1 && visited(second) == 1 && (first == 0 || second == 0))
                {
                    returned = true;
                }
                // Otherwise, if both vertices have been visited, then the alpha shape
                // contains a more complicated structure
                else if (visited(first) == 1 && visited(second) == 1)
                {
                    if (verbose)
                    {
                        std::cout << "- Boundary is not simple cycle: contains non-trivial sub-cycle"
                                  << std::endl;
                    }
                    return false; 
                }
                // Otherwise, if only the first vertex has been visited, then jump to
                // the second vertex
                else if (visited(first) == 1 && visited(second) == 0)
                {
                    i = second;
                    visited(i) = 1;
                    nvisited++; 
                    traversal.push_back(i);  
                } 
                // Otherwise, if only the second vertex has been visited, then jump to
                // the first vertex
                else if (visited(second) == 1 && visited(first) == 0)
                {
                    i = first;
                    visited(i) = 1;
                    nvisited++; 
                    traversal.push_back(i); 
                }
                // Otherwise, if we are at the zeroth vertex (we are just starting our
                // traversal), then choose either vertex
                else if (visited(first) == 0 && visited(second) == 0 && i == 0)
                {
                    i = first;
                    visited(i) = 1;
                    nvisited++; 
                    traversal.push_back(i); 
                } 
                // Otherwise, if neither vertex has been visited, then there is
                // something wrong (this is supposed to be impossible)
                else 
                    throw std::runtime_error("This is not supposed to happen!");  
            }

            // Have we traversed the entire alpha shape and returned to the starting 
            // vertex, and does every vertex lie either along or within the simple cycle?
            if (returned && nvisited == nvertices)
            {
                if (verbose)
                { 
                    std::cout << "- Traversed " << visited.sum() << "/" << nvertices
                              << " boundary vertices in a simple cycle" << std::endl;
                }
                return true; 
            }
            else
            {
                if (verbose)
                {
                    std::cout << "- Traversed " << visited.sum() << "/" << nvertices 
                              << " boundary vertices; boundary contains "
                              << nedges << " edges (not simple)" << std::endl;
                }
                return false;
            }
        }

    public:
        /**
         * Trivial constructor. 
         */
        Boundary2D()
        {
        }

        /** 
         * Constructor with a non-empty input point-set.
         *
         * @param x x-coordinates of input point-set. 
         * @param y y-coordinates of input point-set.
         * @throws std::invalid_argument If there are different numbers of x- 
         *                               and y-coordinates specified.  
         */
        Boundary2D(std::vector<double> x, std::vector<double> y)
        {
            // Check that x and y have the same number of coordinates
            this->n = x.size(); 
            if (this->n != y.size())
            {
                throw std::invalid_argument(
                    "Invalid input point-set (different numbers of x- and y-coordinates)"
                );
            }
            this->x = x;
            this->y = y;
        }

        /**
         * Trivial destructor. 
         */
        ~Boundary2D()
        {
        }

        /**
         * Parse a set of points in the given delimited .txt file and either 
         * (1) append the parsed points to the stored point-set (`clear = false`)
         * or (2) overwrite the stored point-set with the parsed points
         * (`clear = true`). 
         *
         * @param filename Input file name. 
         * @param delim    Delimiter in the input file.
         * @param clear    If true, overwrite all previously stored points with
         *                 the parsed points; if false, append the parsed points
         *                 to the previously stored points.  
         */
        void fromFile(std::string filename, char delim = '\t', bool clear = true)
        {
            // Check that a file exists at the given path
            std::ifstream f(filename);
            if (!f.good())
                throw std::invalid_argument("Specified input file does not exist");

            // Clear all stored points if desired
            if (clear)
            {
                this->x.clear();
                this->y.clear();
            }

            // Parse one line at a time
            std::string line;
            while (std::getline(f, line))
            {
                std::istringstream iss(line);
                double x_, y_;
                char c;
                if (!(iss >> x_ >> c >> y_ && c == delim)) break;
                this->x.push_back(x_);
                this->y.push_back(y_);
            }
        }

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, with no topological 
         * assumption regarding the enclosed region approximated by the point-set.
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the *smallest* value of alpha such that every point in the 
         * the point-set falls within the boundary or its interior.
         *
         * @param verbose If true, output intermittent messages to `stdout`.
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getBoundary(const bool verbose = false) 
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            // Print warning if tag == false 
            if (!tag)
                std::cout << "[WARN] Computing alpha shape with tag == false\n";

            // Instantiate a vector of Point objects
            std::vector<Point_2> points;
            for (int i = 0; i < this->n; ++i)
            {
                double xi = this->x[i]; 
                double yi = this->y[i]; 
                points.emplace_back(Point_2(xi, yi)); 
            }

            // Compute the alpha shape from the Delaunay triangulation
            Alpha_shape shape; 
            try
            {
                shape.make_alpha_shape(points.begin(), points.end());
            }
            catch (CGAL::Assertion_exception& e)
            {
                throw; 
            }
            shape.set_mode(Alpha_shape::REGULARIZED); 

            // Establish an ordering of the vertices in the alpha shape 
            // (this varies as we change alpha) 
            std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
            std::vector<Vertex_handle_2> indices_to_vertices;
            int nvertices = 0;

            // Set up a sparse adjacency matrix for the alpha shape 
            // (this varies as we change alpha) 
            SparseMatrix<int, RowMajor> adj;
            int nedges = 0;

            // Set up a dictionary that maps each vertex in the alpha shape 
            // to the corresponding point in the point-set
            std::unordered_map<Vertex_handle_2, std::pair<int, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the least value of alpha such that each point in the
             * point-set lies either along the boundary or within the boundary 
             * ------------------------------------------------------------------ */
            double opt_alpha = 0.0;
            int opt_alpha_index = 0;

            // Begin with the smallest value of alpha 
            auto alpha_it = shape.alpha_begin();
            int i = 0; 

            // For each larger value of alpha, test that every vertex in the 
            // point-set lies along either along the alpha shape or within 
            // its interior  
            for (auto it = alpha_it; it != shape.alpha_end(); it++)
            {
                shape.set_alpha(*it);

                // Run through the vertices in the underlying Delaunay
                // triangulation, and classify each according to its
                // relationship with the alpha shape
                bool regular = true;
                for (auto itv = shape.finite_vertices_begin(); itv != shape.finite_vertices_end(); itv++)
                {
                    Vertex_handle_2 v = Tds::Vertex_range::s_iterator_to(*itv);
                    auto type = shape.classify(v);
                    if (type != Alpha_shape::REGULAR && type != Alpha_shape::INTERIOR)
                    {
                        regular = false;
                        break;
                    }
                }
                if (regular)
                {
                    opt_alpha = CGAL::to_double(*it);
                    opt_alpha_index = i;
                    break;
                }

                i++; 
            }
            shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

            // Establish an ordering for the vertices in the alpha shape
            nvertices = 0; 
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                // Find the point being pointed to by the vertex handle
                Point_2 point = (*it)->point();
                double xit = point.x(); 
                double yit = point.y(); 
                
                // Find the nearest input point to this vertex 
                double nearest_sqdist = std::numeric_limits<double>::infinity(); 
                int nearest_index = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    double sqdist = std::pow(this->x[i] - xit, 2) + std::pow(this->y[i] - yit, 2);
                    if (sqdist < nearest_sqdist)
                    {
                        nearest_sqdist = sqdist;
                        nearest_index = i; 
                    }
                }
                vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
                vertices_to_indices[*it] = nvertices;
                indices_to_vertices.push_back(*it); 
                nvertices++;
            }

            // Iterate through the edges in the alpha shape and fill in
            // the adjacency matrix 
            adj.resize(nvertices, nvertices); 
            std::vector<Triplet<int> > triplets;
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                // Add the two corresponding entries to the adjacency 
                // matrix 
                int j = it->second; 
                Vertex_handle_2 source = (it->first)->vertex(Delaunay_triangulation_2::cw(j));
                Vertex_handle_2 target = (it->first)->vertex(Delaunay_triangulation_2::ccw(j));
                int source_i = vertices_to_indices[source]; 
                int target_i = vertices_to_indices[target]; 
                triplets.emplace_back(Triplet<int>(source_i, target_i, 1)); 
                triplets.emplace_back(Triplet<int>(target_i, source_i, 1)); 
            }
            adj.setFromTriplets(triplets.begin(), triplets.end());

            // Identify, for each vertex in the alpha shape, the point corresponding to it 
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                // Find the point being pointed to by the boundary vertex 
                Point_2 point = (*it)->point();
                double xit = point.x(); 
                double yit = point.y(); 
                
                // Find the nearest input point to this point
                double nearest_sqdist = std::numeric_limits<double>::infinity(); 
                int nearest_index = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    double sqdist = std::pow(this->x[i] - xit, 2) + std::pow(this->y[i] - yit, 2);
                    if (sqdist < nearest_sqdist)
                    {
                        nearest_sqdist = sqdist;
                        nearest_index = i; 
                    }
                }
                vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
            }

            // Count the edges in the alpha shape
            VectorXi ones = VectorXi::Ones(nvertices); 
            nedges = (adj.triangularView<Upper>() * ones).sum(); 

            // Get the area of the region enclosed by the alpha shape 
            // with the optimum value of alpha
            double total_area = 0.0;
            for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
            {
                Face_handle_2 face = Tds::Face_range::s_iterator_to(*it);
                auto type = shape.classify(face);
                if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
                {
                    Point_2 p = it->vertex(0)->point();
                    Point_2 q = it->vertex(1)->point();
                    Point_2 r = it->vertex(2)->point();
                    total_area += CGAL::area(p, q, r);
                }
            }
            if (verbose)
            {
                std::cout << "- Optimal value of alpha = " << opt_alpha << std::endl
                          << "- Enclosed area = " << total_area << std::endl
                          << "- Number of vertices = " << nvertices << std::endl
                          << "- Number of edges = " << nedges << std::endl;
            } 

            // Finally collect the boundary vertices and edges in arbitrary order 
            std::vector<std::pair<int, int> > edges; 
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                Face_handle_2 f = it->first;
                int i = it->second;
                Vertex_handle_2 s = f->vertex(f->cw(i));
                Vertex_handle_2 t = f->vertex(f->ccw(i));
                edges.emplace_back(std::make_pair(vertices_to_points[s].first, vertices_to_points[t].first));
            }
            std::vector<int> vertices;
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
                vertices.push_back(vertices_to_points[*it].first);

            return AlphaShape2DProperties(
                this->x, this->y, vertices, edges, opt_alpha, total_area, false
            );
        }

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, assuming that the 
         * enclosed region approximated by the point-set is connected. 
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the *smallest* value of alpha such that the boundary encloses
         * a connected region.
         *
         * @param verbose If true, output intermittent messages to `stdout`.
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getConnectedBoundary(const bool verbose = false)
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            // Print warning if tag == false 
            if (!tag)
                std::cout << "[WARN] Computing alpha shape with tag == false\n";

            // Instantiate a vector of Point objects
            std::vector<Point_2> points;
            for (int i = 0; i < this->n; ++i)
            {
                double xi = this->x[i]; 
                double yi = this->y[i]; 
                points.emplace_back(Point_2(xi, yi)); 
            }

            // Compute the alpha shape from the Delaunay triangulation
            Alpha_shape shape; 
            try
            {
                shape.make_alpha_shape(points.begin(), points.end());
            }
            catch (CGAL::Assertion_exception& e)
            {
                throw; 
            }
            shape.set_mode(Alpha_shape::REGULARIZED); 

            // Establish an ordering of the vertices in the alpha shape 
            // (this varies as we change alpha) 
            std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
            std::vector<Vertex_handle_2> indices_to_vertices;
            int nvertices = 0;

            // Set up a sparse adjacency matrix for the alpha shape 
            // (this varies as we change alpha) 
            SparseMatrix<int, RowMajor> adj;
            int nedges = 0;

            // Set up a dictionary that maps each vertex in the alpha shape 
            // to the corresponding point in the point-set 
            std::unordered_map<Vertex_handle_2, std::pair<int, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the least value of alpha such that the boundary encloses
             * a connected region (i.e., the alpha complex consists of a single 
             * connected component) 
             * ------------------------------------------------------------------ */
            int opt_alpha_index = std::distance(shape.alpha_begin(), shape.find_optimal_alpha(1));
            double opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
            shape.set_alpha(shape.get_nth_alpha(opt_alpha_index));

            // Establish an ordering for the vertices in the alpha shape
            nvertices = 0; 
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
            {
                // Find the point being pointed to by the vertex handle
                Point_2 point = (*it)->point();
                double xit = point.x(); 
                double yit = point.y(); 
                
                // Find the nearest point in the point-set to this vertex
                double nearest_sqdist = std::numeric_limits<double>::infinity(); 
                int nearest_index = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    double sqdist = std::pow(this->x[i] - xit, 2) + std::pow(this->y[i] - yit, 2);
                    if (sqdist < nearest_sqdist)
                    {
                        nearest_sqdist = sqdist;
                        nearest_index = i; 
                    }
                }
                vertices_to_points[*it] = std::make_pair(nearest_index, nearest_sqdist);
                vertices_to_indices[*it] = nvertices;
                indices_to_vertices.push_back(*it); 
                nvertices++;
            }

            // Iterate through the edges in the alpha shape and fill in
            // the adjacency matrix 
            adj.resize(nvertices, nvertices); 
            std::vector<Triplet<int> > triplets;
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                // Add the two corresponding entries to the adjacency 
                // matrix 
                int j = it->second; 
                Vertex_handle_2 source = (it->first)->vertex(Delaunay_triangulation_2::cw(j));
                Vertex_handle_2 target = (it->first)->vertex(Delaunay_triangulation_2::ccw(j));
                int source_i = vertices_to_indices[source]; 
                int target_i = vertices_to_indices[target]; 
                triplets.emplace_back(Triplet<int>(source_i, target_i, 1)); 
                triplets.emplace_back(Triplet<int>(target_i, source_i, 1)); 
            }
            adj.setFromTriplets(triplets.begin(), triplets.end());

            // Count the edges in the alpha shape
            VectorXi ones = VectorXi::Ones(nvertices); 
            nedges = (adj.triangularView<Upper>() * ones).sum();

            // Get the area of the region enclosed by the alpha shape 
            // with the optimum value of alpha
            double total_area = 0.0;
            for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
            {
                Face_handle_2 face = Tds::Face_range::s_iterator_to(*it);
                auto type = shape.classify(face);
                if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
                {
                    Point_2 p = it->vertex(0)->point();
                    Point_2 q = it->vertex(1)->point();
                    Point_2 r = it->vertex(2)->point();
                    total_area += CGAL::area(p, q, r);
                }
            }
            if (verbose)
            {
                std::cout << "- Optimal value of alpha = " << opt_alpha << std::endl
                          << "- Enclosed area = " << total_area << std::endl
                          << "- Number of vertices = " << nvertices << std::endl
                          << "- Number of edges = " << nedges << std::endl;
            } 

            // Finally collect the boundary vertices and edges in arbitrary order 
            std::vector<std::pair<int, int> > edges; 
            for (auto it = shape.alpha_shape_edges_begin(); it != shape.alpha_shape_edges_end(); ++it)
            {
                Face_handle_2 f = it->first;
                int i = it->second;
                Vertex_handle_2 s = f->vertex(f->cw(i));
                Vertex_handle_2 t = f->vertex(f->ccw(i));
                edges.emplace_back(std::make_pair(vertices_to_points[s].first, vertices_to_points[t].first));
            }
            std::vector<int> vertices;
            for (auto it = shape.alpha_shape_vertices_begin(); it != shape.alpha_shape_vertices_end(); ++it)
                vertices.push_back(vertices_to_points[*it].first);

            return AlphaShape2DProperties(
                this->x, this->y, vertices, edges, opt_alpha, total_area, false
            );
        }

        /**
         * Identify a subset of vertices in the point-set that form a boundary
         * for the point-set by computing an alpha shape, assuming that the 
         * enclosed region approximated by the point-set is simply connected.  
         *
         * This method returns an `AlphaShape2DProperties` object containing
         * the indices of the points lying along the identified boundary
         * with the given percentile value of alpha chosen among the values 
         * for which the boundary forms a simple cycle (i.e., the enclosed
         * region is simply connected).
         *
         * @param alpha_percentile  Percentile for value of alpha chosen to 
         *                          define the boundary. 
         * @param verbose           If true, output intermittent messages to
         *                          `stdout`.
         * @param traversal_verbose If true, output intermittent messages to
         *                          `stdout` from `traverseSimpleCycle()`. 
         * @returns `AlphaShape2DProperties` object containing the alpha shape 
         *          representing the boundary of the point-set.  
         */
        template <bool tag = true>
        AlphaShape2DProperties getSimplyConnectedBoundary(const double alpha_percentile = 1.0,
                                                          const bool verbose = false,
                                                          const bool traversal_verbose = false)
        {
            typedef CGAL::Alpha_shape_vertex_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> > Vb;
            typedef CGAL::Alpha_shape_face_base_2<K, CGAL::Default, CGAL::Boolean_tag<tag> >   Fb;
            typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                               Tds;
            typedef CGAL::Delaunay_triangulation_2<K, Tds>                                     Delaunay_triangulation_2;
            typedef CGAL::Alpha_shape_2<Delaunay_triangulation_2, CGAL::Boolean_tag<tag> >     Alpha_shape;
            typedef typename Delaunay_triangulation_2::Face_handle                             Face_handle_2;
            typedef typename Delaunay_triangulation_2::Vertex_handle                           Vertex_handle_2;

            // Print warning if tag == false 
            if (!tag)
                std::cout << "[WARN] Computing alpha shape with tag == false\n";

            // Instantiate a vector of Point objects
            std::vector<Point_2> points;
            for (int i = 0; i < this->n; ++i)
            {
                double xi = this->x[i]; 
                double yi = this->y[i]; 
                points.emplace_back(Point_2(xi, yi)); 
            }

            // Compute the alpha shape from the Delaunay triangulation
            Alpha_shape shape; 
            try
            {
                shape.make_alpha_shape(points.begin(), points.end());
            }
            catch (CGAL::Assertion_exception& e)
            {
                throw; 
            }
            shape.set_mode(Alpha_shape::REGULARIZED); 

            // Establish an ordering of the vertices in the alpha shape 
            // (this varies as we change alpha) 
            std::unordered_map<Vertex_handle_2, int> vertices_to_indices;
            std::vector<Vertex_handle_2> indices_to_vertices;
            int nvertices = 0;

            // Set up a sparse adjacency matrix for the alpha shape 
            // (this varies as we change alpha) 
            SparseMatrix<int, RowMajor> adj;
            int nedges = 0;

            // Set up a dictionary that maps each vertex in the alpha shape 
            // to the corresponding point in the point-set 
            std::unordered_map<Vertex_handle_2, std::pair<int, double> > vertices_to_points;

            /* ------------------------------------------------------------------ //
             * Determine the optimal value of alpha such that the boundary encloses
             * a simply connected region (i.e., the boundary consists of one 
             * simple cycle of vertices) 
             * ------------------------------------------------------------------ */
            double opt_alpha = 0.0;
            int opt_alpha_index = 0;
            const int INVALID_ALPHA_INDEX = shape.number_of_alphas();

            // Set the lower value of alpha for which the region is connected
            int low = 0; 
            int last_valid = INVALID_ALPHA_INDEX;

            // Try to find the smallest value of alpha for which the boundary
            // consists of one connected component
            //
            // Alpha_shape_2::find_optimal_alpha() can throw an Assertion_exception
            // if there are collinear points within the alpha shape
            try
            {
                low = std::max(
                    0,
                    static_cast<int>(std::distance(shape.alpha_begin(), shape.find_optimal_alpha(1)) - 1)
                );
            } 
            catch (CGAL::Assertion_exception& e)
            {
                low = 0;
            }
            int high = shape.number_of_alphas() - 1;
            if (verbose)
            { 
                std::cout << "- Searching between alpha = "
                          << CGAL::to_double(shape.get_nth_alpha(low))
                          << " and "
                          << CGAL::to_double(shape.get_nth_alpha(high))
                          << " inclusive ("
                          << high - low + 1
                          << " values of alpha) ..." << std::endl;
            }

            // Also keep track of the vertices and edges in the order in which 
            // they are traversed 
            std::vector<int> traversal;

            // For each larger value of alpha, test that the boundary is 
            // a simple cycle (every vertex has only two incident edges,
            // and traveling along the cycle in one direction gets us 
            // back to the starting point)
            bool is_simple_cycle = false; 
            bool found_simple_cycle_alpha = false; 
            for (int mid = low; mid <= high; ++mid)
            {
                is_simple_cycle = this->traverseSimpleCycle(
                    shape, points, traversal, vertices_to_indices, indices_to_vertices,
                    vertices_to_points, adj, mid, traversal_verbose
                );
                nvertices = std::distance(shape.alpha_shape_vertices_begin(), shape.alpha_shape_vertices_end()); 
                if (is_simple_cycle)
                { 
                    last_valid = mid; 
                    break; 
                }
            }
            
            // Have we found a value of alpha for which the boundary is a
            // simple cycle? 
            found_simple_cycle_alpha = (last_valid != INVALID_ALPHA_INDEX);

            // If so, then set the optimal value of alpha to slightly higher 
            // than this smallest value (the given percentile of the distribution
            // of alphas between this smallest value and the largest possible value)
            if (!found_simple_cycle_alpha)
                throw std::runtime_error("Could not find any simple-cycle boundary"); 
            if (verbose)
            {
                std::cout << "- Least valid value of alpha = "
                          << CGAL::to_double(shape.get_nth_alpha(last_valid)) 
                          << " (index " << last_valid << ", "
                          << nvertices << " vertices)" << std::endl;
                std::cout << "- Number of valid values for alpha = "
                          << shape.number_of_alphas() - last_valid << std::endl; 
            }
            int padding = static_cast<int>(std::floor((alpha_percentile / 100) * (shape.number_of_alphas() - last_valid)));
            opt_alpha_index = last_valid + padding;
            opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
            is_simple_cycle = this->traverseSimpleCycle(
                shape, points, traversal, vertices_to_indices, indices_to_vertices,
                vertices_to_points, adj, opt_alpha_index, traversal_verbose
            );
            while (!is_simple_cycle)
            {
                padding++;
                opt_alpha_index = last_valid + padding;
                opt_alpha = CGAL::to_double(shape.get_nth_alpha(opt_alpha_index));
                is_simple_cycle = this->traverseSimpleCycle(
                    shape, points, traversal, vertices_to_indices, indices_to_vertices,
                    vertices_to_points, adj, opt_alpha_index, traversal_verbose
                );
            }
            nvertices = std::distance(shape.alpha_shape_vertices_begin(), shape.alpha_shape_vertices_end());
            if (verbose)
            {
                std::cout << "- Chosen value of alpha = "
                          << CGAL::to_double(shape.get_nth_alpha(opt_alpha_index)) 
                          << " (index " << opt_alpha_index << ", "
                          << nvertices << " vertices)" << std::endl;
            }

            // Count the edges in the alpha shape
            VectorXi ones = VectorXi::Ones(nvertices); 
            nedges = (adj.triangularView<Upper>() * ones).sum();

            // Define vectors that specify the boundary vertices in the 
            // order in which they were traversed, *in terms of their 
            // indices in the point-set*, as well as the edges in the 
            // order in which they were traversed
            std::vector<int> vertex_indices_in_order; 
            std::vector<std::pair<int, int> > edge_indices_in_order; 

            // Accumulate the indices of the boundary vertices in the
            // order in which they were traversed
            auto it = traversal.begin();
            int curr = vertices_to_points[indices_to_vertices[*it]].first;
            vertex_indices_in_order.push_back(curr);
            ++it;
            while (it != traversal.end())
            {
                int next = vertices_to_points[indices_to_vertices[*it]].first;
                edge_indices_in_order.emplace_back(std::make_pair(curr, next));
                curr = next;  
                vertex_indices_in_order.push_back(curr);
                ++it;  
            }
            edge_indices_in_order.emplace_back(
                std::make_pair(curr, vertices_to_points[indices_to_vertices[*(traversal.begin())]].first)
            );
            
            // Get the area of the region enclosed by the alpha shape 
            // with the optimum value of alpha
            double total_area = 0.0;
            for (auto it = shape.finite_faces_begin(); it != shape.finite_faces_end(); ++it)
            {
                Face_handle_2 face = Tds::Face_range::s_iterator_to(*it);
                auto type = shape.classify(face);
                if (type == Alpha_shape::REGULAR || type == Alpha_shape::INTERIOR)
                {
                    Point_2 p = it->vertex(0)->point();
                    Point_2 q = it->vertex(1)->point();
                    Point_2 r = it->vertex(2)->point();
                    total_area += CGAL::area(p, q, r);
                }
            }
            if (verbose)
            {
                std::cout << "- Number of vertices = " << nvertices << std::endl
                          << "- Number of edges = " << nedges << std::endl
                          << "- Enclosed area = " << total_area << std::endl;
            } 

            return AlphaShape2DProperties(
                x, y, vertex_indices_in_order, edge_indices_in_order,  
                opt_alpha, total_area, is_simple_cycle
            );
        }
};

/**
 * Simplify the given shape with Dyken et al.'s polyline simplification algorithm,
 * using the maximum square distance cost function and a threshold based on the
 * maximum number of desired edges.
 *
 * @param shape     The (alpha) shape to be simplified. 
 * @param max_edges Maximum number of edges to be contained in the alpha shape;
 *                  if the number of edges in the alpha shape exceeds this value,
 *                  the alpha shape is simplified.
 * @param verbose   If true, output intermittent messages to `stdout`.
 * @returns A new `AlphaShape2DProperties` instance containing the simplified shape.
 * @throws std::runtime_error If the given shape is empty. 
 */
AlphaShape2DProperties simplifyAlphaShape(AlphaShape2DProperties& shape, const int max_edges,
                                          const bool verbose = false)
{
    if (verbose)
        std::cout << "- Simplifying boundary ..." << std::endl;

    // Check that the shape has at least 3 vertices 
    if (shape.nv < 3)
        throw std::runtime_error("Cannot simplify shape with < 3 vertices");

    // Instantiate a Polygon object from the given shape 
    std::vector<Point_2> points;
    int v = shape.vertices[0];
    Point_2 p0(shape.x[v], shape.y[v]);
    points.push_back(p0); 
    for (int i = 0; i < shape.nv - 1; ++i)
    {
        v = shape.edges[i].second; 
        Point_2 p(shape.x[v], shape.y[v]); 
        points.push_back(p);
    }
    Polygon_2 polygon(points.begin(), points.end());

    // Simplify the polygon  
    Polygon_2 simplified_polygon = CGAL::Polyline_simplification_2::simplify(polygon, Cost(), Stop(max_edges));
   
    // For each vertex in the simplified polygon ...
    std::vector<int> vertex_indices_in_order_simplified; 
    std::vector<std::pair<int, int> > edge_indices_in_order_simplified;
    int npoints = shape.np; 
    for (auto it = simplified_polygon.vertices_begin(); it != simplified_polygon.vertices_end(); ++it)
    {
        double xit = it->x(); 
        double yit = it->y();

        // ... and identify the index of each vertex in the polygon
        // with respect to the entire point-set 
        double sqdist_to_nearest_point = std::numeric_limits<double>::infinity();
        int idx_nearest_point = 0; 
        for (int i = 0; i < npoints; ++i)
        {
            double xv = shape.x[i]; 
            double yv = shape.y[i];
            double sqdist = std::pow(xit - shape.x[i], 2) + std::pow(yit - shape.y[i], 2); 
            if (sqdist_to_nearest_point > sqdist)
            {
                sqdist_to_nearest_point = sqdist; 
                idx_nearest_point = i; 
            } 
        }
        vertex_indices_in_order_simplified.push_back(idx_nearest_point); 
    }

    // The edges in the polygon are then easy to determine
    for (int i = 0; i < vertex_indices_in_order_simplified.size() - 1; ++i)
    {
        int vi = vertex_indices_in_order_simplified[i];
        int vj = vertex_indices_in_order_simplified[i+1]; 
        edge_indices_in_order_simplified.emplace_back(std::make_pair(vi, vj)); 
    }
    int vi = *std::prev(vertex_indices_in_order_simplified.end());
    int vj = *vertex_indices_in_order_simplified.begin();
    edge_indices_in_order_simplified.emplace_back(std::make_pair(vi, vj)); 
    int nvertices_simplified = vertex_indices_in_order_simplified.size(); 
    int nedges_simplified = edge_indices_in_order_simplified.size();

    // Compute the area of the polygon formed by the simplified 
    // boundary
    double total_area = std::abs(CGAL::to_double(simplified_polygon.area()));
    if (verbose)
    { 
        std::cout << "- Enclosed area of simplified boundary = "
                  << total_area << std::endl
                  << "- Number of vertices of simplified boundary = "
                  << nvertices_simplified << std::endl
                  << "- Number of edges of simplified boundary = "
                  << nedges_simplified << std::endl;
    } 
    
    return AlphaShape2DProperties(
        shape.x, shape.y, vertex_indices_in_order_simplified,
        edge_indices_in_order_simplified, shape.alpha, total_area, true
    );
}

/**
 * Return the area of the symmetric difference of the two given alpha shapes.
 *
 * @param shape1 First alpha shape.
 * @param shape2 Second alpha shape.
 * @returns Area of symmetric difference.  
 */
double getSymmetricDifferenceArea(AlphaShape2DProperties& shape1,
                                  AlphaShape2DProperties& shape2)
{
    typedef CGAL::Exact_predicates_exact_constructions_kernel K_exact;
    typedef CGAL::Polygon_2<K_exact>                          Polygon_2_exact; 
    typedef CGAL::Polygon_with_holes_2<K_exact>               Polygon_with_holes_2;  

    // Check that both shapes have at least 3 vertices 
    if (shape1.nv < 3 || shape2.nv < 3)
        throw std::runtime_error("Cannot compute symmetric difference of shapes with < 3 vertices");

    // Instantiate Polygon objects from the given shapes
    //
    // Use *exact predicates and exact constructions* to define these objects 
    std::vector<K_exact::Point_2> points1, points2;
    int v = shape1.vertices[0];
    K_exact::Point_2 init1(shape1.x[v], shape1.y[v]);
    points1.push_back(init1);
    for (int i = 0; i < shape1.nv - 1; ++i)
    {
        v = shape1.edges[i].second; 
        K_exact::Point_2 p(shape1.x[v], shape1.y[v]);
        points1.push_back(p);
    }
    Polygon_2_exact polygon1(points1.begin(), points1.end());
    v = shape2.vertices[0];
    K_exact::Point_2 init2(shape2.x[v], shape2.y[v]);
    points2.push_back(init2);
    for (int i = 0; i < shape2.nv - 1; ++i)
    {
        v = shape2.edges[i].second; 
        K_exact::Point_2 p(shape2.x[v], shape2.y[v]); 
        points2.push_back(p);
    }
    Polygon_2_exact polygon2(points2.begin(), points2.end());

    // Re-orient polygons if either is clockwise-oriented (negative orientation)
    if (polygon1.orientation() < 0)
        polygon1.reverse_orientation();
    if (polygon2.orientation() < 0)
        polygon2.reverse_orientation();

    // Compute the symmetric difference between the two polygons
    std::vector<Polygon_with_holes_2> sym_diff;
    CGAL::symmetric_difference(polygon1, polygon2, std::back_inserter(sym_diff));

    // Compute the area of each piece of this symmetric difference 
    double area = 0.0;
    for (auto it = sym_diff.begin(); it != sym_diff.end(); ++it)
    {
        // Each piece is a polygon with holes
        //
        // Note that area() returns the *signed* area of each polygon
        area += std::abs(CGAL::to_double(it->outer_boundary().area()));
        for (auto hit = it->holes_begin(); hit != it->holes_end(); ++hit)
            area -= std::abs(CGAL::to_double(hit->area()));
    }

    return area;  
}

#endif
