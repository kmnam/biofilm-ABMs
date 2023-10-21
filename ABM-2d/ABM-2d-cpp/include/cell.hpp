/**
 * A simple class for spherocylindrical cells. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/21/2023
 */

#ifndef BIOFILM_CELL_HPP
#define BIOFILM_CELL_HPP

#include <tuple>
#include <Eigen/Dense>

using namespace Eigen; 

/**
 * A simple containiner class for spherocylindrical cells with a rigid 
 * inner body and an elastic outer coating.
 *
 * The floating-point type used to specify cell coordinates, length,
 * growth rate, birth time, and physical parameters must be defined by
 * the user.
 */
template <typename T>
struct Cell
{
    public:
        Matrix<T, 2, 1> r;    /** Cell center. */
        Matrix<T, 2, 1> n;    /** Cell orientation. */
        T l;                  /** Cell length (excluding caps). */
        T growth_rate;        /** Exponential growth rate. */
        T birth_time;         /** Cell birth time. */ 
        T eta_ambient;        /** Ambient viscosity experienced by cell. */
        T eta_surface;        /** Surface friction experienced by cell. */ 

        /**
         * Straightforward constructor with all attributes specified. 
         *
         * @param rx x-coordinate of cell center.
         * @param ry y-coordinate of cell center.
         * @param nx x-coordinate of cell orientation.
         * @param ny y-coordinate of cell orientation.
         * @param length Cell length.
         * @param growth_rate Exponential growth rate.
         * @param birth_time Cell birth time. 
         * @param eta_ambient Ambient viscosity experienced by cell. 
         * @param eta_surface Surface friction experienced by cell. 
         */
        Cell(const T rx, const T ry, const T nx, const T ny, const T length, 
             const T growth_rate, const T birth_time, const T eta_ambient,
             const T eta_surface)
        {
            this->r(0) = rx;
            this->r(1) = ry;
            this->n(0) = nx;
            this->n(1) = ny;
            this->l = length; 
            this->growth_rate = growth_rate;
            this->birth_time = birth_time;
            this->eta_ambient = eta_ambient;
            this->eta_surface = eta_surface;
        }

        /**
         * Setters for cell attributes. 
         */
        void set_r(const Ref<const Matrix<T, 2, 1> >& r) { this->r = r; }
        void set_rx(const T rx)                          { this->r(0) = rx; }
        void set_ry(const T ry)                          { this->r(1) = ry; }
        void set_n(const Ref<const Matrix<T, 2, 1> >& n) { this->n = n; }
        void set_nx(const T nx)                          { this->n(0) = nx; }
        void set_ny(const T ny)                          { this->n(1) = ny; }
        void setLength(const T length)          { this->l = length; }
        void setGrowthRate(const T growth_rate) { this->growth_rate = growth_rate; }
        void setBirthTime(const T birth_time)   { this->birth_time = birth_time; }
        void setEtaAmbient(const T eta_ambient) { this->eta_ambient = eta_ambient; }
        void setEtaSurface(const T eta_surface) { this->eta_surface = eta_surface; }

        /** 
         * Return the cell-body coordinate along the centerline that is 
         * nearest to a given point.
         *
         * @param q Input point.
         * @returns Distance from cell to ``q``.
         */
        T nearestCellBodyCoordToPoint(const Ref<const Matrix<T, 2, 1> >& q)
        {
            T half_l = this->l / 2;
            T s = (q - this->r).dot(this->n);
            if (std::abs(s) <= half_l)
                return s;
            else if (s > half_l)
                return half_l;
            else    // s < -half_l
                return -half_l;
        }

        /**
         * Return the shortest distance between the cell centerline and 
         * the centerline of another cell, along with the cell-body 
         * coordinates at which the shortest distance is achieved.
         *
         * @param cell2 Pointer to second cell. 
         * @returns Shortest distance from cell to ``cell2``, along with
         *          the cell-body coordinates at which the shortest
         *          distance is achieved.
         */
        std::tuple<Matrix<T, 2, 1>, T, T> distToCell(Cell<T>* cell2)
        {
            // In the comments below, we refer to r1 = this->r, n2 = this->n,
            // and l1 = this->l
            Matrix<T, 2, 1> r2 = cell2->r;
            Matrix<T, 2, 1> n2 = cell2->n;
            Matrix<T, 2, 1> l2 = cell2->l;
            T half_l1 = this->l / 2;
            T half_l2 = l2 / 2;

            // Vector running from r1 to r2
            Matrix<T, 2, 1> r12 = r2 - this->r; 

            // We are looking for the values of s in [-l1/2, l1/2] and 
            // t in [-l2/2, l2/2] such that the norm of r12 + t*n2 - s*n1
            // is minimized
            T r12_dot_n1 = r12.dot(this->n);
            T r12_dot_n2 = r12.dot(n2);
            T n1_dot_n2 = this->n.dot(n2);
            T s_numer = r12_dot_n1 - n1_dot_n2 * r12_dot_n2;
            T t_numer = n1_dot_n2 * r12_dot_n1 - r12_dot_n2;
            T denom = 1 - std::pow(n1_dot_n2, 2);
            Matrix<T, 2, 1> dist; 

            // If the two centerlines are not parallel ...
            if (std::abs(denom) > 1e-6)
            {
                T s = s_numer / denom;
                T t = t_numer / denom; 
                // Check that the unconstrained minimum values of s and t 
                // lie within the desired ranges
                if (std::abs(s) > half_l1 || std::abs(t) > half_l2)
                {
                    // If not, find the side of the square [-l1/2, l1/2] by
                    // [-l2/2, l2/2] in which the unconstrained minimum values
                    // is nearest
                    //
                    // Region 1 (above top side):
                    //     between t = s + X and t = -s + X
                    // Region 2 (right of right side):
                    //     between t = s + X and t = -s + Y
                    // Region 3 (below bottom side): 
                    //     between t = -s + Y and t = s + Y
                    // Region 4 (left of left side):
                    //     between t = s + Y and t = -s + X,
                    // where X = (l2 - l1) / 2 and Y = (l1 - l2) / 2
                    T X = half_l2 - half_l1;
                    T Y = -X; 
                    if (t >= s + X && t >= -s + X)        // In region 1
                    {
                        // In this case, set t = l2 / 2 and find s
                        Matrix<T, 2, 1> q = r2 + half_l2 * n2; 
                        s = this->nearestCellBodyCoordToPoint(q);
                        t = half_l2;
                    }
                    else if (t < s + X && t >= -s + Y)    // In region 2
                    {
                        // In this case, set s = l1 / 2 and find t
                        Matrix<T, 2, 1> q = this->r + half_l1 * this->n;
                        t = cell2->nearestCellBodyCoordToPoint(q); 
                        s = half_l1;
                    }
                    else if (t < -s + Y && t < s + Y)     // In region 3
                    {
                        // In this case, set t = -l2 / 2 and find s
                        Matrix<T, 2, 1> q = r2 - half_l2 * n2;
                        s = this->nearestCellBodyCoordToPoint(q); 
                        t = -half_l2; 
                    }
                    else    // t >= s + Y and t < s + X, in region 4 
                    {
                        // In this case, set s = -l1 / 2 and find t
                        Matrix<T, 2, 1> q = this->r - half_l1 * this->n;
                        t = cell2->nearestCellBodyCoordToPoint(q); 
                        s = -half_l1; 
                    }
                }
                // Compute distance vector to second cell 
                dist = r12 + t * n2 - s * this->n; 
            }
            // Otherwise, take cap centers of cell and compare the distances 
            // to second cell 
            //
            // TODO Is choosing either cap center of cell 1 a suitable choice
            // for determining the cell-cell interaction force?
            else
            {
                Matrix<T, 2, 1> p1 = this->r - half_l1 * this->n;   // Endpoint for s = -l1 / 2
                Matrix<T, 2, 1> q1 = this->r + half_l1 * this->n;   // Endpoint for s = l1 / 2
                T t_p1 = cell2->nearestCellBodyCoordToPoint(p1); 
                T t_q1 = cell2->nearestCellBodyCoordToPoint(q1);
                Matrix<T, 2, 1> dist_to_p1 = p1 - (r2 + t_p1 * n2);   // Vector running towards p1
                Matrix<T, 2, 1> dist_to_q1 = q1 - (r2 + t_q1 * n2);   // Vector running towards q1
                if (dist_to_p1.squaredNorm() < dist_to_q1.squaredNorm())    // Here, s = -l1 / 2
                {
                    dist = -dist_to_p1;
                    s = -half_l1;
                    t = t_p1;
                }
                else                                                        // Here, s = l1 / 2 
                {
                    dist = -dist_to_q1;
                    s = half_l1;
                    t = t_q1;
                }
            }
            
            return std::make_tuple(dist, s, t); 
        }
};

#endif
