/**
 * Implementations of cell-cell and cell-surface interaction forces.
 *
 * In what follows, a population of N cells is represented as a 2-D array of
 * size (N, 9+), where each row represents a cell and stores the following 
 * data: 
 *
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps)
 * 5) timepoint at which cell was formed
 * 6) cell growth rate
 * 7) cell's ambient viscosity with respect to surrounding fluid
 * 8) cell-surface friction coefficient
 *
 * Additional features may be included in the array but these are not
 * relevant for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/22/2023
 */

#ifndef BIOFILM_MECHANICS_HPP
#define BIOFILM_MECHANICS_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;

/** 
 * Return the cell-body coordinate along a cell centerline that is nearest to
 * a given point.
 *
 * @param r Cell center.
 * @param n Cell orientation.
 * @param l Cell length.  
 * @param q Input point.
 * @returns Distance from cell to input point.
 */
template <typename T>
T nearestCellBodyCoordToPoint(const Ref<const Matrix<T, 2, 1> >& r, 
                              const Ref<const Matrix<T, 2, 1> >& n, const T l,
                              const Ref<const Matrix<T, 2, 1> >& q)
{
    T half_l = l / 2;
    T s = (q - r).dot(n);
    if (std::abs(s) <= half_l)
        return s;
    else if (s > half_l)
        return half_l;
    else    // s < -half_l
        return -half_l;
}

/**
 * Return the shortest distance between the centerlines of two cells, along
 * with the cell-body coordinates at which the shortest distance is achieved.
 *
 * @param r1 Center of cell 1.
 * @param n1 Orientation of cell 1.
 * @param l1 Length of cell 1.
 * @param r2 Center of cell 2.
 * @param n2 Orientation of cell 2.
 * @param l2 Length of cell 2.
 * @returns Shortest distance between the two cells, along with the cell-body
 *          coordinates at which the shortest distance is achieved.
 */
template <typename T>
std::tuple<Matrix<T, 2, 1>, T, T> distBetweenCells(const Ref<const Matrix<T, 2, 1> >& r1,
                                                   const Ref<const Matrix<T, 2, 1> >& n1, 
                                                   const T l1,
                                                   const Ref<const Matrix<T, 2, 1> >& r2,
                                                   const Ref<const Matrix<T, 2, 1> >& n2,
                                                   const T l2)
{
    T half_l1 = l1 / 2;
    T half_l2 = l2 / 2;

    // Vector running from r1 to r2
    Matrix<T, 2, 1> r12 = r2 - r1;

    // We are looking for the values of s in [-l1/2, l1/2] and 
    // t in [-l2/2, l2/2] such that the norm of r12 + t*n2 - s*n1
    // is minimized
    T r12_dot_n1 = r12.dot(n1);
    T r12_dot_n2 = r12.dot(n2);
    T n1_dot_n2 = n1.dot(n2);
    T s_numer = r12_dot_n1 - n1_dot_n2 * r12_dot_n2;
    T t_numer = n1_dot_n2 * r12_dot_n1 - r12_dot_n2;
    T denom = 1 - std::pow(n1_dot_n2, 2);
    Matrix<T, 2, 1> dist;
    T s, t;  

    // If the two centerlines are not parallel ...
    if (std::abs(denom) > 1e-6)
    {
        s = s_numer / denom;
        t = t_numer / denom; 
        // Check that the unconstrained minimum values of s and t lie within
        // the desired ranges
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
                s = nearestCellBodyCoordToPoint<T>(r1, n1, l1, q);
                t = half_l2;
            }
            else if (t < s + X && t >= -s + Y)    // In region 2
            {
                // In this case, set s = l1 / 2 and find t
                Matrix<T, 2, 1> q = r1 + half_l1 * n1;
                t = nearestCellBodyCoordToPoint<T>(r2, n2, l2, q); 
                s = half_l1;
            }
            else if (t < -s + Y && t < s + Y)     // In region 3
            {
                // In this case, set t = -l2 / 2 and find s
                Matrix<T, 2, 1> q = r2 - half_l2 * n2;
                s = nearestCellBodyCoordToPoint<T>(r1, n1, l1, q); 
                t = -half_l2; 
            }
            else    // t >= s + Y and t < s + X, in region 4 
            {
                // In this case, set s = -l1 / 2 and find t
                Matrix<T, 2, 1> q = r1 - half_l1 * n1;
                t = nearestCellBodyCoordToPoint<T>(r2, n2, l2, q); 
                s = -half_l1; 
            }
        }
        // Compute distance vector between the two cells 
        dist = r12 + t * n2 - s * n1; 
    }
    // Otherwise, take cap centers of cell 1 and compare the distances to cell 2 
    //
    // TODO Is choosing either cap center of cell 1 a suitable choice for 
    // determining the cell-cell interaction force?
    else
    {
        Matrix<T, 2, 1> p1 = r1 - half_l1 * n1;               // Endpoint for s = -l1 / 2
        Matrix<T, 2, 1> q1 = r1 + half_l1 * n1;               // Endpoint for s = l1 / 2
        T t_p1 = nearestCellBodyCoordToPoint<T>(r2, n2, l2, p1); 
        T t_q1 = nearestCellBodyCoordToPoint<T>(r2, n2, l2, q1);
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

/**
 * Compute the derivatives of the dissipation due to bulk viscosity and 
 * surface friction for each cell with respect to the cell's translational
 * and orientational velocities, *divided by* the cell's translational and
 * orientational velocities. 
 *
 * Namely, if drx, dry, dnx, dny are the translational and orientational
 * velocities of cell i, then the derivative of the total dissipation, P,
 * due to bulk viscosity and surface friction with respect to these velocities
 * are given by 
 *
 * dP/d(drx) = K * drx
 * dP/d(dry) = K * dry
 * dP/d(dnx) = L * dnx
 * dP/d(dny) = L * dny
 *
 * where K and L are prefactors. This function returns these prefactors for
 * each cell in the given population.
 *
 * @param cells Existing population of cells. 
 * @param R Cell radius.
 * @param surface_contact_density Cell-surface contact area density.
 * @returns The prefactors K and L defined above for each cell. 
 */
template <typename T>
Array<T, Dynamic, 2> compositeViscosityForcePrefactors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                       const T R,
                                                       const T surface_contact_density)
{
    Array<T, Dynamic, 2> KL(cells.rows(), 2);
    Array<T, Dynamic, 1> composite_drag = cells.col(7) + cells.col(8) * surface_contact_density / R; 
    KL.col(0) = cells.col(4) * composite_drag; 
    KL.col(1) = cells.col(4).pow(3) * composite_drag / 12;

    return KL; 
}

/**
 * Get all pairs of cells whose distances are within the given threshold. 
 *
 * @param cells Existing population of cells.
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param R Cell radius. 
 * @param Ldiv Cell division length.
 * @returns Array of indices of pairs of neighboring cells.
 */
template <typename T>
Array<T, Dynamic, 6> getCellNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                      const T neighbor_threshold, const T R, const T Ldiv)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return an empty array
    if (n == 1)
        return Array<T, Dynamic, 6>::Zero(0, 6); 

    // Maintain array of neighboring cells 
    //
    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) Cell-body coordinate of contact point along centerline of cell i
    // 5) Cell-body coordinate of contact point along centerline of cell j
    Array<T, Dynamic, 6> neighbors(n * (n - 1) / 2, 6);
    int idx = 0; 

    // For each pair of cells in the population ... 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)   // Note that j < i
        {
            // For two cells to be within neighbor_threshold of each other,
            // their centers must be within neighbor_threshold + Ldiv + 2 * R
            T dist_rij = (cells(i, Eigen::seq(0, 1)) - cells(j, Eigen::seq(0, 1))).matrix().norm();  
            if (dist_rij < neighbor_threshold + Ldiv + 2 * R)
            {
                // In this case, compute their actual distance and check that 
                // they lie within neighbor_threshold of each other 
                auto result = distBetweenCells<T>(
                    cells(i, Eigen::seq(0, 1)).matrix(),
                    cells(i, Eigen::seq(2, 3)).matrix(), cells(i, 4), 
                    cells(j, Eigen::seq(0, 1)).matrix(),
                    cells(j, Eigen::seq(2, 3)).matrix(), cells(j, 4)
                );
                Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
                T si = std::get<1>(result);
                T sj = std::get<2>(result);
                if (dist_ij.norm() < neighbor_threshold)
                {
                    neighbors(idx, 0) = i; 
                    neighbors(idx, 1) = j; 
                    neighbors(idx, Eigen::seq(2, 3)) = dist_ij.array();
                    neighbors(idx, 4) = si; 
                    neighbors(idx, 5) = sj;
                    idx++;  
                }
            }
        }
    }

    // Discard remaining rows and return
    return neighbors(Eigen::seq(0, idx - 1), Eigen::all);
}

/**
 * Update the cell-cell distances in the given array of neighboring pairs
 * of cells.
 *
 * The given array of neighboring pairs of cells is updated in place.  
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array of neighboring pairs of cells. 
 */
template <typename T>
void updateNeighborDistances(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                             Ref<Array<T, Dynamic, 6> > neighbors)
{
    // Each row contains the following information about each pair of 
    // neighboring cells:
    // 0) Index i of first cell in neighboring pair
    // 1) Index j of second cell in neighboring pair
    // 2) x-coordinate of distance vector from cell i to cell j
    // 3) y-coordinate of distance vector from cell i to cell j
    // 4) Cell-body coordinate of contact point along centerline of cell i
    // 5) Cell-body coordinate of contact point along centerline of cell j
    //
    // Columns 2, 3, 4, 5 are updated here
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        auto result = distBetweenCells<T>(
            cells(i, Eigen::seq(0, 1)).matrix(),
            cells(i, Eigen::seq(2, 3)).matrix(), cells(i, 4), 
            cells(j, Eigen::seq(0, 1)).matrix(),
            cells(j, Eigen::seq(2, 3)).matrix(), cells(j, 4)
        ); 
        Matrix<T, 2, 1> dist_ij = std::get<0>(result); 
        T si = std::get<1>(result);
        T sj = std::get<2>(result); 
        neighbors(k, Eigen::seq(2, 3)) = dist_ij.array(); 
        neighbors(k, 4) = si; 
        neighbors(k, 5) = sj;
    }
} 

/**
 * Compute the derivatives of the cell-cell interaction energies for each 
 * cell with respect to the cell's position and orientation coordinates.
 *
 * In this function, the pairs of neighboring cells in the population have
 * been pre-computed. 
 *
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population.
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell.
 * @returns Derivatives of the cell-cell interaction energies with respect 
 *          to cell positions and orientations.   
 */
template <typename T>
Array<T, Dynamic, 4> cellCellForcesFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                 const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                                 const T R, const T Rcell,
                                                 const T E0, const T Ecell)
{
    int n = cells.rows();   // Number of cells

    // If there is only one cell, return zero
    if (n == 1)
        return Array<T, Dynamic, 4>::Zero(n, 4); 

    // Maintain array of partial derivatives of the interaction energies 
    // with respect to x-position, y-position, x-orientation, y-orientation
    Array<T, Dynamic, 4> dEdq = Array<T, Dynamic, 4>::Zero(n, 4); 

    // For each pair of neighboring cells ...
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int i = static_cast<int>(neighbors(k, 0)); 
        int j = static_cast<int>(neighbors(k, 1)); 
        Array<T, 2, 1> dist_ij = neighbors(k, Eigen::seq(2, 3));   // Distance vector from i to j
        T si = neighbors(k, 4);                         // Cell-body coordinate along cell i
        T sj = neighbors(k, 5);                         // Cell-body coordinate along cell j
        T delta_ij = dist_ij.matrix().norm();           // Magnitude of distance vector
        assert(delta_ij > 0 && "Distance vector with zero norm encountered");  
        Array<T, 2, 1> dir_ij = dist_ij / delta_ij;     // Normalized distance vector

        // Get the overlapping distance between cells i and j (this distance
        // is negative if the cells are not overlapping)
        T overlap = 2 * R - delta_ij; 

        // Define prefactors that determine the magnitudes of the interaction
        // forces, depending on the size of the overlap 
        //
        // Case 1: the overlap is positive but less than R - Rcell (i.e., it 
        // is limited to within the EPS coating)
        T prefactor;  
        if (overlap > 0 && overlap < R - Rcell)
        {
            prefactor = 2.5 * E0 * std::sqrt(R) * std::pow(overlap, 1.5); 
        }
        // Case 2: the overlap is instead greater than R - Rcell (i.e., it 
        // encroaches into the bodies of the two cells)
        else if (overlap >= R - Rcell)
        {
            T prefactor1 = E0 * std::pow(R - Rcell, 1.5);
            T prefactor2 = Ecell * std::pow(overlap - R + Rcell, 1.5);
            prefactor = 2.5 * std::sqrt(R) * (prefactor1 + prefactor2); 
        }

        if (overlap > 0)
        {
            // Derivative of cell-cell interaction energy w.r.t position of cell i
            Array<T, 2, 1> vij = prefactor * dir_ij;
            dEdq(i, Eigen::seq(0, 1)) += vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell i
            dEdq(i, Eigen::seq(2, 3)) += vij * si; 
            // Derivative of cell-cell interaction energy w.r.t position of cell j
            dEdq(j, Eigen::seq(0, 1)) -= vij; 
            // Derivative of cell-cell interaction energy w.r.t orientation of cell j
            dEdq(j, Eigen::seq(2, 3)) -= vij * sj;
        }
    }

    return dEdq;  
}

/**
 * Given the current positions, orientations, lengths, viscosity coefficients,
 * and surface friction coefficients for the given population of cells, compute
 * their translational and orientational velocities.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * @param cells Existing population of cells. 
 * @param neighbors Array specifying pairs of neighboring cells in the
 *                  population. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell. 
 * @param surface_contact_density Cell-surface contact area density. 
 * @returns Array of translational and orientational velocities.   
 */
template <typename T>
Array<T, Dynamic, 4> getVelocitiesFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                const Ref<const Array<T, Dynamic, 6> >& neighbors,
                                                const T R, const T Rcell, 
                                                const T E0, const T Ecell, 
                                                const T surface_contact_density)
{
    // For each cell, the relevant Lagrangian mechanics are given by 
    // 
    // dP/d(dq) = -dE/dq + lambda * d(nx^2 + ny^2 - 1)/dq, 
    //
    // where:
    // - P is the total dissipation due to bulk viscosity and surface friction,
    // - E is the total cell-cell interaction energy involving the given cell,
    // - q is a generalized coordinate (x-position, y-position, x-orientation, 
    //   y-orientation),
    // - nx and ny are the x-orientation and y-orientation, respectively,
    // - dq is the corresponding velocity, and 
    // - lambda is a Lagrange multiplier. 
    //
    // Note that dP/d(dq) = K * dq for some prefactor K, which is given in 
    // `composite_viscosity_force_prefactors()`. Moreover, the constraint 
    // 
    // nx^2 + ny^2 - 1 == 0
    // 
    // implies the constraint 
    //
    // 2 * nx * dnx + 2 * ny * dny == 0
    //
    // where dnx and dny are the orientational velocities. This yields the 
    // following value of the Lagrange multiplier:
    //
    // lambda = -0.5 * (nx * dE/dnx + ny * dE/dny)
    //
    int n = cells.rows(); 
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 2> prefactors = compositeViscosityForcePrefactors<T>(
        cells, R, surface_contact_density
    );
    Array<T, Dynamic, 1> K = prefactors.col(0);
    Array<T, Dynamic, 1> L = prefactors.col(1);
    Array<T, Dynamic, 4> dEdq = cellCellForcesFromNeighbors<T>(
        cells, neighbors, R, Rcell, E0, Ecell
    );
    Array<T, Dynamic, 1> mult = cells.col(2) * dEdq.col(2) + cells.col(3) * dEdq.col(3);
    Array<T, Dynamic, 2> dEdn_constrained = (
        dEdq(Eigen::all, Eigen::seq(2, 3)) +
        cells(Eigen::all, Eigen::seq(2, 3)).colwise() * mult
    );
    assert((K != 0).all() && "Composite viscosity force prefactors for positions have zero values"); 
    assert((L != 0).all() && "Composite viscosity force prefactors for orientations have zero values"); 
    velocities.col(0) = -dEdq.col(0) / K;
    velocities.col(1) = -dEdq.col(1) / K; 
    velocities.col(2) = -dEdn_constrained(0) / L;
    velocities.col(3) = -dEdn_constrained(1) / L;

    return velocities;  
}

/**
 * Normalize the orientation vectors of all cells in the given population.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells. 
 */
template <typename T>
void normalizeOrientations(Ref<Array<T, Dynamic, Dynamic> > cells)
{
    Array<T, Dynamic, 1> norms = cells(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array();
    assert((norms > 0).all() && "Zero norms encountered during orientation normalization");
    cells.col(2) /= norms; 
    cells.col(3) /= norms;
}

/**
 * Run one step of an adaptive Runge-Kutta method with the given Butcher 
 * tableau for the given timestep.
 *
 * In this function, the pairs of neighboring cells in the population have 
 * been pre-computed.
 *
 * @param A Runge-Kutta matrix of Butcher tableau. Should be lower triangular
 *          with zero diagonal. 
 * @param b Weights of Butcher tableau. Entries should sum to one. 
 * @param bs Error weights of Butcher tableau. Entries should sum to one. 
 * @param c Nodes of Butcher tableau. First entry should be zero. 
 * @param cells Existing population of cells.
 * @param neighbors Array specifying pairs of neighboring cells in the 
 *                  population. 
 * @param dt Timestep. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell.
 * @param surface_contact_density Cell-surface contact area density.
 * @returns Updated population of cells, along with the array of errors in
 *          the cell positions and orientations.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, Array<T, Dynamic, 4> >
    stepRungeKuttaAdaptiveFromNeighbors(const Ref<const Array<T, Dynamic, Dynamic> >& A,
                                        const Ref<const Array<T, Dynamic, 1> >& b,
                                        const Ref<const Array<T, Dynamic, 1> >& bs, 
                                        const Ref<const Array<T, Dynamic, 1> >& c,
                                        const Ref<const Array<T, Dynamic, Dynamic> >& cells,  
                                        const Ref<const Array<T, Dynamic, 6> >& neighbors, 
                                        const T dt, const T R, const T Rcell,
                                        const T E0, const T Ecell,
                                        const T surface_contact_density)
{
    // Compute velocities at given partial timesteps 
    int n = cells.rows(); 
    int s = b.size(); 
    std::vector<Array<T, Dynamic, 4> > velocities; 
    velocities.push_back(
        getVelocitiesFromNeighbors<T>(
            cells, neighbors, R, Rcell, E0, Ecell, surface_contact_density
        )
    );
    for (int i = 1; i < s; ++i)
    {
        Array<T, Dynamic, 4> multipliers = Array<T, Dynamic, 4>::Zero(n, 4);
        for (int j = 0; j < i; ++j)
            multipliers += velocities[j] * A(i, j);
        Array<T, Dynamic, Dynamic> cells_i(cells); 
        cells_i(Eigen::all, Eigen::seq(0, 3)) += multipliers * dt; 
        velocities.push_back(
            getVelocitiesFromNeighbors<T>(
                cells_i, neighbors, R, Rcell, E0, Ecell, surface_contact_density
            )
        );
    }

    // Compute Runge-Kutta update from computed velocities
    Array<T, Dynamic, Dynamic> cells_new(cells); 
    Array<T, Dynamic, 4> velocities_final1 = Array<T, Dynamic, 4>::Zero(n, 4); 
    Array<T, Dynamic, 4> velocities_final2 = Array<T, Dynamic, 4>::Zero(n, 4); 
    for (int i = 0; i < s; ++i)
    {
        velocities_final1 += velocities[i] * b(i); 
        velocities_final2 += velocities[i] * bs(i); 
    }
    Array<T, Dynamic, 4> delta1 = velocities_final1 * dt; 
    Array<T, Dynamic, 4> delta2 = velocities_final2 * dt; 
    cells_new(Eigen::all, Eigen::seq(0, 3)) += delta1;
    Array<T, Dynamic, 4> errors = delta1 - delta2; 
    
    // Renormalize orientations 
    normalizeOrientations<T>(cells_new); 

    return std::make_pair(cells_new, errors); 
}

#endif
