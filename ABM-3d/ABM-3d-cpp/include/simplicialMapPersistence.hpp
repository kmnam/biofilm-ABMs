/**
 * Classes and functions for simplicial-map persistence. 
 *
 * Author:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/9/2026
 */

#ifndef SIMPLICIAL_MAP_PERSISTENCE_HPP
#define SIMPLICIAL_MAP_PERSISTENCE_HPP

#include <iostream>
#include <memory>
#include <utility>
#include <tuple>
#include <vector>
#include <stack>
#include <queue>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include "fields.hpp"
#include "topology.hpp"
#include "utils.hpp"

using namespace Eigen;

/**
 * A simple implementation of simplicial-map persistence for a filtration of
 * 3-D simplicial complexes. 
 */
template <typename T>
class SimplicialMapPersistence
{
    private:
        SimplicialComplex3D<T> cplex;      // Simplicial complex
        int n0, n1, n2, n3;                // Numbers of simplices 
        int b0, b1, b2, b3;                // Betti numbers 

        // Arrays of homology class generators  
        //
        // Each row in each array corresponds to a generator for a homology
        // class, and therefore has shape (bj, nj)
        Matrix<Z2, Dynamic, Dynamic> H0, H1, H2, H3;

        // Annotation vectors for each simplex 
        //
        // For each simplex in each dimension, store the homology class 
        // generators for which the corresponding annotation is 1
        //
        // In other words, given a simplex (v0, ..., vD), the vector (or set)
        // annotations_D[(v0, ..., vD)] contains k if the cocycle, \phi_k,
        // corresponding to the k-th homology generator evaluates to 1 on the
        // simplex (v0, ..., vD)
        //
        // Note that this homology generator may be of any dimension 
        std::vector<std::unordered_set<int> > annotations_0;
        std::unordered_map<std::pair<int, int>,
                           std::unordered_set<int>,    // generator indices (regardless of dimension)
                           boost::hash<std::pair<int, int> > > annotations_1; 
        std::unordered_map<std::tuple<int, int, int>,
                           std::unordered_set<int>,    // generator indices (regardless of dimension)
                           boost::hash<std::tuple<int, int, int> > > annotations_2; 
        std::unordered_map<std::tuple<int, int, int, int>, 
                           std::unordered_set<int>,    // generator indices (regardless of dimension)
                           boost::hash<std::tuple<int, int, int, int> > > annotations_3; 

        /**
         * Get the full annotation vector for a simplex. 
         */ 
        template <int Dim>
        Matrix<Z2, Dynamic, 1> getAnnotationVector(const Ref<const Array<int, Dim + 1, 1> >& simplex)
        {
            const int b_total = this->b0 + this->b1 + this->b2 + this->b3; 
            Matrix<Z2, Dynamic, 1> annotations = Matrix<Z2, Dynamic, 1>::Zero(b_total);

            // Look up the appropriate annotations for this simplex
            if (Dim == 0)
            {
                int p = simplex(0);
                for (const int k : this->annotations_0[p])
                    annotations(k) = 1;  
            }
            else if (Dim == 1)
            {
                int u = simplex(0); 
                int v = simplex(1); 
                auto pair = std::make_pair(u, v);
                for (const int k : this->annotations_1[pair])
                    annotations(k) = 1; 
            }
            else if (Dim == 2)
            {
                int u = simplex(0); 
                int v = simplex(1); 
                int w = simplex(2); 
                auto tuple = std::make_tuple(u, v, w); 
                for (const int k : this->annotations_2[tuple])
                    annotations(k) = 1; 
            }
            else    // Dim == 3
            {
                int u = simplex(0); 
                int v = simplex(1); 
                int w = simplex(2);
                int x = simplex(3);  
                auto tuple = std::make_tuple(u, v, w, x); 
                for (const int k : this->annotations_3[tuple])
                    annotations(k) = 1; 
            }

            return annotations; 
        }

        /**
         * Get the sum of the full annotation vectors for the boundary of a
         * simplex.
         *
         * This function is used to build up the annotation vector of the 
         * full simplex, which is equal to this sum. 
         */
        template <int Dim>
        Matrix<Z2, Dynamic, 1> getBoundaryAnnotationVector(const Ref<const Array<int, Dim + 1, 1> >& simplex)
        {
            // Get the boundary faces of the simplex 
            Array<int, Dynamic, Dim> boundary(Dim + 1, Dim); 
            for (int i = 0; i < Dim + 1; ++i)
            {
                // Exclude vertex i from the simplex
                for (int j = 0; j < i; ++j)
                    boundary(i, j) = simplex(j);
                for (int j = i + 1; j < Dim + 1; ++j)
                    boundary(i, j - 1) = simplex(j);  
            }

            // Get the sum of the annotation vectors for the boundary faces
            const int b_total = this->b0 + this->b1 + this->b2 + this->b3; 
            Matrix<Z2, Dynamic, 1> boundary_annotations = Array<Z2, Dynamic, 1>::Zero(b_total); 
            for (int i = 0; i < Dim + 1; ++i)
                boundary_annotations += getAnnotationVector<Dim - 1>(boundary.row(i));

            return boundary_annotations;  
        }

        /**
         * Update the homology class generators and annotation vectors 
         * to incorporate the edges in the complex.
         *
         * The points are assumed to already have been incorporated. 
         */
        void insertEdges()
        {
            Array<int, Dynamic, 2> edges = this->cplex.template getSimplices<1>();
            const int b_total = this->b0 + this->b1 + this->b2 + this->b3;

            // For each edge ...
            for (int i = 0; i < this->n1; ++i)
            {
                int u = edges(i, 0); 
                int v = edges(i, 1);
                auto pair = std::make_pair(u, v);

                // Get the boundary annotation vector 
                Matrix<Z2, Dynamic, 1> boundary_annotations
                    = this->getBoundaryAnnotationVector<1>(edges.row(i));

                // If this vector is zero, then create a new 1-dimensional
                // homology generator (and corresponding 1-cocycle) 
                if ((boundary_annotations.array() == 0).all())
                {
                    // The new edge has a new entry in its annotation vector,
                    // corresponding to the new cocycle
                    //
                    // The index of this cocycle is this->b0 + this->b1
                    this->annotations_1[pair].insert(this->b0 + this->b1);

                    // Increment this->b1 
                    this->b1++; 
                }
                // If this vector is nonzero, then ... 
                else 
                {
                    // Find the cocycle with the greatest index that evaluates
                    // to 1 along the boundary
                    int kill_idx; 
                    for (int j = this->b0 + this->b1 - 1; j >= 0; j--)
                    {
                        if (boundary_annotations(j) == 1)
                        {
                            kill_idx = j; 
                            break; 
                        }
                    }
                    int cocycle_dim = (kill_idx < this->b0 ? 0 : 1);

                    // Kill this "pivot" cocycle in the annotations
                    //
                    // For each 0-simplex ...  
                    for (int point = 0; point < this->n0; ++point)
                    {
                        // If the pivot cocycle evaluates to 1 on this 0-simplex ... 
                        if (this->annotations_0[point].find(kill_idx) != this->annotations_0[point].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 0-simplex is zero, then set it to 1
                                    if (this->annotations_0[point].find(j) == this->annotations_0[point].end())
                                        this->annotations_0[point].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_0[point].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_0[point].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater
                        // than the pivot cocycle index
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_0[point])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_0[point].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_0[point].insert(k);
                    }
                    // For each 1-simplex ... 
                    for (auto&& pair_ : this->annotations_1)
                    {
                        std::pair<int, int> edge = pair_.first; 

                        // If the pivot cocycle evaluates to 1 on this 1-simplex ... 
                        if (this->annotations_1[edge].find(kill_idx) != this->annotations_1[edge].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_1[edge].find(j) == this->annotations_1[edge].end())
                                        this->annotations_1[edge].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_1[edge].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_1[edge].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_1[edge])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_1[edge].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_1[edge].insert(k);
                    }

                    // Update all homology generators associated with the 
                    // pivot cocycle
                    if (cocycle_dim == 0)
                    {
                        this->b0--; 
                    }
                    else    // cocycle_dim == 1
                    {
                        this->b1--; 
                    }
                }
            } 
        }

        /**
         * Update the homology class generators and annotation vectors 
         * to incorporate the triangles in the complex.
         *
         * The points and edges are assumed to already have been incorporated. 
         */
        void insertTriangles()
        {
            Array<int, Dynamic, 3> triangles = this->cplex.template getSimplices<2>();
            const int b_total = this->b0 + this->b1 + this->b2 + this->b3; 

            // For each triangle ... 
            for (int i = 0; i < this->n2; ++i)
            {
                int u = triangles(i, 0); 
                int v = triangles(i, 1);
                int w = triangles(i, 2); 
                auto tuple = std::make_tuple(u, v, w);

                // Get the boundary annotation vector 
                Matrix<Z2, Dynamic, 1> boundary_annotations
                    = this->getBoundaryAnnotationVector<2>(triangles.row(i));

                // If this vector is zero, then create a new 2-dimensional
                // homology generator (and corresponding 2-cocycle) 
                if ((boundary_annotations.array() == 0).all())
                {
                    // The new edge has a new entry in its annotation vector,
                    // corresponding to the new cocycle
                    //
                    // This cocycle has index this->b0 + this->b1 + this->b2 
                    this->annotations_2[tuple].insert(this->b0 + this->b1 + this->b2);

                    // Increment this->b2 
                    this->b2++; 
                }
                // If this vector is nonzero, then ... 
                else 
                {
                    // Find the cocycle with the greatest index that evaluates
                    // to 1 along the boundary
                    int kill_idx;
                    for (int j = this->b0 + this->b1 + this->b2 - 1; j >= 0; j--)
                    {
                        if (boundary_annotations(j) == 1)
                        {
                            kill_idx = j; 
                            break; 
                        }
                    }
                    int cocycle_dim;
                    if (kill_idx < this->b0)
                        cocycle_dim = 0;
                    else if (kill_idx < this->b0 + this->b1)
                        cocycle_dim = 1;
                    else 
                        cocycle_dim = 2;

                    // Kill this "pivot" cocycle in the annotations
                    //
                    // For each 0-simplex ... 
                    for (int point = 0; point < this->n0; ++point)
                    {
                        // If the pivot cocycle evaluates to 1 on this 0-simplex ... 
                        if (this->annotations_0[point].find(kill_idx) != this->annotations_0[point].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ...
                                if (boundary_annotations(j) == 1)
                                { 
                                    // ... and the current j-th annotation for 
                                    // the 0-simplex is zero, then set it to 1
                                    if (this->annotations_0[point].find(j) == this->annotations_0[point].end())
                                        this->annotations_0[point].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_0[point].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_0[point].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater
                        // than the pivot cocycle index
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_0[point])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_0[point].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_0[point].insert(k);
                    }
                    // For each 1-simplex ... 
                    for (auto&& pair_ : this->annotations_1)
                    {
                        std::pair<int, int> edge = pair_.first;

                        // If the pivot cocycle evaluates to 1 on this 1-simplex ... 
                        if (this->annotations_1[edge].find(kill_idx) != this->annotations_1[edge].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_1[edge].find(j) == this->annotations_1[edge].end())
                                        this->annotations_1[edge].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_1[edge].erase(j); 
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_1[edge].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_1[edge])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_1[edge].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_1[edge].insert(k);
                    }
                    // For each 2-simplex ... 
                    for (auto&& tuple_ : this->annotations_2)
                    {
                        std::tuple<int, int, int> triangle = tuple_.first;

                        // If the pivot cocycle evaluates to 1 on this 2-simplex ... 
                        if (this->annotations_2[triangle].find(kill_idx) != this->annotations_2[triangle].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_2[triangle].find(j) == this->annotations_2[triangle].end())
                                        this->annotations_2[triangle].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_2[triangle].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_2[triangle].erase(kill_idx);
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_2[triangle])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_2[triangle].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_2[triangle].insert(k);
                    }

                    // Remove the homology generator from the corresponding array
                    if (cocycle_dim == 0)
                    {
                        this->b0--;  
                    }
                    else if (cocycle_dim == 1)
                    {
                        this->b1--; 
                    }
                    else    // cocycle_dim == 2
                    {
                        this->b2--; 
                    }
                }
            } 
        }

        /**
         * Update the homology class generators and annotation vectors 
         * to incorporate the tetrahedra in the complex. 
         *
         * The points, edges, and triangles are assumed to already have been
         * incorporated. 
         */
        void insertTetrahedra()
        {
            Array<int, Dynamic, 4> tetrahedra = this->cplex.template getSimplices<3>(); 
            const int b_total = this->b0 + this->b1 + this->b2 + this->b3; 

            // For each tetrahedron ... 
            for (int i = 0; i < this->n3; ++i)
            {
                int u = tetrahedra(i, 0); 
                int v = tetrahedra(i, 1);
                int w = tetrahedra(i, 2);
                int x = tetrahedra(i, 3);  
                auto tuple = std::make_tuple(u, v, w, x);

                // Get the boundary annotation vector 
                Matrix<Z2, Dynamic, 1> boundary_annotations
                    = this->getBoundaryAnnotationVector<3>(tetrahedra.row(i)); 

                // If this vector is zero, then create a new 3-dimensional
                // homology generator (and corresponding 3-cocycle) 
                if ((boundary_annotations.array() == 0).all())
                {
                    // The new edge has a new entry in its annotation vector,
                    // corresponding to the new cocycle
                    //
                    // This cocycle has index this->b0 + this->b1 + this->b2 + this->b3 
                    this->annotations_3[tuple].insert(this->b0 + this->b1 + this->b2 + this->b3);  

                    // Increment this->b3 
                    this->b3++; 
                }
                // If this vector is nonzero, then ... 
                else 
                {
                    // Find the cocycle with the greatest index that evaluates
                    // to 1 along the boundary
                    int kill_idx;
                    for (int j = this->b0 + this->b1 + this->b2 + this->b3 - 1; j >= 0; j--)
                    {
                        if (boundary_annotations(j) == 1)
                        {
                            kill_idx = j; 
                            break; 
                        }
                    }
                    int cocycle_dim;
                    if (kill_idx < this->b0)
                        cocycle_dim = 0;
                    else if (kill_idx < this->b0 + this->b1)
                        cocycle_dim = 1;
                    else if (kill_idx < this->b0 + this->b1 + this->b2) 
                        cocycle_dim = 2;
                    else 
                        cocycle_dim = 3; 

                    // Kill this "pivot" cocycle in the annotations
                    //
                    // For each 0-simplex ... 
                    for (int point = 0; point < this->n0; ++point)
                    {
                        // If the pivot cocycle evaluates to 1 on this 0-simplex ... 
                        if (this->annotations_0[point].find(kill_idx) != this->annotations_0[point].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2 + this->b3; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ...
                                if (boundary_annotations(j) == 1)
                                { 
                                    // ... and the current j-th annotation for 
                                    // the 0-simplex is zero, then set it to 1
                                    if (this->annotations_0[point].find(j) == this->annotations_0[point].end())
                                        this->annotations_0[point].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_0[point].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_0[point].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater
                        // than the pivot cocycle index
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_0[point])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_0[point].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_0[point].insert(k);
                    }
                    // For each 1-simplex ... 
                    for (auto&& pair_ : this->annotations_1)
                    {
                        std::pair<int, int> edge = pair_.first;

                        // If the pivot cocycle evaluates to 1 on this 1-simplex ... 
                        if (this->annotations_1[edge].find(kill_idx) != this->annotations_1[edge].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2 + this->b3; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_1[edge].find(j) == this->annotations_1[edge].end())
                                        this->annotations_1[edge].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_1[edge].erase(j); 
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_1[edge].erase(kill_idx); 
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_1[edge])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_1[edge].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_1[edge].insert(k);
                    }
                    // For each 2-simplex ... 
                    for (auto&& tuple_ : this->annotations_2)
                    {
                        std::tuple<int, int, int> triangle = tuple_.first;

                        // If the pivot cocycle evaluates to 1 on this 2-simplex ... 
                        if (this->annotations_2[triangle].find(kill_idx) != this->annotations_2[triangle].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2 + this->b3; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_2[triangle].find(j) == this->annotations_2[triangle].end())
                                        this->annotations_2[triangle].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_2[triangle].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_2[triangle].erase(kill_idx);
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_2[triangle])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_2[triangle].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_2[triangle].insert(k);
                    }
                    // Finally, for each 3-simplex ... 
                    for (auto&& tuple_ : this->annotations_3)
                    {
                        std::tuple<int, int, int, int> tetrahedron = tuple_.first;

                        // If the pivot cocycle evaluates to 1 on this 3-simplex ... 
                        if (this->annotations_3[tetrahedron].find(kill_idx) != this->annotations_3[tetrahedron].end())
                        {
                            // For each bit in the boundary annotation vector ... 
                            for (int j = 0; j < this->b0 + this->b1 + this->b2 + this->b3; ++j)
                            {
                                // If the j-th boundary annotation is nonzero ... 
                                if (boundary_annotations(j) == 1)
                                {
                                    // ... and the current j-th annotation for 
                                    // the 1-simplex is zero, then set it to 1
                                    if (this->annotations_3[tetrahedron].find(j) == this->annotations_3[tetrahedron].end())
                                        this->annotations_3[tetrahedron].insert(j);
                                    // Otherwise, set it to 0
                                    else 
                                        this->annotations_3[tetrahedron].erase(j);  
                                }
                            }

                            // Remove the pivot cocycle from the annotations 
                            this->annotations_3[tetrahedron].erase(kill_idx);
                        }

                        // Decrement all cocycle indices that are greater than
                        // the pivot cocycle index 
                        std::vector<int> to_erase, to_add_back; 
                        for (const int k : this->annotations_3[tetrahedron])
                        {
                            if (k > kill_idx)
                            {
                                to_erase.push_back(k); 
                                to_add_back.push_back(k - 1); 
                            }
                        }
                        for (const int k : to_erase)
                            this->annotations_3[tetrahedron].erase(k); 
                        for (const int k : to_add_back)
                            this->annotations_3[tetrahedron].insert(k);
                    }


                    // Remove the homology generator from the corresponding array
                    if (cocycle_dim == 0)
                    {
                        this->b0--;  
                    }
                    else if (cocycle_dim == 1)
                    {
                        this->b1--; 
                    }
                    else if (cocycle_dim == 2)
                    {
                        this->b2--; 
                    }
                    else    // cocycle_dim == 3
                    {
                        this->b3--; 
                    }
                }
            } 

        }

    public:
        /**
         * Constructor with an input simplicial complex and a filtration
         * value. 
         */
        SimplicialMapPersistence(SimplicialComplex3D<T>& cplex)
        {
            this->cplex = cplex;
            this->n0 = this->cplex.getNumPoints(); 
            Array<int, Dynamic, 2> edges = this->cplex.template getSimplices<1>(); 
            Array<int, Dynamic, 3> triangles = this->cplex.template getSimplices<2>(); 
            Array<int, Dynamic, 4> tetrahedra = this->cplex.template getSimplices<3>();
            this->n1 = edges.rows(); 
            this->n2 = triangles.rows(); 
            this->n3 = tetrahedra.rows();

            // Initialize the Betti numbers (start with a collection of isolated
            // points, and update the Betti numbers as we add the higher-dimensional
            // simplices)
            this->b0 = this->n0;
            this->b1 = 0; 
            this->b2 = 0; 
            this->b3 = 0;  

            // Initialize annotations  
            for (int i = 0; i < this->n0; ++i)
            {
                this->annotations_0.push_back(    // this->annotations_0 is a vector 
                    std::unordered_set<int>()
                );  
            }
            for (int i = 0; i < this->n1; ++i)
            {
                auto pair = std::make_pair(edges(i, 0), edges(i, 1)); 
                this->annotations_1[pair]; 
            }
            for (int i = 0; i < this->n2; ++i)
            {
                auto tuple = std::make_tuple(
                    triangles(i, 0), triangles(i, 1), triangles(i, 2)
                ); 
                this->annotations_2[tuple];
            }
            for (int i = 0; i < this->n3; ++i)
            {
                auto tuple = std::make_tuple(
                    tetrahedra(i, 0), tetrahedra(i, 1), tetrahedra(i, 2),
                    tetrahedra(i, 3)
                ); 
                this->annotations_3[tuple];
            }

            // Update the H0 array 
            this->H0 = Matrix<Z2, Dynamic, Dynamic>::Identity(this->b0, this->n0);

            // Update the annotation vector for each point (which is merely
            // (0, ..., 0, 1, 0, ..., 0) for each 0-simplex)
            for (int i = 0; i < this->n0; ++i)
                this->annotations_0[i].insert(i); 

            // Insert the simplices and update the homology generator arrays
            // and annotations
            if (this->n1 > 0) 
                this->insertEdges(); 
            if (this->n2 > 0)
                this->insertTriangles(); 
            if (this->n3 > 0)
                this->insertTetrahedra();  
        }

        /**
         * Return the Betti numbers of the current simplicial complex.
         *
         * @returns Betti numbers of the current simplicial complex. 
         */
        Array<int, Dynamic, 1> getCurrBettiNumbers()
        {
            Array<int, Dynamic, 1> betti(4); 
            betti << this->b0, this->b1, this->b2, this->b3; 

            return betti;  
        }
};

#endif
