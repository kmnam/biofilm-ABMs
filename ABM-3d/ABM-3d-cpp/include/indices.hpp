/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/28/2025
 */

#ifndef CELL_ARRAY_INDEX_DECLARATIONS_3D_HPP
#define CELL_ARRAY_INDEX_DECLARATIONS_3D_HPP

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

const int __colidx_id = 0; 
const int __colidx_rx = 1; 
const int __colidx_ry = 2;
const int __colidx_rz = 3;
const ArithmeticSequence<> __colseq_r = Eigen::seq(1, 3); 
const int __colidx_nx = 4;  
const int __colidx_ny = 5;
const int __colidx_nz = 6;
const ArithmeticSequence<> __colseq_n = Eigen::seq(4, 6);
const ArithmeticSequence<> __colseq_coords = Eigen::seq(1, 6);
const int __colidx_drx = 7; 
const int __colidx_dry = 8;
const int __colidx_drz = 9; 
const ArithmeticSequence<> __colseq_dr = Eigen::seq(7, 9); 
const int __colidx_dnx = 10; 
const int __colidx_dny = 11;
const int __colidx_dnz = 12;
const ArithmeticSequence<> __colseq_dn = Eigen::seq(10, 12); 
const ArithmeticSequence<> __colseq_velocities = Eigen::seq(7, 12); 
const int __colidx_l = 13; 
const int __colidx_half_l = 14; 
const int __colidx_t0 = 15; 
const int __colidx_growth = 16; 
const int __colidx_eta0 = 17; 
const int __colidx_eta1 = 18;
const int __colidx_maxeta1 = 19;
const int __colidx_sigma0 = 20; 
const int __colidx_group = 21;
const std::vector<int> __colvec_required {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
};
const int __ncols_required = 22;

#endif 
