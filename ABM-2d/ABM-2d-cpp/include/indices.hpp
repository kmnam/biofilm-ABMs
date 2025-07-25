/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/12/2024
 */

#ifndef CELL_ARRAY_INDEX_DECLARATIONS_HPP
#define CELL_ARRAY_INDEX_DECLARATIONS_HPP

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

const int __colidx_id = 0; 
const int __colidx_rx = 1; 
const int __colidx_ry = 2;
const ArithmeticSequence<> __colseq_r = Eigen::seq(1, 2); 
const int __colidx_nx = 3;  
const int __colidx_ny = 4; 
const ArithmeticSequence<> __colseq_n = Eigen::seq(3, 4);
const ArithmeticSequence<> __colseq_coords = Eigen::seq(1, 4);
const int __colidx_drx = 5; 
const int __colidx_dry = 6;
const ArithmeticSequence<> __colseq_dr = Eigen::seq(5, 6); 
const int __colidx_dnx = 7; 
const int __colidx_dny = 8;
const ArithmeticSequence<> __colseq_dn = Eigen::seq(7, 8); 
const ArithmeticSequence<> __colseq_velocities = Eigen::seq(5, 8); 
const int __colidx_l = 9; 
const int __colidx_half_l = 10; 
const int __colidx_t0 = 11; 
const int __colidx_growth = 12; 
const int __colidx_eta0 = 13; 
const int __colidx_eta1 = 14;
const int __colidx_maxeta1 = 15; 
const int __colidx_group = 16;
const std::vector<int> __colvec_required {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};
const int __ncols_required = 17;

// Optional column indices 
const int __colidx_boundary = 17; 
const int __colidx_negpole_t0 = 18; 
const int __colidx_pospole_t0 = 19;
const ArithmeticSequence<> __colseq_poles_t0 = Eigen::seq(18, 19); 

#endif 
