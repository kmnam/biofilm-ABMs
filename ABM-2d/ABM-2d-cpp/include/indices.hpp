/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/10/2024
 */

#ifndef CELL_ARRAY_INDEX_DECLARATIONS_HPP
#define CELL_ARRAY_INDEX_DECLARATIONS_HPP

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
const int __colidx_dnx = 7; 
const int __colidx_dny = 8;
const ArithmeticSequence<> __colseq_velocities = Eigen::seq(5, 8); 
const int __colidx_l = 9; 
const int __colidx_half_l = 10; 
const int __colidx_t0 = 11; 
const int __colidx_growth = 12; 
const int __colidx_eta0 = 13; 
const int __colidx_eta1 = 14;
const int __colidx_group = 15; 
const int __colidx_plasmid = 16;

#endif 
