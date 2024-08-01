/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/1/2024
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
const int __colidx_l = 5; 
const int __colidx_half_l = 6; 
const int __colidx_t0 = 7; 
const int __colidx_growth = 8; 
const int __colidx_eta0 = 9; 
const int __colidx_eta1 = 10;
const int __colidx_group = 11; 
const int __colidx_plasmid = 12;

#endif 
