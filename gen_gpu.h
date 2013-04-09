#ifndef GEN_GPU_H
#define GEN_GPU_H

// ******************** General Mat-Mat Functions ******************

__global__ void gen_matvec(float *A, float *x, float *y, const int m, const int n);

__global__ void gen_matvecT(float *A, float *x, float *y, const int m, const int n);


/*
*******************************
** The matrix multiplication **
*******************************
*/

void A_gen(float * out, float * in, float * A, const int m, const int n, dim3 numBlocksm, dim3 threadsPerBlockm);


/*
*****************************************
** The matrix Transpose multiplication **
*****************************************
*/

void AT_gen(float * out, float * in, float * A, const int m, const int n, dim3 numBlocks, dim3 threadsPerBlock);

#endif









