#ifndef VMULT_H
#define VMULT_H

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
#define TYPE float

/***************MatMulKernel*****************/

__global__ void MatMulKernel(TYPE *, TYPE *, TYPE *, const int, const int );
__global__ void MatMulKernelT(TYPE *, TYPE *, TYPE *, const int , const int );
 
/***********createRandomMatrix***************/

void createRandomMatrix(TYPE *, int, int);

float matVecMul (float *, float *, float *, const int, const int);
float matVecMulT (float *, float *, float *, const int, const int);
float matVecMulTransposed (float *, float *, float *, float *, const int, const int);
float matVecNaive (float * out, float * in, float * A, const int m, const int n);
float matVecNaiveTrans (float * out, float * in, float * A, const int m, const int n);
#endif
