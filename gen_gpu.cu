#include "gen_gpu.h"

// ******************** General Mat-Mat Functions ******************

__global__ void gen_matvec(float *A, float *x, float *y, const int m, const int n) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < m ){
    float c = 0.0f;
    for(int i=0; i<n; i++)
      c = c + x[i] * A[xIndex + m * i];
    y[xIndex] = c;
  }
}

__global__ void gen_matvecT(float *A, float *x, float *y, const int m, const int n) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ) {
    float c = 0.0f;
    for(int i=0; i<m; i++)
      c = c + y[i] * A[xIndex * m + i];
    x[xIndex] = c;
  }
}



/*
*******************************
** The matrix multiplication **
*******************************
*/

void A_gen(float * out, float * in, float * A, const int m, const int n, dim3 numBlocksm, dim3 threadsPerBlockm)
{
// perform the multiplication
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((float*)A, (float*)in, (float*)out, m, n);
  cudaThreadSynchronize();

  return;
}


/*
*****************************************
** The matrix Transpose multiplication **
*****************************************
*/

void AT_gen(float * out, float * in, float * A, const int m, const int n, dim3 numBlocks, dim3 threadsPerBlock)
{

// perform the multiplication
  gen_matvecT <<< numBlocks, threadsPerBlock >>>((float*)A, (float*)out, (float*)in, m, n);
  cudaThreadSynchronize();

  return;
}








