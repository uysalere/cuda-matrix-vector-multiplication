#include <curand.h>
#include "mult_kernels.h"
#include "zero_kernels.h"
#include <math.h>
#include <stdio.h>
#include "transpose_kernel.h"
#include "gen_gpu.h"

/***************MatMulKernel*****************/

__global__ void MatMulKernel(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n / BLOCK_WIDTH blocks horizontally
  // m / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixWidth % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * BLOCK_HEIGHT;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ TYPE b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt) 
    b[threadIdx.x] = in[blockxInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  TYPE cSum = (TYPE) 0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
      // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
      //printf("csum = %f\n", cSum);
    
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadyInd, cSum);
  }
  
}

__global__ void MatMulKernelT(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening vertically on the matrix
  // BLOCK_WIDTH is going vertical
  // BLOCK_HEIGHT is going horizontal
  // m / BLOCK_WIDTH blocks vertically
  // n / BLOCK_HEIGHT block horizontally
 
  // get variables for loop
  // variable for loop length: blockElt
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.y + 1) * BLOCK_WIDTH <= matrixHeight)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixHeight % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_HEIGHT;
    blockyInd = blockIdx.y * BLOCK_WIDTH;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ TYPE b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt)
    b[threadIdx.x] = in[blockyInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  TYPE cSum = (TYPE) 0;
  int threadxInd = blockxInd + threadIdx.x;

  // make sure we are inside the array horizontally
  if (threadxInd < matrixWidth) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockxInd + threadIdx.x : threadxInd 
      // A row index : blockIdx.y * BLOCK_WIDTH + i : blockyInd + i
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(threadxInd) * (matrixHeight) + (blockyInd + i)];

    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadxInd , cSum);
    //printf("el[%d%d;%d] csum = %f tot = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, cSum, *(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x));
  }
}


/***********createRandomMatrix***************/


void createRandomMatrix(TYPE *A, int size, int seed) {
  float *d_A;
  float *h_A = (float *) malloc (size * sizeof(float));
  curandGenerator_t gen;
  size_t size_d_A = size * sizeof(TYPE);

  cudaMalloc((void **) &d_A, size_d_A);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  cudaMemcpy(h_A, d_A, size_d_A, cudaMemcpyDeviceToHost);

  // for (int j = 0; j < 10; j++) 
  //  printf("h_A[%d] = %l=f\n", j, 10* h_A[j]);

  for (int j = 0; j < size; j++) 
    A[j] = h_A[j] / sqrt (size); 

  curandDestroyGenerator(gen);
  cudaFree(d_A);
  free(h_A);
}

float matVecMul (float * out, float * in, float * A, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  int blockCols = (int) ceil(n / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(m / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernels
  zero_vector_float<<<numBlocksm, threadsPerBlockm>>>(out, m);
  MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}

float matVecMulT (float * out, float * in, float * A, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockn = min(n, max_threads_per_block);
  dim3 threadsPerBlockn(threads_perblockn);
  int num_blocksn = (int)ceil((float)n/(float)threads_perblockn);
  dim3 numBlocksn(num_blocksn);

  int blockCols = (int) ceil(n / (double) BLOCK_HEIGHT);
  int blockRows = (int) ceil(m / (double) BLOCK_WIDTH);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernels
  zero_vector_float<<<numBlocksn, threadsPerBlockn>>>(out, n);
  MatMulKernelT<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}

float matVecMulTransposed(float * out, float * in, float * A, float * AT, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockn = min(n, max_threads_per_block);
  dim3 threadsPerBlockn(threads_perblockn);
  int num_blocksn = (int)ceil((float)n/(float)threads_perblockn);
  dim3 numBlocksn(num_blocksn);

  int blockCols = (int) ceil(n / (double) BLOCK_HEIGHT);
  int blockRows = (int) ceil(m / (double) BLOCK_WIDTH);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGridt(blockRows, blockCols);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);


  dim3 blocks((int)ceil (m / (float)BLOCK_DIM), (int) ceil(n / (float)BLOCK_DIM));
  dim3 threads(BLOCK_DIM, BLOCK_DIM);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


  // execute kernels
  zero_vector_float<<<numBlocksn, threadsPerBlockn>>>(out, n);
  transpose<<<blocks, threads>>>(AT, A, m, n);
  MatMulKernel<<<dimGridt, dimBlock, sharedMem>>>(out, in, AT, n, m);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}

float matVecNaive (float * out, float * in, float * A, const int m, const int n) {

  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernel
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((float*)A, (float*)in, (float*)out, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}

float matVecNaiveTrans (float * out, float * in, float * A, const int m, const int n) {
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockn = min(n, max_threads_per_block);
  dim3 threadsPerBlockn(threads_perblockn);
  int num_blocksn = (int)ceil((float)n/(float)threads_perblockn);
  dim3 numBlocksn(num_blocksn);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernel
  gen_matvecT <<< numBlocksn, threadsPerBlockn >>>((float*)A, (float*)out, (float*)in, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}
