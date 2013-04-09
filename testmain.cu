#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "gen_gpu.h"
#include "zero_kernels.h"
#include "mult_kernels.h"

/* first 6 must be on GPU, last 4 on CPU */
void freeMem (float * a, float * b, float * c, float * d,
               float * e, float * f, float * g, float * h,
               float * i, float * j)
{
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(e);
  cudaFree(f);
  free(g);
  free(h);
  free(i); 
  free(j); 
}

/* first 10 must be on GPU, last 5 on CPU */
void freeMem (float * a, float * b, float * c, float * d,
               float * e, float * f, float * g, float * h,
                float * i, float * j, float * k, float * l,
                float * m, float * n, float * o)
{
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(e);
  cudaFree(f);
  cudaFree(g);
  cudaFree(h);
  cudaFree(i);
  cudaFree(j);
  free(k);
  free(l); 
  free(m); 
  free(n); 
  free(o); 
}


/* Makes a randomly seeded matrix of m x n dimensions
 *  and multiplies it with a randomly seeded vector of
 *  length n using both our and Blanchard's algorithms.
 *  Times them and makes sure outputs match.
 *
 * If our code is faster, prints nothing and returns 0.
 * If our code is slower, prints results and returns 1.
 * If outputs match, prints nothing and returns 0.
 * If outputs don't match, prints results and returns 1.
 *
 * Adds time of our algorithms to times[0], and adds
 * time of Blanchard's algorithm to times[1].
 */
float test (int m, int n, float * times)
{
  // declare matrices for CPU and allocate memory
  TYPE *A = (TYPE *) malloc (m * n * sizeof(TYPE));
  TYPE *B = (TYPE *) malloc (n * sizeof(TYPE));
  TYPE *C = (TYPE *) malloc (m * sizeof(TYPE));
  TYPE *C2 = (TYPE *) malloc (m * sizeof(TYPE));

  // randomly fill in elements of CPU matrices
  createRandomMatrix(A, m * n, time(NULL));
  createRandomMatrix(B, n, time(NULL));
  
  // declare matrices for GPU and allocate memory
  float *d_A, *d_B, *d_C, *d_A2, *d_B2, *d_C2;

  int size_A = n * m * sizeof(float);
  int size_B = n * sizeof(float);
  int size_C = m * sizeof(float);

  cudaMalloc((void**) &d_A, size_A);
  cudaMalloc((void**) &d_B, size_B);
  cudaMalloc((void**) &d_C, size_C);

  cudaMalloc((void**) &d_A2, size_A);
  cudaMalloc((void**) &d_B2, size_B);
  cudaMalloc((void**) &d_C2, size_C);
  
  // copy elements from CPU to GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A2, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B2, B, size_B, cudaMemcpyHostToDevice);

  // run test with our function
  float time1 = matVecMul (d_C, d_B, d_A, m, n);
  
  times[0] += time1;

  // copy results back to CPU
  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

  
  // run naive matrix vector multiplication
  float time2 = matVecNaive (d_C2, d_B2, d_A2, m, n);
  times[1] += time2;

  // copy results back to CPU
  cudaMemcpy(C2, d_C2, size_C, cudaMemcpyDeviceToHost);
  
  
  // check timing results
  if (time2 < time1) {
    printf("Naive was faster at m = %d, n = %d with naiveTime = %f and ourTime = %f\n", m, n, time2, time1);
    freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, A, B, C, C2);
    return 1;
  }

  // check vector c output results
  if (n >= m)
    for (int f = 0; f < m; f++)
      if (fabs (C2[f] - C[f]) > .000001 )
        {
          printf("output mismatch:  for m = %d, n = %d, mismatch at c[%d], with naiveC[%d] = %f, ourC[%d] = %f ******************************\n", m, n, f, f, C2[f], f, C[f]);
          freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, A, B, C, C2);
          return 1;
        }

  // free memory
  freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, A, B, C, C2);

  // return success
  return 0;
}


/* test transpose function */
float testT (int m, int n, float * times)
{
  // declare matrices for CPU and allocate memory
  TYPE *A = (TYPE *) malloc (m * n * sizeof(TYPE));
  TYPE *B = (TYPE *) malloc (m * sizeof(TYPE));
  TYPE *C = (TYPE *) malloc (n * sizeof(TYPE));
  TYPE *C2 = (TYPE *) malloc (n * sizeof(TYPE));
  TYPE *C3 = (TYPE *) malloc (n * sizeof(TYPE));

  TYPE *AT = (TYPE *) malloc (m * n * sizeof(TYPE));

  // randomly fill in elements of CPU matrices
  createRandomMatrix(A, m * n, time(NULL));
  createRandomMatrix(B, m, time(NULL));
  
  // declare matrices for GPU and allocate memory
  float *d_A, *d_B, *d_C, *d_A2, *d_B2, *d_C2, *d_A3, *d_B3, *d_C3, *d_A3T;

  int size_A = n * m * sizeof(float);
  int size_B = m * sizeof(float);
  int size_C = n * sizeof(float);

  cudaMalloc((void**) &d_A, size_A);
  cudaMalloc((void**) &d_B, size_B);
  cudaMalloc((void**) &d_C, size_C);

  cudaMalloc((void**) &d_A2, size_A);
  cudaMalloc((void**) &d_B2, size_B);
  cudaMalloc((void**) &d_C2, size_C);

  cudaMalloc((void**) &d_A3, size_A);
  cudaMalloc((void**) &d_B3, size_B);
  cudaMalloc((void**) &d_C3, size_C);

  cudaMalloc((void**) &d_A3T, size_A);

  // copy elements from CPU to GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A2, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A3, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B2, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B3, B, size_B, cudaMemcpyHostToDevice);


  // run test with our function, making sure to time it
  float time1 = matVecMulT(d_C, d_B, d_A, m, n);

  times[0] += time1;

  // copy results back to CPU
  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

  
  // run test with naive transpose function, making sure to time it
  float time2 = matVecNaiveTrans (d_C2, d_B2, d_A2, m, n);
  times[1] += time2;

  // copy results back to CPU
  cudaMemcpy(C2, d_C2, size_C, cudaMemcpyDeviceToHost);
  
  // run test with Transpose first and use our initial function
  float time3 = matVecMulTransposed(d_C3, d_B3, d_A3, d_A3T, m, n);
  times[2] += time3;

  cudaMemcpy(C3, d_C3, size_C, cudaMemcpyDeviceToHost);
  
  // check vector c output results
  if (m >= n)
    for (int f = 0; f < n; f++)
      if ((fabs (C2[f] - C[f]) > .000001 ) || (fabs (C2[f] - C3[f]) > .000001 ))
        {
          printf("TRANSPOSE output mismatch:  for m = %d, n = %d, mismatch at c[%d], with naiveC[%d] = %f, ourC[%d] = %f , ourTRANSPOSEDC[%d] = %f ******************************\n", m, n, f, f, C2[f], f, C[f], f, C3[f]);
          freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, d_A3, d_B3, d_C3, d_A3T, A, B, C, C2, C3);

          return 1;
        }
  //else  printf(" match:  for m = %d, n = %d, mismatch at c[%d], with naiveC[%d] = %f, ourC[%d] = %f , ourTRANSPOSEDC[%d] = %f ******************************\n", m, n, f, f, C2[f], f, C[f], f, C3[f]);

  // check timing results
  if (time2 < time1) {
    //printf("TRANSPOSE Naive was faster at m = %d, n = %d with naiveTime = %f and ourTime = %f\n", m, n, time2, time1);
    freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, d_A3, d_B3, d_C3, d_A3T, A, B, C, C2, C3); 
    return 1;
  }
  
  if (time2 < time3) {
    //printf("TRANSPOSE Naive was faster at m = %d, n = %d with naiveTime = %f and ourTRANSPOSEDTime = %f\n", m, n, time2, time1);
    freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, d_A3, d_B3, d_C3, d_A3T, A, B, C, C2, C3); 
    return 1;
  }

  // free memory
  freeMem(d_A, d_B, d_C, d_A2, d_B2, d_C2, d_A3, d_B3, d_C3, d_A3T, A, B, C, C2, C3);

  // return success
  return 0;
}


/* no commandline arguments call the entire suite which uses powers of 2,
 * or can run with arguments with m = argv[1] and n = argv[2] and run
 * tests with just those dimensions.
 */
int main(int argc, char *argv[]) {

  // check input
  if (argc != 3 && argc != 1) {
    printf("USAGE ERROR:  Use no parameters for full powers of 2 testing suite, or input matrix dimensions for specific testing as ./testmain m n\n");
    return 1;
  }

  int maxP = 13;
  int testsPerDim = 5;

  float times[2] = {0.0, 0.0};
  float timesT[3] = {0.0, 0.0, 0.0};
  
  int numFailed = 0;
  int numSuccess = 0;
  int numTests = 0;

  int numFailedT = 0;
  int numSuccessT = 0;
  int numTestsT = 0;

  int m, n;

  // run tests if no command line arguments
  
  if (argc == 1)
    for (int p = 7; p <= maxP; p++) {
      for (int q = 4; q <= p; q++) {
        n = (int) pow ((float) 2, (float) p);
        m = (int) pow ((float) 2, (float) q);
        for (int i = 0; i <= testsPerDim; i++) {
          // call test
          if (test (m, n, times))
            numFailed++;
          else
            numSuccess++;
          numTests++;
          // call testT 
          if (testT (m, n, timesT))
            numFailedT++;
          else
            numSuccessT++;
          numTestsT++;
        }
      }
    }
  else {
    for (int i = 0; i <= testsPerDim; i++) {
      // call test
      if (test (atoi(argv[1]), atoi(argv[2]), times))
        numFailed++;
      else
        numSuccess++;
      numTests++;
      // call testT 
      if (testT (atoi(argv[1]), atoi(argv[2]), timesT))
        numFailedT++;
      else
        numSuccessT++;
      numTestsT++;
    }
  }
        
  printf("Tests Succeeded:\t%d\nTests Failed:\t\t%d\nTotal Tests\t\t%d\n", numSuccess, numFailed, numTests);
  printf("TestsT Succeeded:\t%d\nTestsT Failed:\t\t%d\nTotalT Tests\t\t%d\n", numSuccessT, numFailedT, numTestsT);
  
  printf("Our Average:\t%lf ms\nNaive Average:\t%lf ms\n", times[0]/(double)numTests, times[1]/(double)numTests);
  printf("Our AverageT:\t%lf ms\nOur AverageT2:\t%lf ms\nNaive AverageT:\t%lf ms\n", timesT[0]/(double)numTests,  timesT[2]/(double)numTests, timesT[1]/(double)numTests);
  
  printf("ratio: Naive / Ours\t\t=\t%f\n", times[1]/times[0]);
  printf("ratioT: NaiveT / OursT\t\t=\t%f\n", timesT[1]/timesT[0]);
  printf("ratioT2: NaiveT / OursT2\t=\t%f\n", timesT[1]/timesT[2]);
  
  return 0;
}
