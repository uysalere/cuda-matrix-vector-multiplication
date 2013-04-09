/* Copyright 2012 Jeffrey Blanchard and Jared Tanner
 *   
 * GPU Accelerated Greedy Algorithms for Compressed Sensing
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



// ************  GLOBAL FUNCTIONS (kernels) **************

// ************  Thresholding functions  *************

__global__ void threshold_one(float *vec, float *vec_thres, int *bin, const int k_bin, const int n) 
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   // xIndex is a value from 1 to k from the vector ind
   
   if ( (xIndex < n) & (bin[xIndex]<=k_bin) )
     vec_thres[xIndex]=vec[xIndex];
}


__global__ void threshold(float *vec, int *bin, const int k_bin, const int n) 
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   // xIndex is a value from 1 to k from the vector ind
   
   if ( (xIndex < n) & (bin[xIndex]>k_bin) )
     vec[xIndex]=0.0f;
}


// This is used in findSupportSet_sort
__global__ void threshold_and_support(float *vec, int *support, const int n, const float T)
{  
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex < n) { 
	if (abs(vec[xIndex])<T) {
		vec[xIndex] = 0.0f;
		support[xIndex]=2;
	}
  } 
}
    

// ******** Vector Definitions ************

__global__ void zero_vector_float(float *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=0.0f;
}

__global__ void zero_vector_int(int *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ){
    int z=0;
    vec[xIndex]=z;
  }
}

__global__ void one_vector_float(float *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=1.0f;
}

__global__ void one_vector_int(int *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=1;
}




// ********** Kernels for Linear Binning Operations when finding supports **************

__global__ void LinearBinning(float *vec, int *bin, int *bin_counters, const int num_bins, const int MaxBin, const int n, const float slope, const float intercept)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  float temp = abs(vec[xIndex]);
  if ( xIndex < n ){
    if ( temp > (intercept *.000001) ){
      bin[xIndex]=max(0.0f,slope * (intercept - temp));
      if (bin[xIndex]<MaxBin) atomicAdd(bin_counters+bin[xIndex],1);
    }
    else bin[xIndex] = slope * intercept + 1.0f; 
  }
}





// ********** Kernels that deal with index sets ***************

// This kernel takes MATLAB 1-indexing and transforms to 0-indexing
void __global__ indexShiftDown(int * d_rows, const int m)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if (xIndex < m) d_rows[xIndex] = d_rows[xIndex]-1;
}

//  This is a kernel to take the union of two supports for CSMPSP
void __global__ joinSupports(int * d_bin, int * d_bin_grad, const int k_bin, const int k_bin_grad, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if (xIndex < n){
	if (d_bin_grad[xIndex] <= k_bin_grad){
		if (d_bin[xIndex] > k_bin){
			d_bin[xIndex] = k_bin;
		}
	}
  }
}


// *********** Used in results functions (results.cu) *************


__global__ void  check_support(float * vec_input, float * vec, const int n, int * support_counter)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex < n) {
	if ( vec_input[xIndex] != 0 ) {
		if (vec[xIndex] != 0) {
			atomicAdd(support_counter, 1);
		}
	}
	else {
		if (vec[xIndex] == 0) {
			atomicAdd(support_counter + 1, 1);
		}	
	} 	
  }
}


// ************** Used in functions.cu ****************

// This is used in FindSupportSet_sort in order to sort the magnitudes

__global__ void magnitudeCopy(float *mag_vec, float *vec, const int n)
{  
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex < n) { mag_vec[xIndex] = abs(vec[xIndex]); }
}







/*
**************************************************************
** VARIOUS KERNELS USED IN DEVELOPMENT BUT NO LONGER ACTIVE **
**************************************************************

__global__ void make_bins(float *vec, int *bin, const int num_bins, const int n, const float slope, const float intercept)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
  if ( xIndex < n ){
    int bin_new_val;
    float temp = abs(vec[xIndex]);
    if ( temp > (intercept *.000001) ){
      bin_new_val=slope * (intercept - temp);
    }
    else bin_new_val = num_bins; 
  bin[xIndex]=bin_new_val;
  }
}


__global__ void count_bins(int *bin, int *bin_counters, const int num_bins, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( (xIndex < n) & (bin[xIndex]<num_bins) )
    atomicAdd(bin_counters+bin[xIndex],1);
}


__global__ void make_and_count_bins(float *vec, int *bin, int *bin_counters, const int num_bins, const int n, const float slope, const float intercept)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  float temp = abs(vec[xIndex]);
  if ( xIndex < n ){
    if ( temp > (intercept *.01) ){
      bin[xIndex]=max(0.0f,slope * (intercept - temp));
      atomicAdd(bin_counters+bin[xIndex],1);
    }
    else bin[xIndex] = slope * intercept + 1.0f; 
  }
}



// This is used to test behavior of skipping cudaThreadSync();
__global__ void count_zero_one(float *vec, float *data, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( (xIndex < n) ){
    if (vec[xIndex] == 0)
      atomicAdd(data,1);
    else if (vec[xIndex] == 1)
      atomicAdd(data+1,1);
  }
}




__global__ void countRest(int *bin, int *bin_counters, const int num_bins, const int maxBin, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( (xIndex < n) & (bin[xIndex]<num_bins) )
    if (bin[xIndex]>= maxBin) atomicAdd(bin_counters+bin[xIndex],1);
}


__global__ void make_and_count_bins_alt(float *vec, int *bin, int *bin_counters, const int num_bins, const int n, const float slope, const float intercept, const float )
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  float temp = abs(vec[xIndex]);
  if ( xIndex < n ){
    bin[xIndex]=max(0.0f,slope * (intercept - temp));
  if ( temp > (intercept *.000001) ){
      atomicAdd(bin_counters+bin[xIndex],1);
    }
    else bin[xIndex] = slope * intercept + 1.0f; 
  }
}



__global__ void update_bins(float *vec, int *bin, int *bin_counters, const int num_bins, const int n, const float slope, const float intercept)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if ( xIndex < n ){
    int bin_new_val;
    float temp = abs(vec[xIndex]);
    if ( temp > (intercept *.000001) ){
      bin_new_val=slope * (intercept - temp);
    }
    else bin_new_val = num_bins; 

    if ( bin[xIndex] != bin_new_val ){
      if (bin[xIndex] < num_bins)
        atomicAdd(bin_counters+bin[xIndex],-1);
      if ( bin_new_val < num_bins )
        atomicAdd(bin_counters+bin[xIndex],1);
      bin[xIndex]=bin_new_val;
    }


  }
}


__global__ void dyadicAdd(int * counter, const int length, const int shift)
{
  if (shift > 0) {
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  	int adds = 2*shift;
	int Index = adds*(xIndex+1)-1;

	if (Index < length) {
		counter[Index] = counter[Index] + counter[Index-shift];
	}
  }
}



// Soft thresholding used in Lee and Wright's SPARSA
__global__ void __soft(float* y, const float* x, float T, int m)
{

    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    float x_e, y_e;

    if(xIndex < m)
    {
	x_e = x[xIndex];
	y_e = fmaxf(fabsf(x_e) - T, 0.f);
	y[xIndex] = y_e / (y_e + T) * x_e;
    }
}



__global__ void halve_bins(int *bin, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
  if ( xIndex < n )
    bin[xIndex] = bin[xIndex]/2;  

}


__global__ void add_adjacent(int *vec, int *vec_shorter, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
  if ( xIndex < n )
    vec_shorter[xIndex] = vec[2 * xIndex] + vec[(2 * xIndex) +1];  

}


__global__ void int_copy(int *vec_to, int *vec_from, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
  if ( xIndex < n )
    vec_to[xIndex] = vec_from[xIndex];  

}


__device__ float getAbsMax(float * d_vec, const int length)
{
  int jj=0;
  float segmentMax = 0;

  for (jj=0; jj<length; jj++) {
	if ( segmentMax < abs(d_vec[jj]) ) segmentMax = abs(d_vec[jj]);
  }

  return segmentMax;
}

__host__ __device__ float getMax(float * vec, const int length)
{
  int jj=0;
  float segmentMax = 0.0f;

  for (jj=0; jj<length; jj++) {
	if ( segmentMax < vec[jj] ) segmentMax = vec[jj];
  }

  return segmentMax;
}




__global__ void segmentMax(float* d_vec, float *segmentMaxes, const int length, const int HighLength, const int HighSegmentLength, const int threadsHigh, const int LowSegmentLength)
{ 
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int startIndex, SegmentLength;

  if ( (xIndex*HighSegmentLength > HighLength) & ( (HighLength + (xIndex-threadsHigh+1)*LowSegmentLength) < length ) ){
	startIndex = HighLength + (xIndex-threadsHigh)*LowSegmentLength;
	SegmentLength = LowSegmentLength;
  }
  else {
	startIndex = xIndex*HighSegmentLength;
	SegmentLength = HighSegmentLength;
  }
  segmentMaxes[xIndex] = getAbsMax(d_vec+startIndex, SegmentLength);
}



__global__ void GetSegmentMax(float * segmentMaxes, float* maxValue, const int length)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex <1) {
  	float mxVal = getMax(segmentMaxes, length);
	maxValue[0] = 0.5f; //mxVal;
  }
}



// ** Speed up the counting by not using atomicAdd  **


__device__ void MakeCountSegment(float *segment, int *bins, const int seglength, int *segCounter, const int countlength, const float low, const float high, const float slope)
{
  int bin;
  float temp;
  for (int jj=0; jj<seglength; jj++){
	temp = abs(segment[jj]);
	if ( ( temp > low ) & ( temp < high ) ) {
	 	bin = (int)ceil(slope*abs(high-temp));
	}
	else if (temp >= high) {
		bin = 0;
	}
	else bin = countlength - 1;
  bins[jj]=bin;
  segCounter[bin] = segCounter[bin] + 1;
  }

  return;
}


__device__ void MakeCountSegment_sharedAtomic(float *segment, int *bins, const int seglength, int *segCounter, int *s_counter, const int countlength, const float low, const float high, const float slope)
{
  int bin;
  float temp;
  for (int jj=0; jj<seglength; jj++){
	temp = abs(segment[jj]);
	if ( ( temp > low ) & ( temp < high ) ) {
	 	bin = (int)ceil(slope*abs(high-temp));
	}
	else if (temp >= high) {
		bin = 0;
	}
	else bin = countlength - 1;
  bins[jj]=bin;
  atomicAdd(s_counter+bin,1);
  }

  for (int jj=0; jj<countlength; jj++) segCounter[jj]=s_counter[jj];

  return;
}




__global__ void make_and_count_seg(float *vec, int *bin, int *segcounter, const int length, const int countlength, const int HighLength, const int HighSegmentLength, const int threadsHigh, const int LowSegmentLength, const float low, const float high, const float slope)
{ 
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int startIndex, SegmentLength, startCountIndex;

  startCountIndex = xIndex*countlength;

  if ( (xIndex*HighSegmentLength > HighLength) & ( (HighLength + (xIndex-threadsHigh+1)*LowSegmentLength) < length ) ){
	startIndex = HighLength + (xIndex-threadsHigh)*LowSegmentLength;
	SegmentLength = LowSegmentLength;
  }
  else {
	startIndex = xIndex*HighSegmentLength;
	SegmentLength = HighSegmentLength;
  }
  MakeCountSegment(vec+startIndex, bin+startIndex, SegmentLength, segcounter+startCountIndex, countlength, low, high, slope);
}


__global__ void make_and_count_seg_sharedAtomic(float *vec, int *bin, int *segcounter, const int length, const int countlength, const int HighLength, const int HighSegmentLength, const int threadsHigh, const int LowSegmentLength, const float low, const float high, const float slope)
{ 
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int startIndex, SegmentLength, startCountIndex;
  extern __shared__ int s_counter[];

  startCountIndex = xIndex*countlength;

  if ( (xIndex*HighSegmentLength > HighLength) & ( (HighLength + (xIndex-threadsHigh+1)*LowSegmentLength) < length ) ){
	startIndex = HighLength + (xIndex-threadsHigh)*LowSegmentLength;
	SegmentLength = LowSegmentLength;
  }
  else {
	startIndex = xIndex*HighSegmentLength;
	SegmentLength = HighSegmentLength;
  }
  MakeCountSegment_sharedAtomic(vec+startIndex, bin+startIndex, SegmentLength, segcounter+startCountIndex, s_counter, countlength, low, high, slope);
}




__global__ void segCountSum(int *counter, int *segcounter, const int countlength)
{ 
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex < countlength){
	for (int jj=0; jj<countlength; jj++){
		counter[xIndex] = counter[xIndex] + segcounter[xIndex + jj*countlength];
	}
  }
}



__global__ void segCountSum_shared(int *counter, int *segcounter, const int countlength)
{ 
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  extern __shared__ int s_counter[];

  if (xIndex < countlength){
	for (int jj=0; jj<countlength; jj++){
		s_counter[xIndex] = s_counter[xIndex] + segcounter[xIndex + jj*countlength];
	}
  }
  counter[xIndex] = s_counter[xIndex];
}






__global__ void magnitude(float *vec, const int n)
{  
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (xIndex < n) { vec[xIndex] = abs(vec[xIndex]); }
}

*/












