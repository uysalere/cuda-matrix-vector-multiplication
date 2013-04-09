CC=nvcc
CFLAGS=-I./lib/ -I. -arch=sm_20 -lcurand -lm

default: testmain

testmain:	testmain.cu
	$(CC) -o testmain testmain.cu mult_kernels.cu transpose_kernel.cu gen_gpu.cu zero_kernels.cu $(CFLAGS)

clean:
	rm -f testmain
