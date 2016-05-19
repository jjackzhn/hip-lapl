#include <stdio.h>
#include <cuda.h>
#include <defs.h>

static int idx;			/* holds the current iteration index */
static float *dev_arr[2];	/* GPU arrays */
__constant__ float delta;
__constant__ float lapl_norm;

/*
 * Naive implementation of a single iteration of the lapl
 * equation. Each thread takes one site of the output array
 */
__global__ void
dev_lapl_iter(float *out, float *in)
{
  int itx = threadIdx.x, ity = threadIdx.y;
  int ibx = blockIdx.x, iby = blockIdx.y;
  int nbx = gridDim.x, nby = gridDim.y;
  int x = itx + ibx*NTX;
  int Lx = nbx*NTX;
  int Ly = nby*NTY;
  int y = ity + iby*NTY;
  int v00 = y*Lx + x;
  int v0p = y*Lx + (x + 1)%Lx;
  int v0m = y*Lx + (Lx + x - 1)%Lx;
  int vp0 = ((y+1)%Ly)*Lx + x;
  int vm0 = ((Ly+y-1)%Ly)*Lx + x;
  out[v00] = lapl_norm*in[v00]
    + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
  return;
}

/*
 * Optimized implementation of a single iteration of the lapl
 * equation. Uses shared memory for input array with halos. Subset of
 * threads used to fill-in halo data for input array
 */
__global__ void
dev_lapl_iter_optim(float *out, float *in)
{
  int itx = threadIdx.x, ity = threadIdx.y;
  int ibx = blockIdx.x, iby = blockIdx.y;
  int nbx = gridDim.x, nby = gridDim.y;
  int x = itx + ibx*NTX;
  int y = ity + iby*NTY;
  int Lx = nbx*NTX;
  int Ly = nby*NTY;
  __shared__ float in_sub[(NTX+2)*(NTY+2)];
  /* Fill in bulk (inner volume) sites */
  in_sub[(itx+1)+(ity+1)*(NTX+2)] = in[y*Lx + x];
  int ithr = itx + NTX*ity;
  int xoff = NTX*ibx;
  int yoff = NTY*iby;
  /* Use sub set of available threads to fill in halo data */
  if (ithr < NTX) {
    /* y = -1 */
    int ix = ithr;
    int ii = ix+1;
    int vv = ix+xoff + ((Ly+yoff-1)%Ly)*Lx;
    in_sub[ii] = in[vv];
    /* y = NTY */
    ii = (ix+1) + (NTY+1)*(NTX+2);
    vv = (ix+xoff) + ((NTY+yoff)%Ly)*Lx;
    in_sub[ii] = in[vv];
  } else if ((NTX-1)*NTY <= ithr) {
    /* x = -1 */
    int iy = ithr - (NTX-1)*NTY;
    int ii = (iy+1)*(NTX+2);
    int vv = (Lx+xoff-1)%Lx + (iy+yoff)*Lx;
    in_sub[ii] = in[vv];
    /* x = NTX */
    ii = (NTX+1) + (iy+1)*(NTX+2);
    vv = (NTX + xoff)%Lx + (yoff+iy)*Lx;
    in_sub[ii] = in[vv];
  }
  /* Sync before continuing, as the next step relies on all threads
     having filled in the input shared array */
  __syncthreads();
  /* Heat iteration. The output array is the array in global
     memory. The input array is the subarray, with the halo data */
  {
    int v00 = y*Lx + x;
    int x00 = (ity+1)*(NTX+2) + (itx+1);
    int x0p = (ity+1)*(NTX+2) + (itx+2);
    int x0m = (ity+1)*(NTX+2) + itx;
    int xp0 = (ity+2)*(NTX+2) + itx+1;
    int xm0 = (ity)*(NTX+2) + itx+1;
    out[v00] = lapl_norm*in_sub[x00]
      + delta*(in_sub[x0p] + in_sub[x0m] + in_sub[xp0] + in_sub[xm0]);
  }
  return;
}

/*
 * Initialize by writing norm and delta to constant memory,
 * allocating device arrays and copying input array to device
 */
extern "C" void
init_lapl_cuda(float *arr, float sigma)
{
  cudaSetDevice(0);
  cudaMalloc((void **)&dev_arr[0], sizeof(float)*Lx*Ly);
  cudaMalloc((void **)&dev_arr[1], sizeof(float)*Lx*Ly);
  cudaMemcpy(dev_arr[0], arr, sizeof(float)*Lx*Ly, cudaMemcpyHostToDevice);
  float xdelta = sigma / (1.0+4.0*sigma);
  float xnorm = 1.0/(1.0+4.0*sigma);
  cudaMemcpyToSymbol("norm", &xnorm, sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("delta", &xdelta, sizeof(float), 0, cudaMemcpyHostToDevice);
  idx = 0;
  return;
}

/*
 * This function drives the CUDA kernel. It also blocks such that
 * returning from this function guarantees completion of kernel. Each
 * call will increment in the lapl iteration index "idx"
 */
extern "C" void
lapl_iter_cuda(int block_dims[], int thread_dims[])
{
  dim3 thr(thread_dims[2], thread_dims[1], thread_dims[0]);
  dim3 blk(block_dims[2], block_dims[1], block_dims[0]);
  dev_lapl_iter_optim <<< blk, thr >>> (dev_arr[(idx+1)%2], dev_arr[idx%2]);
  idx++;
  cudaDeviceSynchronize();
  return;
}

/*
 * Finalize by copying the output array to main memory and freeing
 * arrays on device
 */
extern "C" void
fini_lapl_cuda(float *arr)
{
  cudaMemcpy(arr, dev_arr[idx % 2], sizeof(float)*Lx*Ly, cudaMemcpyDeviceToHost);
  cudaFree(dev_arr[0]);
  cudaFree(dev_arr[1]);
  return;
}
