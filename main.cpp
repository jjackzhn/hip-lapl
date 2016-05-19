#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "defs.h"
#include "io.h"
#include "lapl_ss.h"
#include "lapl_cuda.h"

int Lx, Ly;

double
stop_watch(double t0) 
{
  double time;
  struct timeval t;
  gettimeofday(&t, NULL);
  time = (double) t.tv_sec + (double) t.tv_usec * 1e-6;  
  return time-t0;
}

void
usage(char *argv[]) {
  fprintf(stderr, " Usage: %s LX LY NITER IN_FILE OUT_FILE\n", argv[0]);
  return;
}

int
main(int argc, char *argv[]) {
  /* Check the number of command line arguments */
  if(argc != 6) {
    usage(argv);
    exit(1);
  }
  /* The length of the array in x and y is read from the command
     line */
  Lx = atoi(argv[1]);
  Ly = atoi(argv[2]);
  /* The number of iterations */
  int niter = atoi(argv[3]);
  /* Fixed "sigma" */
  float sigma = 0.01;
  printf(" Ly,Lx = %d,%d\n", Ly, Lx);
  printf(" niter = %d\n", niter);
  printf(" input file = %s\n", argv[4]);
  printf(" output file = %s\n", argv[5]);
  /* Allocate the buffer for the data */
  float *arr = (float*)malloc(sizeof(float)*Lx*Ly);
  /* read file to buffer */
  read_from_file(arr, argv[4]);
  /* allocate super-site buffers */
  supersite *ssarr[2];
  posix_memalign((void**)&ssarr[0], 16, sizeof(supersite)*Lx*Ly/4);
  posix_memalign((void**)&ssarr[1], 16, sizeof(supersite)*Lx*Ly/4);
  /* convert input array to super-site packed */
  to_supersite(ssarr[0], arr);
  /* do iterations, record time */
  double t0 = stop_watch(0);
  for(int i=0; i<niter; i++) {
    lapl_iter_supersite(ssarr[(i+1)%2], sigma, ssarr[i%2]);
  }
  t0 = stop_watch(t0)/(double)niter;
  /* write the result after niter iteraions */
  char fname[256];
  /* construct filename */
  sprintf(fname, "%s.ss%08d", argv[5], niter);
  /* convert from super-site packed */
  from_supersite(arr, ssarr[niter%2]);
  /* write to file */
  write_to_file(fname, arr);
  /* write timing info */
  printf(" iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
	 niter, Lx, Ly, t0*1e6, 
	 Lx*Ly*sizeof(float)*2.0/(t0*1.0e9), 
	 (Lx*Ly*6.0)/(t0*1.0e9));
  /* free super-site buffers */
  for(int i=0; i<2; i++) {
    free(ssarr[i]);
  }
  /*
   * GPU part
   */

  /* read file again for GPU run */
  read_from_file(arr, argv[4]);
  /* Fixed number of threads per block (in x- and y-direction), number
     of blocks per direction determined by dimensions Lx, Ly */
  int threads[] = {1, NTY, NTX};
  int blocks[] = {1, Ly/NTY, Lx/NTX};
  /* Initialize: allocate GPU arrays and load array to GPU */
  init_lapl_cuda(arr, sigma);
  /* Do iterations on GPU, record time */
  t0 = stop_watch(0);
  for(int i=0; i<niter; i++) {
    lapl_iter_cuda(blocks, threads);
  }
  t0 = stop_watch(t0)/(double)niter;
  /* construct filename for writing  */
  sprintf(fname, "%s.cu%08d", argv[5], niter);
  /* copy GPU array to main memory and free GPU arrays */
  fini_lapl_cuda(arr);
  /* write to file */
  write_to_file(fname, arr);
  /* write timing info */
  printf(" iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
  	 niter, Lx, Ly, t0*1e6,
  	 Lx*Ly*sizeof(float)*2.0/(t0*1.0e9),
  	 (Lx*Ly*6.0)/(t0*1.0e9));
  /* free main memory array */
  free(arr);
  return 0;
}
