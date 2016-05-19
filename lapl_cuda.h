#ifndef _HEAT_CUDA_H
#define _HEAT_CUDA_H
void init_lapl_cuda(float *, float);
void fini_lapl_cuda(float *);
void lapl_iter_cuda(int [], int []);
#endif /* _HEAT_CUDA_H */
