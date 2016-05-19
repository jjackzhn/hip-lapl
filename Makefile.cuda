CC=cc -fopenmp
NVCC=nvcc
CFLAGS=-I./ -O3
NVCCFLAGS=-I./ -O3 -gencode arch=compute_52,code=sm_52
LDFLAGS=-L/usr/local/cuda/lib64
LIBS=-lcuda -lcudart -lm
E=echo 

all: main

main: main.o io.o lapl_ss.o lapl_cuda.o
	$(CC) $^ -o main $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -std=gnu99 -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) *.o main
