CC=hipcc -g
HIPCC=hipcc -g
CFLAGS=-I./ -O3
HIPCCFLAGS=-I./ -O3
LDFLAGS=
LIBS=-lm

all: main

main: main.o io.o lapl_ss.o lapl_cuda.o
	$(HIPCC) $^ -o main $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

clean:
	rm *.o main
