CC      ?= gcc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -fopenmp
LDLIBS  ?= -lm

poisson: src/poisson.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: poisson
	./poisson --verificar

bench: poisson
	./poisson --n 1024 --iter 500

clean:
	rm -f poisson resultado.csv resultado.md

.PHONY: test bench clean
