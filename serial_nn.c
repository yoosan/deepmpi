/*
-- Author: yoosan, SYSUDNLP Group
-- Date: 16/1/18
-- Licence MIT. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Tensor.h"

int main(int argc, char *argv[]) {

    unsigned int insize = 5000;
    unsigned int outsize = 5000;

    double *input = (double *) malloc(sizeof(double) * insize);
    double *output = (double *) malloc(sizeof(double) * outsize);
    double *weights = (double *) malloc(sizeof(double) * insize * outsize);
    double start, end, duration;
    start = MPI_Wtime();
    multipy_Tensor(weights, input, output, outsize, insize, 1);
    end = MPI_Wtime();
    duration = end - start;
    print_Tensor(output, outsize, 1);
    printf("=============> Running mpi costs %f seconds. <==============\n", duration);
    return 0;

}