/*
-- Author: yoosan, SYSUDNLP Group
-- Date: 16/1/15
-- Licence MIT. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "Tensor.h"

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


struct mpi_nn {
    unsigned int insize;
    unsigned int outsize;
    double *weights;
    double *bias;
};

int mpi_sigmoid(double *input, unsigned int insize) {
    if (input == NULL || insize < 1) return 0;
    double temp;
    for (int i = 0; i < insize; i++) {
        temp = 1.0 + exp(-input[i]);
        temp = 1.0 / temp;
        input[i] = temp;
    }
    return 1;
}

int mpi_linear_forward(double *input, double *output, struct mpi_nn lr,
                       int taskid, int tasks, MPI_Status status) {

    int workers = tasks - 1, offset, source, dest, div, mod, rows;
    double start, end, duration;
    if (taskid == MASTER) {
        div = lr.outsize / workers;
        mod = lr.outsize % workers;
        offset = 0;
        start = MPI_Wtime();
        for (dest = 1; dest <= workers; dest++) {
            rows = (dest <= mod) ? (div + 1) : div;
            printf("{Forward} Sending %d rows to task %d offset=%d\n", rows, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&lr.weights[offset], rows * lr.insize,
                     MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&input[0], lr.insize, MPI_DOUBLE, dest,
                     FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows * lr.insize;
        }
        //Receive results from workers
        for (source = 1; source <= workers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&output[offset / lr.insize], rows,
                     MPI_DOUBLE, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            printf("{Forward} Received results from task %d\n", source);
        }
        end = MPI_Wtime();
        add_Tensor(output, lr.bias, lr.outsize, 1);
        mpi_sigmoid(output, lr.outsize);
        printf("Result Tensor:\n");
        print_Tensor(output, lr.outsize, 1);
        duration = end - start;
        printf("=============> Running mpi costs %f seconds. <==============\n", duration);
    }

    if (taskid > MASTER) {
        //Receiving from master
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(lr.weights, rows * lr.insize, MPI_DOUBLE, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(input, lr.insize, MPI_DOUBLE, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        multipy_Tensor(lr.weights, input, output, rows, lr.insize, 1);

        //Sending result to master
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&output[0], rows, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }
    return 1;
}

int mpi_linear_backward(double *input, double *grad_output, double *grad_input, struct mpi_nn lr,
                        int taskid, int tasks, MPI_Status status) {

    int workers = tasks - 1, offset, source, dest, div, mod, rows;
    double start, end, duration;
    double *weights = lr.weights;
    start = MPI_Wtime();
    if (taskid == MASTER) {
        div = lr.insize / workers;
        mod = lr.insize % workers;
        trans_Tensor(weights, lr.outsize, lr.insize);
        offset = 0;
        for (dest = 1; dest <= workers; dest++) {
            rows = (dest <= mod) ? (div + 1) : div;
            printf("{Backward} Sending %d rows to task %d offset=%d\n", rows, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&lr.weights[offset], rows * lr.outsize,
                     MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&grad_output[0], lr.outsize, MPI_DOUBLE, dest,
                     FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows * lr.outsize;
        }
        //Receive results from workers
        for (source = 1; source <= workers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&grad_input[offset / lr.outsize], rows,
                     MPI_DOUBLE, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n", source);
        };
        end = MPI_Wtime();
        add_Tensor(input, grad_input, lr.insize, 1);
        printf("Result Tensor:\n");
//        print_Tensor(input, lr.insize, 1);
        duration = end - start;
        printf("=============> Running mpi costs %f seconds. <==============\n", duration);
    }

    if (taskid > MASTER) {
        //Receiving from master
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(lr.weights, rows * lr.outsize, MPI_DOUBLE, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(grad_output, lr.outsize, MPI_DOUBLE, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        multipy_Tensor(lr.weights, grad_output, grad_input, rows, lr.outsize, 1);

        //Sending result to master
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&grad_input[0], rows, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int numtasks;                   /* number of tasks in partition */
    int taskid;                     /* a task identifier */
    int rc;                         /* misc */
    MPI_Status status;              /* mpi status */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Malloc memory
    unsigned int insize = 5000;
    unsigned int outsize = 5000;
    double *input = (double *) malloc(sizeof(double) * insize);
    double *output = (double *) malloc(sizeof(double) * outsize);
    double *weights = (double *) malloc(sizeof(double) * insize * outsize);
    double *bias = (double *) malloc(sizeof(double) * outsize);
    double *grad_input = (double *) malloc(sizeof(double) * insize);
    init_Tensor(input, insize, 1);
    zeros_Tensor(output, outsize, 1);
    uniform_Tensor(weights, outsize, insize);
    uniform_Tensor(bias, outsize, 1);
    uniform_Tensor(grad_input, insize, 1);
    struct mpi_nn linear;
    linear.insize = insize;
    linear.outsize = outsize;
    linear.weights = weights;
    linear.bias = bias;

    // run mpi
    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    mpi_linear_forward(input, output, linear, taskid, numtasks, status);
    MPI_Finalize();
//    mpi_linear_backward(input, output, grad_input, linear, taskid, numtasks, status);
//    MPI_Finalize();
}


