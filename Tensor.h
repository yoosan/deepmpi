/*
-- Author: yoosan, SYSUDNLP Group
-- Date: 16/1/17
-- Licence MIT. 
*/

#ifndef DEEPMPI_TENSOR_H
#define DEEPMPI_TENSOR_H

/* Tensor multiplication  */
int multipy_Tensor(double *, double *, double *,
                   int _row, int _column, int column);

/* Print a tensor */
int print_Tensor(double *, int _row, int _column);

/* Initialize tensor with random value between [0, 1] */
int init_Tensor(double *tensor, int _row, int _column);

/* Make sequece initialization */
int seq_Tensor(double * tensor, int _row, int _column);

/* Do adding operation of two tensors*/
int add_Tensor(double *tensorA, double *tensorB, int _row, int _column);

/* Uniform tensor with interval value [-0.5, 0.5]*/
int uniform_Tensor(double *tensor, int _row, int _column);

/* Initialize tensor with 1s. */
int ones_Tensor(double *tensor, int _row, int _column);

/* Initialize tensor with 0s. */
int zeros_Tensor(double *tensor, int _row, int _column);

/* Deep copy */
double* copy_Tensor(double *tensor, int _rowm, int _column);

/* Transposing tensor */
int trans_Tensor(double *tensor, int _rowm, int _column);

#endif //DEEPMPI_TENSOR_H
