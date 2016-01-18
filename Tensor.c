/*
-- Author: yoosan, SYSUDNLP Group
-- Date: 16/1/15
-- Licence MIT. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const unsigned int MODCONST = (const unsigned int) (2 << (31 - 1));

int multipy_Tensor(double *A, double *B, double *C,
                   int _row, int _column, int column) {
    if (A == NULL || B == NULL) {
        printf("Tensor should malloce...");
        return 0;
    };
    if (_row < 0 || _column < 0 || column < 0) return 0;

    for (int k = 0; k < column; k++) {
        for (int i = 0; i < _row; i++) {
            C[i * column + k] = 0.0;
            for (int j = 0; j < _column; j++) {
                C[i * column + k] += A[i * _column + j] * B[j * column + k];
            }
        }
    }

    return 1;
}

int print_Tensor(double *tensor, int _row, int _column) {
    printf("[Tensor of size %d x %d]\n", _row, _column);
    for (int i = 0; i < _row; i++) {
        for (int j = 0; j < _column - 1; j++) {
            printf("%f ", *(tensor + i * _column + j));
        }
        printf("%f\n", *(tensor + i * _column + _column - 1));
    }
    printf("\n");
    return 1;
}

int init_Tensor(double *tensor, int _row, int _column) {
    srand((unsigned int) time(0));
    double rd;
    for (int i = 0; i < _row; i++) {
        for (int j = 0; j < _column; j++) {
            rd = (double) rand() / MODCONST;
            *(tensor + i * _column + j) = rd;
        }
    }
    return 1;
}

int seq_Tensor(double *tensor, int _row, int _column) {
    for (int i = 0; i < _row; i++) {
        for (int j = 0; j < _column; j++) {
            *(tensor + i * _column + j) = i * j;
        }
    }
    return 1;
}

int add_Tensor(double *tensorA, double *tensorB, int _row, int _column) {
    for (int i = 0; i < _row; i++) {
        for (int j = 0; j < _column; j++) {
            *(tensorA + i * _column + j) += *(tensorB + i * _column + j);
        }
    }
    return 1;
}

int uniform_Tensor(double *tensor, int _row, int _column) {
    srand((unsigned int) time(0));
    double rd;
    for (int i = 0; i < _row; i++) {
        for (int j = 0; j < _column; j++) {
            rd = (double) rand() / MODCONST;
            *(tensor + i * _column + j) = rd - 0.5;
        }
    }
    return 1;
}

int ones_Tensor(double *tensor, int _row, int _column) {
    for (int i = 0; i < _row; i++)
        for (int j = 0; j < _column; j++)
            *(tensor + i * _column + j) = 1.0;
    return 1;
}

int zeros_Tensor(double *tensor, int _row, int _column) {
    for (int i = 0; i < _row; i++)
        for (int j = 0; j < _column; j++)
            *(tensor + i * _column + j) = 0.0;
    return 1;
}

double *copy_Tensor(double *tensor, int _row, int _column) {
    double *copy = (double *) malloc(sizeof(double) * _row * _column);
    for (int i = 0; i < _row; i++)
        for (int j = 0; j < _column; j++)
            copy[i * _column + j] = tensor[i * _column + j];
    return copy;
}

int trans_Tensor(double *tensor, int _row, int _column) {
    double *temp = copy_Tensor(tensor, _row, _column);
    for(int i = 0; i < _column; i++) {
        for (int j = 0; j < _row; j++) {
            tensor[i * _row + j] = temp[j * _column + i];
        }
    }
    return 1;
}


