CC = mpicc

all: nn serial

nn: Tensor.c mpi_nn.c
	$(CC) Tensor.c mpi_nn.c -o nn

serial: Tensor.c serial_nn.c
	$(CC) Tensor.c serial_nn.c -o serial

clean:
	rm -rf nn serial *.o