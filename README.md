## INTRODUCTION
Coursework of THE parallel and distributed system, implementing MPI for matrix multiplication.

Install mpi(I use OpenMPI) on your machine, firstly. Then clone this repo and complie and run it by: 


```bash
make
mpirun -np 6 ./nn 
or 
mpirun -np 6 ./nn > mpi.txt &

./serial 
or 
./serial > serial.txt &

```
