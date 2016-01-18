## INTRODUCTION of deepmpi
Exercise of MPI for deep leaning, mainly mpi for matrix multiplication.

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
