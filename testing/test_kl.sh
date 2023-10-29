#!/bin/bash

N=180000
Nbs=1

############# morton ordering ###############
for Ncs in 10 30 60 90 120
do
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn
done

for Ncs in 150 
do
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --ngpu 2
done

############# random ordering ###############
for Ncs in 10 30 60 90 120
do
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --randomordering
done 

for Ncs in 150 
do
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --ngpu 2 --randomordering
done