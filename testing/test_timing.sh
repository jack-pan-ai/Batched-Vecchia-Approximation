#!/bin/bash

N=260000
Nbs=1
Ncs=120

# nvprof --print-gpu-summary 
# CUDA_VISIBLE_DEVICES=0 
# cuda-memcheck 
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs
#  --ngpu 2
