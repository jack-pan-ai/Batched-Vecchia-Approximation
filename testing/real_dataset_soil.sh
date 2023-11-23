#!/bin/bash 

for num_loc in 250000 500000
    for Ncs in 60 90
        ./bin/test_dvecchia_batch -N 1:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $num_loc --omp_threads 40 --vecchia_cs $Ncs --ikernel ?:?:? --randomordering --knn
    done 
    for Ncs in 120 150
        ./bin/test_dvecchia_batch -N 1:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $num_loc --omp_threads 40 --vecchia_cs $Ncs --ikernel ?:?:? --randomordering --knn --ngpu 2
    done 
done

# test simulation for powexp
# ./bin/test_dvecchia_batch -N 1:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc 20 --omp_threads 1 --vecchia_cs 10 --ikernel ?:?:0.5 --randomordering