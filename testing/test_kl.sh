#!/bin/bash

N=180000
# N=260000
# Assuming your data is in a file called "data.txt"
filename="params.txt"

# Loop through the lines in the file
while IFS=' ' read -r sigma beta nu seed; do
    echo "===================================================="
    echo "===================================================="
    echo "sigma: $sigma, beta: $beta, nu: $nu, seed:$seed"
    ############# morton ordering ###############
    for Ncs in 10 30 60 90 
    do
    ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn  --seed $seed
    done
    for Ncs in 120 150 
    do
        ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --ngpu 2  --seed $seed
    done
    # ############# random ordering ###############
    for Ncs in 10 30 60 90 
    do
    ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --randomordering --seed $seed
    done 
    for Ncs in 120 150 
    do
    ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --ngpu 2 --randomordering --seed $seed
    done
done < "$filename"