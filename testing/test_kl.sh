#!/bin/bash

N=20000
Nbs=1
# Assuming your data is in a file called "data.txt"
filename="params.txt"

# # Loop through the lines in the file
while IFS=' ' read -r sigma beta nu seed; do
    echo "===================================================="
    echo "===================================================="
    echo "sigma: $sigma, beta: $beta, nu: $nu, seed:$seed"
    # ############# orderings ###############
    for Ncs in 0 10 30 60 90 120 150 180 210 240 270 300 330 360 390 420 450
    do
        for Nbs in 4
        do
            ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed
            ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --randomordering
            ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --kdtreeordering
            ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --hilbertordering
        done
    done 
done < "$filename"

# # fixed cs
# # Loop through the lines in the file
# while IFS=' ' read -r sigma beta nu seed; do
#     echo "===================================================="
#     echo "===================================================="
#     echo "sigma: $sigma, beta: $beta, nu: $nu, seed:$seed"
#     # ############# orderings ###############
#     for Ncs in 150
#     do
#         for Nbs in {1..150}
#         do
#         ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed
#         ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --randomordering
#         ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --kdtreeordering
#         ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --hilbertordering
#         # ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --knn --seed $seed --mmdordering
#         done
#     done 
# done < "$filename"


mkdir ./log/20k-kl-bs4
mv ./log/locs_* ./log/20k-kl-bs4