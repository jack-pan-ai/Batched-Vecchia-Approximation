#!/bin/bash

N=20000
# Assuming your data is in a file called "data.txt"
filename="params.txt"

# Loop through the lines in the file
while IFS=' ' read -r sigma beta nu seed; do
    echo "===================================================="
    echo "===================================================="
    echo "sigma: $sigma, beta: $beta, nu: $nu, seed:$seed"
    # ############# orderings ###############
    for seedi in {1..50}
    do  
        # # block vecchia
        # for Ncs in 10
        # do
        #     for Nbs in 20
        #     do
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --randomordering 
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --kdtreeordering 
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --hilbertordering 
        #     done
        # done
        # block vecchia
        # for Ncs in 180
        # do
        #     for Nbs in 20
        #     do
        #     # Read the second row from the file and extract the values using awk
        #     values=$(sed -n '2p' ./log/locs_20000_cs_60_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_morton | awk '{print $2, $3, $4}')
        #     # Remove trailing comma if it exists
        #     values="${values//,}"
        #     # Assign the values to variables
        #     read sigma_pre beta_pre nu_pre llh_pre <<< "$values"
        #     ./bin/test_dvecchia_batch --kernel_init $sigma_pre:$beta_pre:$nu_pre --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi 


        #     values=$(sed -n '2p' ./log/locs_20000_cs_60_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_random | awk '{print $2, $3, $4}')
        #     values="${values//,}"
        #     read sigma_pre beta_pre nu_pre <<< "$values"
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --randomordering --kernel_init $sigma_pre:$beta_pre:$nu_pre

        #     values=$(sed -n '2p' ./log/locs_20000_cs_60_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_kdtree | awk '{print $2, $3, $4}')
        #     values="${values//,}"
        #     read sigma_pre beta_pre nu_pre <<< "$values"
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --kdtreeordering --kernel_init $sigma_pre:$beta_pre:$nu_pre

        #     values=$(sed -n '2p' ./log/locs_20000_cs_60_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_hilbert | awk '{print $2, $3, $4}')
        #     values="${values//,}"
        #     read sigma_pre beta_pre nu_pre <<< "$values"
        #     ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --hilbertordering --kernel_init $sigma_pre:$beta_pre:$nu_pre 
        #     done
        # done
        # classic vecchia
        for Ncs in 30 #30 60 
        do
            for Nbs in 1
            do
            values=$(sed -n '2p' ./log/locs_20000_cs_10_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_morton | awk '{print $2, $3, $4}')
            values="${values//,}"
            read sigma_pre beta_pre nu_pre <<< "$values"
            ./bin/test_dvecchia_batch --kernel_init $sigma_pre:$beta_pre:$nu_pre  --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --randomordering 
            done
        done
        # classic vecchia
        for Ncs in 60 #30 60 
        do
            for Nbs in 1
            do
            values=$(sed -n '2p' ./log/locs_20000_cs_60_bs_20_seed_${seedi}_kernel_1.500000:0.014290:2.500000_morton | awk '{print $2, $3, $4}')
            values="${values//,}"
            read sigma_pre beta_pre nu_pre <<< "$values"
            ./bin/test_dvecchia_batch --kernel_init $sigma_pre:$beta_pre:$nu_pre  --ikernel $sigma:$beta:$nu -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs $Ncs --knn --seed $seedi --randomordering 
            done
        done
    done 
done < "$filename"

# mkdir ./log/20k-simu
# mv ./log/* ./log/20k-simu