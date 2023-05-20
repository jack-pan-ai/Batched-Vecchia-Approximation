#!/bin/bash

# Define an array with the different values
range=(0.3 0.1 0.03)
alpha=(0.5 1.5)

# Loop over the array and execute the commands with each value
for ra in "${range[@]}"
do
    for al in "${alpha[@]}"
    do
        mv "./data_1k_${ra}_${al}_bv" "../data"
        cd ..
        bash ./data/run_script.sh
        mv "./data" "./dat1_bv_dependent/data_1k_${ra}_${al}_bv"
        cd ./dat1_bv_dependent
    done
done

