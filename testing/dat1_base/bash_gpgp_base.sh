#!/bin/bash

# Define an array with the different values
range=(0.3 0.1 0.03)
alpha=(0.5 1.5)

# Loop over the array and execute the commands with each value
for ra in "${range[@]}"
do
    for al in "${alpha[@]}"
    do
        echo "[INFO] ./data_1k_${ra}_${al}_gpgp is starting now!"
        cd "./data_1k_${ra}_${al}_gpgp"
        Rscript ./train_helper_1k_base.R
        cd ..
        echo "[INFO] ./data_1k_${ra}_${al}_gpgp is done!"
    done
done

