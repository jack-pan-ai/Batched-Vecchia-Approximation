#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <string>


location *loadXYcsv(const std::string& filename, int n)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(filename);
    if (!testFile) {
        std::cerr << "Error: File " << filename << " does not exist\n";
        exit(0);
    }
    location *loc = (location *) malloc(sizeof(location));
    loc->x = (double* ) malloc(n * sizeof(double));
    loc->y = (double* ) malloc(n * sizeof(double));
    loc->z = NULL;
    std::ifstream file(filename);
    std::string line;
    int i = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        loc->x[i] = std::stod(token);
        std::getline(ss, token, ',');
        loc->y[i] = std::stod(token);
        // std::getline(ss, token, ',');
        // loc->z[i] = std::stod(token);
        ++i;
    }
    file.close();
    return loc;
}

template <class T>
void loadObscsv(const std::string& filename, int n, T* obs)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(filename);
    if (!testFile) {
        std::cerr << "Error: File " << filename << " does not exist\n";
        exit(0);
    }
    // T* obs = (double* ) malloc(n * sizeof(double));
    std::ifstream file(filename);
    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        T value;
        if (ss >> value) {
            obs[i] = value;
            ++i;
        }
    }
    file.close();
}
template <class T>
void saveLogFileParams(int iter,T x, T y, T z, T llh, T time_llh, T time_Xcmg, int num_loc, int batchsize, int zvecs)
{
    // zvecs is the zvecs th replicate
    std::string filepath = "./data/batchsize_" + std::to_string(batchsize) \
                            +"/params_" + std::to_string(num_loc) + '_' \
                            + std::to_string(batchsize) + '_' + std::to_string(zvecs) + ".csv";
    // Open the log file in append mode
    std::ofstream file(filepath, std::ios::app); // open file in append mode
    if (!file.is_open()) // check if file opened successfully
    {
        std::cerr << "Unable to open file " << filepath << " for writing." << std::endl;
        return;
    }
    
    std::ostringstream oss;
    oss << iter << "," << x << "," << y << "," << z << "," << llh << "," << time_llh << "," << time_Xcmg  << std::endl; // create a comma-separated string of the values
    
    file << oss.str(); // write the string to the file
    file.close(); // close the file
}

void createLogFileParams(int num_loc, int batchsize, int zvecs, int p)
{
    // zvecs is the zvecs th replicate
    std::string filename = "./data/batchsize_" + std::to_string(batchsize) \
                            +"/params_" + std::to_string(num_loc) + '_' \
                            + std::to_string(batchsize) + '_' + std::to_string(zvecs) + ".csv";
    std::ofstream file(filename, std::ios::app); // open file in append mode
    if (!file.is_open()) // check if file opened successfully
    {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    
    if (file.tellp() == 0) // if the file is empty, write the headers
    {   
        if (p == 1){ // univariate matern stationary
            file << "iteration,sigma,range,smoothness,llh,time_llh,time_Xcmg" << std::endl;
        }else if (p == 2){
            file << "iteration,sigma1,sigma2,range,smoothness1,smoothness2,beta,llh,time_llh,time_Xcmg" << std::endl;
        }else{
            printf("It is developing now");
            exit(0);
        }
    }

    file.close(); // close the file
}

template <class T>
void saveLogFileSum(int iterations, std::vector<T> theta, T llk, 
                    double time, int batchsize, int num_loc, int zvecs) {
    // zvecs is the zvecs th replicate
    std::string filename = "./data/batchsize_" + std::to_string(batchsize) \
                            +"/sum_" + std::to_string(num_loc) + '_' \
                            + std::to_string(batchsize) + '_' + std::to_string(zvecs) + ".csv";

    // Print the log message to the log file using printf
    printf("Total Number of Iterations = %d \n", iterations);
    printf("Total Optimization Time = %lf secs \n", time);
    if (theta.size() == 3){
        printf("Model Parameters (Variance, range, smoothness): (%lf, %lf, %lf) -> Loglik: %lf \n",\
                 theta[0], theta[1], theta[2], llk);
        std::ofstream outfile(filename);

        // Write the headers for the CSV file
        outfile << "Iterations, Time, variance, range, smoothness, log-likelihood" << std::endl;
        // Write the log data to the CSV file
        outfile << iterations << ", " << time << ", " << theta[0] << ", " << theta[1] << ", " \
                << theta[2] << ", " << llk << std::endl;

        // Close the file
        outfile.close();
    }
    else if (theta.size() == 6){
        printf("Model Parameters (Variance1, Variance2, range, smoothness1, smoothness2, beta): (%lf, %lf, %lf, %lf, %lf, %lf) -> Loglik: %lf \n", \
                theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], llk);
                // Open the CSV file for writing
        std::ofstream outfile(filename);

        // Write the headers for the CSV file
        outfile << "Iterations, Time, variance1, variance2, range, smoothness1, smoothness2, beta, log-likelihood" << std::endl;
        // Write the log data to the CSV file
        outfile << iterations << ", " << time << ", " << theta[0] << ", " << theta[1] << ", " \
                << theta[2] << ", " << theta[3] << ", " << theta[4] << ", " << theta[5] << ", " \
                << llk << std::endl;
        // Close the file
        outfile.close();
    }
}
#endif