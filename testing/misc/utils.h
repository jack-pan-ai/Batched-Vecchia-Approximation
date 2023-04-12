#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>

bool fileExists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

location *load_dist_CSV(const char* filename, int n)
{
    if (!fileExists(filename)) {
        std::cout << "File does not exist" << std::endl;
        exit(0);
    }
    location *loc = (location *) malloc(sizeof(location));
    loc->x = (double* ) malloc(n * sizeof(double));
    loc->y = (double* ) malloc(n * sizeof(double));
    loc->z = NULL;
    std::ifstream file(filename);
    std::string line;
    // std::getline(file, line); // skip header line
    // int numLines = 0;
    // while (std::getline(file, line))
    // {
    //     ++numLines;
    // }
    // file.clear();
    // file.seekg(0, std::ios::beg);
    // std::getline(file, line); // skip header line
    // loc.z = new double[numLines];  // assuming z is also present in the file
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
void load_obs_CSV(const char* filename, int n, T* obs)
{
    if (!fileExists(filename)) {
        std::cout << "File does not exist" << std::endl;
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

#endif