#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


#define pow_e (2.71828182845904)
#define PI (3.141592653589793)
#define earthRadiusKm (6371.0)

// used for nearest neighbor selection
double calEucDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}


/**
 * This function converts decimal degrees to radians
 * @param deg decimal degree
 */
static double deg2rad(double deg) {
	return (deg * PI / 180);
}

/**
 * This function converts radians to decimal degrees
 * @param rad radians
 */
static double rad2deg(double rad) {
	return (rad * 180 / PI);
}

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @return The distance between the two points in kilometers
 */
static double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
	double lat1r, lon1r, lat2r, lon2r, u, v;
	lat1r = deg2rad(lat1d);
	lon1r = deg2rad(lon1d);
	lat2r = deg2rad(lat2d);
	lon2r = deg2rad(lon2d);
	u = sin((lat2r - lat1r) / 2);
	v = sin((lon2r - lon1r) / 2);
	return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}


void findNearestPoints(double *h_C_conditioning, double *h_C, location* locations_con, location* locations, int l0, int l1, int l2, int k, int i_blcok, int distance_metric) {
    // For example, p(y_{6, 7, 8, 9}|y_{2, 3, 4, 5}) -> p(y1|y2)
    // l0: starting point, 2 in the example, affecting how large conditioning set you will choose
    // l1: the end of conditioning set, e.g., 5
    // l2: the end of conditioned set, e.g., 9
    // l2 - l1, conditioned set, y1
    // l1 - l0, conditioning set, y2, have not been sorted yet
    // k: number of nearest neighbor 
    // i_blcok: ith conditioning set
    // std::vector<std::pair<double, double>> nearestPoints;
    if (k > (l1 - l0)) {
        std::cout << "Not enough points available." << std::endl;
        k = l1 - l0;
    }
    // // Calculate mean of x and y coordinates
    // double meanX = 0.0;
    // double meanY = 0.0;
    // for (int i = l1; i < l2; i++) {
    //     // printf("(%lf, %lf) \n", locations->x[i], locations->y[i]);
    //     meanX += locations->x[i];
    //     meanY += locations->y[i];
    // }
    // meanX /= double(l2 - l1);
    // meanY /= double(l2 - l1);
    // // std::cout << "Nearest Points to (" << meanX << ", " << meanY << "):" << std::endl;

    // used to modify the conditioning set without affecting original values;
    double distance;
    // indices redefining 
    std::vector<int> indices(l1 - l0);
    for (int j = l0; j < l1; j++) {
        indices[j - l0] = j;
        // printf("%d \n", indices[j - l0]);
    }

    // some points in the block needs more than 1 nearest neighbor
    int bs = l2 - l1;
    int cs_all = l1 - l0;
    int cs = k;
    // each point has cs/bs neighbors and the first cs % bs has one extra than others
    std::vector<int> num_nn(bs, cs / bs);
    for (int i = 0; i < cs % bs ; ++i){
        num_nn[i] += 1;
    }
    std::vector<int> accumulatedSum(num_nn.size());
    int sum = 0;
    for (int i = 0; i < num_nn.size(); ++i) {
        sum += num_nn[i];
        accumulatedSum[i] = sum;
        // fprintf(stderr, "%d %d %d %d \n",i, num_nn[i], sum, accumulatedSum[i]);
    }
    // fprintf(stderr, "=====================\n");

    // for each point in the block, calculate its nearest neighbor
    for (size_t i = l1; i < l2; i++){
        // calculate pairwise distance (x1|x1_con)
        std::vector<double> distances;
        for (size_t j = l0; j < l1; j++) {
            if (indices[j - l0] == -1){
                distance = 999999999.;
            }else{
                if (distance_metric == 1){
                    distance = distanceEarth(locations->x[j], locations->y[j],
                                                locations->x[i], locations->y[i]);
                }else{
                    distance = calEucDistance(locations->x[j], locations->y[j],
                                                locations->x[i], locations->y[i]);
                }
            }
            // note that the length of distance is (l1 - l0)
            // printf("%lf \n", distance);
            distances.push_back(distance);
        }

        // Create a vector of pairs to associate each index with its corresponding distance
        std::vector<std::pair<int, double>> indexDistancePairs;
        for (size_t j = 0; j < indices.size(); ++j) {
            indexDistancePairs.push_back({indices[j], distances[j]});
        }

        // Define a custom comparator to sort based on the distance value
        auto customComparator = [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second < b.second;
        };

        // Sort the vector of pairs based on the distance value
        std::sort(indexDistancePairs.begin(), indexDistancePairs.end(), customComparator);

        
        // for (int j = l0; j < l1; j++) {
        //     fprintf(stderr, "%d \n", indices[j - l0]);
        //     fprintf(stderr, "%lf \n", distances[j - l0]);
        // }
        // fprintf(stderr, "-------\n");
        // // Update the indices vector with the sorted values
        // for (int j = 0; j < indices.size(); ++j) {
        //     indices[j] = indexDistancePairs[j].first;
        // }
        // for (int j = l0; j < l1; j++) {
        //     fprintf(stderr, "%d \n", indexDistancePairs[j].first);
        //     fprintf(stderr, "%lf \n", indexDistancePairs[j].second);
        // }
        // fprintf(stderr, "-------\n");
        // printf("=======(%lf, %lf) =======\n", locations->x[l1], locations->y[l1]);
        for (int j=0; j<num_nn[i-l1]; j++){
            // printf("(%lf, %lf)\n", locations->x[indexDistancePairs[j].first], locations->y[indexDistancePairs[j].first]);
            locations_con->x[cs * (i_blcok + 1) - accumulatedSum[i - l1] + j] = locations->x[indexDistancePairs[j].first];
            locations_con->y[cs * (i_blcok + 1) - accumulatedSum[i - l1] + j] = locations->y[indexDistancePairs[j].first];
            h_C_conditioning[cs * (i_blcok + 1) - accumulatedSum[i - l1] + j] = h_C[indexDistancePairs[j].first];
            // means jth index in indices will be not used anymore 
            indices[indexDistancePairs[j].first] = -1;
        }
        // printf("==============\n");
    }
}

#endif