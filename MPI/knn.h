#ifndef PRIR_KNN_H
#define PRIR_KNN_H

#include "Service.h"

class knn {

public:
    knn();
    ~knn();
    double euclideanDistance(wineDataSet *a, wineDataSet *b);
    void runAlgorithm(std::vector<wineDataSet *> *training, std::vector<wineDataSet *> *test);
    double showAccuracy(std::vector<wineDataSet *> *training);
};

#endif //PRIR_KNN_H
