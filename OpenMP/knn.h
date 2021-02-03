#ifndef PRIR_KNN_H
#define PRIR_KNN_H

#include "Service.h"

class knn {

public:
    knn();
    ~knn();
    double euclideanDistance(mnistDataSet *a, mnistDataSet *b);
    void runAlgorithm(std::vector<mnistDataSet *> *training, std::vector<mnistDataSet *> *test);
    void showAccuracy(std::vector<mnistDataSet *> *training);
};

#endif //PRIR_KNN_H
