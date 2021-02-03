#ifndef PRIR_MinMax_H
#define PRIR_MinMax_H

#include "Service.h"

class MinMax {
public:
    MinMax();
    ~MinMax();
    std::vector<mnistDataSet *> *runMinMaxNormalization(Service *service);
    double calculateMax(std::vector<mnistDataSet *> *data, int cols, int rows);
    double calculateMin(std::vector<mnistDataSet *> *data, int cols, int rows);
};

#endif //PRIR_MinMax_H
