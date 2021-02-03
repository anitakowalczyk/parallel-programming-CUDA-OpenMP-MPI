#ifndef PRIR_MinMax_H
#define PRIR_MinMax_H

#include "Service.h"

class MinMax {
public:
    MinMax();
    ~MinMax();
    double* runMinMaxNormalization(Service *service, int np, int myRank);
    double calculateMax(std::vector<wineDataSet *> *data, int cols, int rows);
    double calculateMin(std::vector<wineDataSet *> *data, int cols, int rows);
};

#endif //PRIR_MinMax_H
