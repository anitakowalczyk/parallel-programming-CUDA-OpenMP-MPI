#ifndef PRIR_STANDARDSCALER_H
#define PRIR_STANDARDSCALER_H

#include "Service.h"
#include <vector>

class StandardScaler {
public:
    StandardScaler() = default;
    ~StandardScaler() = default;
    double* runStandardization(Service * service, int np, int myRank);
    double calculateStandardDeviation(std::vector<wineDataSet *> *data, int col, int rows, double mean);
    double calculateMean(std::vector<wineDataSet *> *data, int col, int n_row);
};

#endif //PRIR_STANDARDSCALER_H