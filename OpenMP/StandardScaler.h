#ifndef PRIR_STANDARDSCALER_H
#define PRIR_STANDARDSCALER_H

#include "Service.h"
#include <vector>

class StandardScaler {
public:
    StandardScaler() = default;
    ~StandardScaler() = default;
    std::vector<mnistDataSet *> * runStandardization(Service * service);
    double calculateStandardDeviation(std::vector<mnistDataSet *> *data, int col, int rows, double mean);
    double calculateMean(std::vector<mnistDataSet *> *data, int col, int n_row);
};

#endif //PRIR_STANDARDSCALER_H