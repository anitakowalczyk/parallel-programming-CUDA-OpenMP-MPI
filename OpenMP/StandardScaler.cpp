#include "StandardScaler.h"
#include <omp.h>

double StandardScaler::calculateStandardDeviation(std::vector<mnistDataSet *> *mnistData, int col, int rows, double mean) {
    double standardDeviation = 0;
    #pragma omp parallel for shared(standardDeviation, mean, rows, col)
    for (int row = 0; row < rows; row++) {
        auto temp = mnistData->at(row)->getFeatures()->at(col) - mean;
        standardDeviation += temp;
    }
    return (standardDeviation/rows)-1;
}

double StandardScaler::calculateMean(std::vector<mnistDataSet *> *mnistData, int col, int rows) {
    double mean = 0;
    #pragma omp parallel for shared(mnistData, mean, rows, col)
    for (int row = 0; row < rows; row++) {
        auto temp = mnistData->at(row)->getFeatures()->at(col);
        mean += temp;
    }
    return mean/rows;
}

std::vector<mnistDataSet *> *StandardScaler::runStandardization(Service *service) {
    auto dataset = service->getAllData();
    int rows = dataset->size();
    int columns = dataset->at(0) -> getFeatures()->size();
    double* mean = new double[columns];
    double* std = new double[columns];

    #pragma omp parallel for shared(mean, std, columns, dataset)
    for(int i = 0; i < columns; i++) {
        mean[i] = calculateMean(dataset, i, rows);
        std[i] = calculateStandardDeviation(dataset, i, rows, mean[i]);
    }

    std::vector<mnistDataSet *> *normalizedData = new std::vector<mnistDataSet *>;
    for(int row = 0; row < rows; row++) {
        mnistDataSet *dataSet = new mnistDataSet();
        for(int col = 0; col < columns; col++) {
            double value = (dataset->at(row)->getFeatures()->at(col) - mean[col])/std[col];
            dataSet->addToFeatures(value);
        }
        normalizedData->push_back(dataSet);
    }
    service->writeFeaturesToCsv(normalizedData);

    return normalizedData;
}

int main(int argc, char* argv[])
{
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    Service *service = new Service();
    std::string path(argv[1]);

    service->readFeatures(path);
    double start = omp_get_wtime();
    StandardScaler *standardScaler = new StandardScaler();
    auto scaledWithStandardScaler = standardScaler->runStandardization(service);

    double end = omp_get_wtime();
    double result = end - start;
    printf("Algorithm took: %f (seconds).\n", result);
}
