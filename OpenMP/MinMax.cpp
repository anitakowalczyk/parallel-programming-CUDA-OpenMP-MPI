#include <omp.h>
#include "MinMax.h"

MinMax::MinMax() = default;
MinMax::~MinMax() = default;

double MinMax::calculateMin(std::vector<mnistDataSet *> *data, int col, int rows)
{
    double calculatedMin = 0;
    #pragma omp parallel for shared(data, rows, calculatedMin)
    for (int row = 0; row < rows; row++) {
        auto temp = data->at(row)->getFeatures()->at(col);
        if (row == 0) {
            calculatedMin = temp;
        }
        else {
            if (temp < calculatedMin) {
                calculatedMin = temp;
            }
        }

    }
    return calculatedMin;
}

double MinMax::calculateMax(std::vector<mnistDataSet *> *data, int col, int rows) {
    double calculatedMax = std::numeric_limits<double>::min();
    #pragma omp parallel for shared(data, rows, calculatedMax)
    for (int row = 0; row < rows; row++) {
        auto temp = data->at(row)->getFeatures()->at(col);
        if (temp > calculatedMax) {
            calculatedMax = temp;
        }
    }
    return calculatedMax;
}

std::vector<mnistDataSet *> *MinMax::runMinMaxNormalization(Service *service)
{
    auto dataset = service->getAllData();
    int rows = dataset->size();
    int columns = dataset->at(0)->getFeatures()->size();
    double *mins = new double[columns];
    double *maxs = new double[columns];

    #pragma omp parallel for shared(maxs, mins, columns, dataset)
    for(int col = 0; col < columns; col++) {
        maxs[col] = calculateMax(dataset, col, rows);
        mins[col] = calculateMin(dataset, col, rows);
    }

    std::vector<mnistDataSet *> *normalizedData = new std::vector<mnistDataSet *>;
    for(int row = 0; row < rows; row++) {
        mnistDataSet *dataSet = new mnistDataSet();
        for(int col = 0; col < columns; col++) {
            double value = (dataset->at(row)->getFeatures()->at(col)-mins[col])/(maxs[col]-mins[col]);
            dataSet-> addToFeatures(value);
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
    MinMax *algorithm = new MinMax();
    auto normalizedWithMinMax = algorithm->runMinMaxNormalization(service);

    double end = omp_get_wtime();
    double result = end - start;
    printf("Algorithm took: %f (seconds).\n", result);
}
