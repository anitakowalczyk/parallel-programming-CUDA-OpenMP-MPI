#include "knn.h"
#include "math.h"
#include <map>
#include <limits>
#include <omp.h>
#include <string>
#include "Service.h"

knn::knn() = default;
knn::~knn() = default;

double knn::euclideanDistance(mnistDataSet *a, mnistDataSet *b) {
    double temporary = 0;
    #pragma omp parallel for shared(temporary, a, b)
    for(unsigned i = 0; i < a->getFeatures()->size(); i++) {
        temporary += (a->getFeatures()->at(i) - b->getFeatures()->at(i)) * (a->getFeatures()->at(i) - b->getFeatures()->at(i));
    }

    return sqrt(temporary);
}

void knn::runAlgorithm(std::vector<mnistDataSet *> *training, std::vector<mnistDataSet *> *test) {
    int minDistanceIndex;
    double minDistance = std::numeric_limits<double>::max();

    #pragma omp parallel for
    for(int i = 0; i < training->size(); ++i) {
        #pragma omp parallel for
        for (int j = 0; j < test->size(); ++j) {
            double distance = euclideanDistance(training->at(i), test->at(j));
            if (distance < minDistance) {
                minDistance = distance;
                minDistanceIndex = j;
            }
        }
        minDistance = std::numeric_limits<double>::max();
        auto predictedLabel = test->at(minDistanceIndex)->getLabel();
        training->at(i)->setPredictedLabel(predictedLabel);
    }
}

void knn::showAccuracy(std::vector<mnistDataSet *> *training) {
    double count = 0;
    #pragma omp parallel for shared(training, count)
    for(int i = 0; i < training->size(); ++i) {
        auto trainingDataAtI = training->at(i);
        if (trainingDataAtI->getLabel() == trainingDataAtI->getPredictedLabel()) {
            count++;
        }
    }

    double accuracy = (count / training->size()) * 100;
    printf("KNN accuracy isy: %f%%\n", accuracy);
}

int main(int argc, char* argv[])
{
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    Service *service = new Service();
    service->readFeaturesFromCsv("data.csv");
    std::vector<mnistDataSet *> *testData = service->getTestData();
    std::vector<mnistDataSet *> *trainingData = service->getTrainingData();

    knn *algorithm = new knn();
    double start = omp_get_wtime();
    algorithm->runAlgorithm(trainingData, testData);
    double end = omp_get_wtime();
    algorithm->showAccuracy(trainingData);

    double result = end - start;
    printf("Algorithm took: %f (seconds).\n", result);
}