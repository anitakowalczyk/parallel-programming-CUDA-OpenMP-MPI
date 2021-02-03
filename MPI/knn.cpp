#include "knn.h"
#include "math.h"
#include <map>
#include <limits>
#include <string>
#include "Service.h"
#include <mpi.h>

knn::knn() = default;
knn::~knn() = default;

double knn::euclideanDistance(wineDataSet *a, wineDataSet *b) {
    double temporary = 0;
    for(unsigned i = 0; i < a->getFeatures()->size(); i++) {
        temporary += (a->getFeatures()->at(i) - b->getFeatures()->at(i)) * (a->getFeatures()->at(i) - b->getFeatures()->at(i));
    }

    return sqrt(temporary);
}

void knn::runAlgorithm(std::vector<wineDataSet *> *training, std::vector<wineDataSet *> *test) {
    int minDistanceIndex;
    double minDistance = std::numeric_limits<double>::max();

    for(int i = 0; i < training->size(); ++i) {
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

double knn::showAccuracy(std::vector<wineDataSet *> *training) {
    double count = 0;
    for(int i = 0; i < training->size(); ++i) {
        auto trainingDataAtI = training->at(i);
        if (trainingDataAtI->getLabel() == trainingDataAtI->getPredictedLabel()) {
            count++;
        }
    }

    double accuracy = (count / training->size()) * 100;
    return accuracy;
}

int main(int argc, char* argv[]) {
    int np;
    int myrank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    double algorithmResult;
    double algorithmTime;
    if (myrank != 0) {
        Service *service = new Service();
        service->readFeaturesFromCsv("data.csv");
        std::vector<wineDataSet *> *testData = service->getTestData();
        std::vector<wineDataSet *> *trainingData = service->getTrainingData(np, myrank);
        knn *algorithm = new knn();

        double start = MPI_Wtime();
        algorithm->runAlgorithm(trainingData, testData);
        double end = MPI_Wtime();
        algorithmResult = algorithm->showAccuracy(trainingData);
        algorithmTime = end - start;
    }
    double *algorithmResults = NULL;
    double *algorithmTimes = NULL;
    if (myrank == 0) {
        algorithmResults = (double *)malloc(sizeof(double) * np);
        algorithmTimes = (double *)malloc(sizeof(double) * np);
    }
    MPI_Gather(&algorithmResult, 1, MPI_DOUBLE, algorithmResults, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&algorithmTime, 1, MPI_DOUBLE, algorithmTimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double tempResult = 0;
    double tempTime = 0;
    if (myrank == 0) {
        for(int i=0; i<np;++i) {
            tempResult += algorithmResults[i];
            tempTime += algorithmTimes[i];
        }
        tempResult /= (np-1);
        tempTime /= (np-1);
        printf("KNN accuracy isy: %f%%\n", tempResult);
        printf("Algorithm took: %f (seconds).\n", tempTime);
    }

    MPI_Finalize();
}