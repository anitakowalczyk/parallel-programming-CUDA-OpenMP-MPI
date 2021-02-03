#include "StandardScaler.h"
#include "math.h"
#include <mpi.h>

double StandardScaler::calculateStandardDeviation(std::vector<wineDataSet *> *mnistData, int col, int rows, double mean) {
    double standardDeviation = 0;
    for (int row = 0; row < rows; row++) {
        auto temp = (mnistData->at(row)->getFeatures()->at(col) - mean);
        standardDeviation += (temp * temp);
    }
    return sqrt(standardDeviation/rows);
}
 
double StandardScaler::calculateMean(std::vector<wineDataSet *> *mnistData, int col, int rows) {
    double mean = 0;
    for (int row = 0; row < rows; row++) {
        mean += mnistData->at(row)->getFeatures()->at(col);
    }
    return mean/rows;
}
 
double* StandardScaler::runStandardization(Service *service, int np, int myRank) {
    np--;
    myRank--;
    auto *dataset = service->getAllData();

    int rows = dataset->size();
    int columns = dataset->at(0)->getFeatures()->size();
    int dataSize = rows/np;

    int start = dataSize * myRank;
    int end = start + dataSize;
    if (end < rows && np == myRank+1)
        end++;

    std::vector<double> * mean = new std::vector<double>;
    std::vector<double> * std = new std::vector<double>;
 
    for(int i = 0; i < columns; i++) {
        mean->push_back(calculateMean(dataset, i, rows));
        std->push_back(calculateStandardDeviation(dataset, i, rows, mean -> at(i)));
    }

    int resultSize = (end-start)*(columns+1);
    double *result = new double[resultSize];
    int i = 0;
    for(int row = start; row < end; row++) {
        for(int col = 0; col < columns; col++) {
            double value = dataset->at(row)->getFeatures()->at(col);
            if(std -> at(col) != 0){
                value -= mean->at(col);
                value /= std->at(col);
            }
            result[i] = value;
            i++;
        }
        result[i] = dataset->at(row)->getLabel();
        i++;
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    int np;
    int myrank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    Service *service = new Service();
    service->readFeaturesFromCsv("./winequality-white.csv");
    int rows = service->getAllData()->size();
    int nMin = service->getAllData()->size()/(np-1);
    int columns = service->getAllData()->at(0)->getFeatures()->size()+1;
    int size = rows*columns;
    MPI_Status Stat;
    
    double algorithmTime;
    if (myrank != 0) {
        StandardScaler *standardScaler = new StandardScaler();

        double start = MPI_Wtime();
        double* algorithmResult = standardScaler->runStandardization(service, np, myrank);
        double end = MPI_Wtime();
        
        MPI_Send(algorithmResult, sizeof(double) * nMin * columns, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        algorithmTime = end - start;
    } 
    else {
        double *result = new double[size];
        for (int id_process = 1; id_process < np; id_process++) 
        {
            MPI_Recv(result, sizeof(double) * nMin * columns, MPI_BYTE, id_process, 1, MPI_COMM_WORLD, &Stat);
        }

        service->writeFeaturesToCsv(result,  rows,  columns);
    }

    double *algorithmTimes = NULL;
    if (myrank == 0) {
        algorithmTimes = (double *)malloc(sizeof(double) * np);
    }
    MPI_Gather(&algorithmTime, 1, MPI_DOUBLE, algorithmTimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double tempTime = 0;
    if (myrank == 0) {
        for(int i=0; i<np;++i) {
            tempTime += algorithmTimes[i];
        }
        tempTime /= (np-1);
        printf("Algorithm took: %f (seconds)\n", tempTime);
    }

    MPI_Finalize();
}