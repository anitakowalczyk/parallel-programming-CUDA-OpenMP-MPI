#include "MinMax.h"
#include <mpi.h>

MinMax::MinMax() = default;
MinMax::~MinMax() = default;

double MinMax::calculateMin(std::vector<wineDataSet *> *data, int col, int rows)
{
    double calculatedMin = 0;
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

double MinMax::calculateMax(std::vector<wineDataSet *> *data, int col, int rows) {
    double calculatedMax = std::numeric_limits<double>::min();
    for (int row = 0; row < rows; row++) {
        auto temp = data->at(row)->getFeatures()->at(col);
        if (temp > calculatedMax) {
            calculatedMax = temp;
        }
    }
    return calculatedMax;
}

double* MinMax::runMinMaxNormalization(Service *service, int np, int myRank) {
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

    double *mins = new double[columns];
    double *maxs = new double[columns];

    for(int col = 0; col < columns; col++) {
        maxs[col] = calculateMax(dataset, col, rows);
        mins[col] = calculateMin(dataset, col, rows);
    }

    int resultSize = (end-start)*(columns+1);
    double *result = new double[resultSize];
    int i = 0;
    for(int row = start; row < end; row++) {
        wineDataSet *dataSet = new wineDataSet();
        for(int col = 0; col < columns; col++) {
            double value = (dataset->at(row)->getFeatures()->at(col)-mins[col])/(maxs[col]-mins[col]);
            result[i] = value;
            i++;
        }
        result[i] = dataset->at(row)->getLabel();
        i++;
    }

    return result;
}


int main(int argc, char* argv[])
{
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
        MinMax *algorithm = new MinMax();

        double start = MPI_Wtime();
        double* algorithmResult = algorithm->runMinMaxNormalization(service, np, myrank);
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
