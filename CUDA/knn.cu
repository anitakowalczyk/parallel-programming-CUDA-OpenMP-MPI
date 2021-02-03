#include "math.h"
#include <fstream>
#include <sstream>
#include <float.h>

__device__
double euclideanDistance(double *training, double *test, int trainingIndexStart, int testIndexStart, int size) {
    double temporary = 0;
    for(unsigned i = 0; i < size; i++) {
        temporary += (training[trainingIndexStart + i] - training[testIndexStart + i]) * (training[trainingIndexStart + i] - training[testIndexStart + i]);
    }

    return sqrt(temporary);
}

__global__
void runAlgorithm(double* result, double *training, double *test, int rowsTraining, int rowsTest, int columns) {
    int currentTrainingRow = 0;
    int minDistanceLabel = 0;
    double minDistance = DBL_MAX;
    int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rowsTraining * columns; i += stride, currentTrainingRow++) {
        for (int j = 0; j < rowsTest * columns; j += columns) {
            double distance = euclideanDistance(training, test, i, j, columns-1);
            if (distance < minDistance) {
                minDistance = distance;
                minDistanceLabel = test[j+columns-1];
            }
        }
        minDistance = DBL_MAX;
        result[currentTrainingRow] = minDistanceLabel;
    }
}

__host__
double showAccuracy(double *data, double* predictions, int rows, int columns) {
    double count = 0;
    int row = 0;
    for(int i = columns-1; i < rows * columns; i += columns, row++) {
        if (data[i] == predictions[row]) {
            count++;
        }
    }
    return (count / rows) * 100;
}

__host__
int countWords(const std::string& text, char delimiter) {
    std::stringstream stream(text);
    std::string temp;
    int counter = 0;
    while(getline(stream, temp, delimiter)) {
        counter++;
    }

    return counter;
}

__host__
void countRowsAndColumns(std::string filename, int* rows, int* columns) {
    std::ifstream featuresFile(filename);
    std::string line;
    std::getline(featuresFile, line);

    *columns = countWords(line, ',');

    *rows = 1;
    while (std::getline(featuresFile, line))
        (*rows)++;

    featuresFile.close();
}

__host__
void readFeaturesFromCsv(std::string filename, double* result, int rows, int columns) {
    std::ifstream featuresFile(filename);
    std::string line;
    int i = 0;

    while(std::getline(featuresFile, line))
    {
        std::stringstream ss(line);
        double value;
        while(ss >> value){
            result[i] = value;
            if(ss.peek() == ',') ss.ignore();
            i++;
        }
    }
    featuresFile.close();
}

__host__
void splitForTestAndTrainingData(double *data, int columns, double *training, double*test, int rowsTraining, int rowsTest)
{
    int counter = 0;
    int dataCounter = 0;
    while(++counter < rowsTest)
    {
        for(int j = 0; j < columns; j++, dataCounter++)
            test[dataCounter] = data[dataCounter];
    }

    counter = 0;
    int i = 0;
    while(++counter < rowsTraining)
    {
        for(int j = 0; j < columns; j++, dataCounter++, i++)
            training[i] = data[dataCounter];
    }
}

int main(int argc, char* argv[]) {
    int rows, columns;
    std::string fileName("./data.csv");
    countRowsAndColumns(fileName, &rows, &columns);

    double *input = new double[rows * columns];
    readFeaturesFromCsv(fileName, input, rows, columns);

    int rowsTraining = rows * 0.8;
    int rowsTest = rows - rowsTraining;

    double *test = new double[rowsTest*columns];
    double *training = new double[rowsTraining*columns];
    double *predictions = new double[rowsTraining * columns];
    double *cudaTraining;
    double *cudaTest;
    double *cudaPrediction;  

    splitForTestAndTrainingData(input, columns, training, test, rowsTraining, rowsTest);

    cudaMalloc((void **) &cudaTraining, rowsTraining * columns * sizeof(double));
    cudaMemcpy(cudaTraining, training, rowsTraining * columns * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &cudaTest, rowsTest * columns * sizeof(double));
    cudaMemcpy(cudaTest, test, rowsTest * columns * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &cudaPrediction, rowsTraining * sizeof(double));
    cudaMemcpy(cudaPrediction, predictions, rowsTraining * sizeof(double), cudaMemcpyHostToDevice);
    
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime);

    runAlgorithm<<<1, 1>>>(cudaPrediction, cudaTraining, cudaTest, rowsTraining, rowsTest, columns);

    float resultTime;
    cudaEventRecord(stopTime);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&resultTime, startTime, stopTime);

    cudaMemcpy(predictions, cudaPrediction, rowsTraining * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(cudaTraining);
    cudaFree(cudaTest);
    cudaFree(cudaPrediction);

    printf("Algorithm took: %f (ms)\n", resultTime);

    double accuracy = showAccuracy(training, predictions, rowsTraining, columns);
    printf("Accuracy: %f\n", accuracy);
}