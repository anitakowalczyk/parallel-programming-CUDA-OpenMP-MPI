#include <fstream>
#include <sstream>
#include <vector>
#include "math.h"
#include <limits>
#include <float.h>

__device__
double calculateMin(double *data, int col, int rows, int columns) {
    double calculatedMin = DBL_MAX;
    for (int i = 0; i < rows * columns; i += columns) {
        auto temp = data[i];
        if (i == 0) {
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

__device__
double calculateMax(double *data, int col, int rows, int columns) {
    double calculatedMax = DBL_MIN;
    for (int i = 0; i < rows * columns; i += columns) {
        auto temp = data[i];
        if (temp > calculatedMax) {
            calculatedMax = temp;
        }
    }
    return calculatedMax;
}

__global__
void runMinMaxNormalization(double* data, int rows, int columns) {
    double *mins = new double[columns];
    double *maxs = new double[columns];

    for(int col = 0; col < columns; col++) {
        maxs[col] = calculateMax(data, col, rows, columns);
        mins[col] = calculateMin(data, col, rows, columns);
    }

    int stride = blockDim.x * gridDim.x;
    int columnsCounter = 0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * columns; i += stride) {
        if (columnsCounter + 1 == columns) {
            columnsCounter = 0;
        }
        else {
            double value = (data[i]-mins[columnsCounter])/(maxs[columnsCounter]-mins[columnsCounter]);
            data[i] = value;
            columnsCounter++;
        }
    }
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
void writeFeaturesToCsv(double *output, int rows, int columns){
    std::ofstream file("data.csv");

    int size = rows * columns;
    int columnsCounter = 0;
    for(int i = 0; i < size; ++i)    {
        double value = output[i];
        file << value;
        if (columnsCounter + 1 != columns) {
            file << ",";
            columnsCounter++;

        } else {
            file << "\n";
            columnsCounter = 0;
        }
    }

    file.close();
}


int main(int argc, char* argv[]) {
    int rows, columns;
    std::string fileName("./winequality-white.csv");
    countRowsAndColumns(fileName, &rows, &columns);

    double *input = new double[rows * columns];
    readFeaturesFromCsv(fileName, input, rows, columns);

    double *normalized;
    cudaMalloc((void **) &normalized, rows * columns * sizeof(double));
    cudaMemcpy(normalized, input, rows * columns * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime);

    runMinMaxNormalization<<<1, 1>>>(normalized, rows, columns);

    float resultTime;
    cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&resultTime, startTime, stopTime);

    printf("Algorithm took: %f (ms)\n", resultTime);

    cudaMemcpy(input, normalized, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(normalized);

    writeFeaturesToCsv(input, rows, columns);
}