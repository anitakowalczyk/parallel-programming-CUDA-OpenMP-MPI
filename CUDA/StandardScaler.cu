#include <fstream>
#include <sstream>
#include <vector>
#include "math.h"

__device__
double calculateStandardDeviation(double *data, int col, int rows, int columns, double mean) {
    double standardDeviation = 0;
    for (int i = col; i < rows * columns; i += columns) {
        auto temp = (data[i] - mean);
        standardDeviation += (temp * temp);
    }
    return sqrt(standardDeviation/rows);
}

__device__
double calculateMean(double *data, int col, int rows, int columns) {
    double mean = 0;
    for (int i = col; i < rows * columns; i+= columns) {
        mean += data[i];
    }
    return mean/rows;
}

__global__
void runStandardization(double* data, int rows, int columns) {
    double *mean = new double[columns - 1];
    double *std = new double[columns - 1];

    for(int i = 0; i < columns-1; i++) {
        mean[i] = calculateMean(data, i, rows, columns);
        std[i] = calculateStandardDeviation(data, i, rows, columns, mean[i]);
    }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int columnsCounter = 0;
    for(int i = index; i < rows * columns; i+=stride) {
        double value = data[i];
        if (columnsCounter+1 == columns) {
            data[i] = value;
            columnsCounter = 0;
        }
        else {
            if(std[columnsCounter] != 0) {
                value -= mean[columnsCounter];
                value /= std[columnsCounter];
            }
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

    runStandardization<<<1, 1>>>(normalized, rows, columns);

    float resultTime;
    cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&resultTime, startTime, stopTime);

    printf("Algorithm took: %f (ms)\n", resultTime);

    cudaMemcpy(input, normalized, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(normalized);

    writeFeaturesToCsv(input, rows, columns);
}