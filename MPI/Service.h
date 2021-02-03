#ifndef PRIR_Service_H
#define PRIR_Service_H

#include <fstream>
#include "stdint.h"
#include <string>
#include <map>
#include <unordered_set>
#include "wineDataSet.h"
#include <vector>

class Service {
    std::vector<wineDataSet *> *trainingData;
    std::vector<wineDataSet *> *testData;
    std::vector<wineDataSet *> *allData;

public:
    Service();
    ~Service();
    std::vector<wineDataSet *> *getTrainingData(int np, int myRank);
    std::vector<wineDataSet *> *getTestData();
    std::vector<wineDataSet *> *getAllData();
    void splitForTestAndTrainingData();
    void writeFeaturesToCsv(double *dataset, int rows, int columns);
    void readFeaturesFromCsv(std::string file);
    int countWords(const std::string& text, char delimiter);
};

#endif //PRIR_Service_H