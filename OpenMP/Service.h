#ifndef PRIR_Service_H
#define PRIR_Service_H

#include <fstream>
#include "stdint.h"
#include <string>
#include <map>
#include <unordered_set>
#include "mnistDataSet.h"
#include <vector>

class Service {
    std::vector<mnistDataSet *> *trainingData;
    std::vector<mnistDataSet *> *testData;
    std::vector<mnistDataSet *> *allData;

public:
    Service();
    ~Service();
    void readFeatures(const std::string &path);
    std::vector<mnistDataSet *> *getTrainingData();
    std::vector<mnistDataSet *> *getTestData();
    std::vector<mnistDataSet *> *getAllData();
    void readLabels(const std::string& path);
    void splitForTestAndTrainingData();
    void writeFeaturesToCsv(std::vector<mnistDataSet *> *dataSet);
    void readFeaturesFromCsv(std::string file);
    uint32_t getLittleEndian(const unsigned char* bytes);
    uint32_t replaceEndian(uint32_t val);
};

#endif //PRIR_Service_H