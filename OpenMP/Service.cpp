#include "Service.h"
#include "knn.h"
#include <stdlib.h>
#include <sstream>
#include <cstdio>
#include "MinMax.h"

Service::Service()
{
    allData = new std::vector<mnistDataSet *>;
    testData = new std::vector<mnistDataSet *>;
    trainingData = new std::vector<mnistDataSet *>;
}
Service::~Service() = default;

uint32_t Service::getLittleEndian(const unsigned char* bytes) {
    return (uint32_t ) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

uint32_t Service::replaceEndian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void Service::readFeatures(const std::string &path) {
    FILE *file = fopen(path.c_str(), "r");
    uint32_t header[4];
    if(file) {
        for (unsigned int & i : header) {
            unsigned char bytes[4];
            fread(bytes, sizeof(bytes), 1, file);
            i = getLittleEndian(bytes);
        }

        int size = 10000;
        for( int i = 0; i < size; ++i) {
            mnistDataSet *dataSet = new mnistDataSet();
            uint8_t feature[1];
            for(int j=0; j < header[2] * header[3]; ++j) {
                fread(feature, sizeof(feature), 1, file);
                dataSet->addToFeatures((double) feature[0]);
            }
            allData->push_back(dataSet);
        }
    }
}

void Service::readLabels(const std::string& filePath)
{
    std::ifstream label_file(filePath, std::ios::in | std::ios::binary);
    uint32_t magic;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = replaceEndian(magic);
    uint32_t labelsCount;
    label_file.read(reinterpret_cast<char*>(&labelsCount), 4);
    labelsCount = replaceEndian(labelsCount);

    FILE *file = fopen(filePath.c_str(), "rb");
    if(file)
    {
        int size = 10000;
        for(int i = 0; i < size; ++i)
        {
            int8_t label[1];
            fread(label, sizeof(label), 1, file);
            allData->at(i)->setLabel(label[0]);
        }
    }
}

void Service::writeFeaturesToCsv(std::vector<mnistDataSet *> *dataset){
    std::ofstream file("data.csv");

    auto dataSize = dataset->size();
    for(int i = 0; i < dataSize; ++i)    {
        auto featuresSize = dataset->at(0)->getFeatures()->size();
        for(int j = 0; j < featuresSize; ++j)
        {
            auto value = dataset->at(i)->getFeatures()->at(j);
            file << value;
            if (!(j != dataset->size() - 1)) continue;
            file << ",";
        }
        file << "\n";
    }
    file.close();
}

void Service::readFeaturesFromCsv(std::string filename) {
    std::ifstream featuresFile(filename);
    std::string line;

    while(std::getline(featuresFile, line))
    {
        std::stringstream ss(line);
        int column = 0;
        mnistDataSet *dataSet = new mnistDataSet();
        double value;
        while(ss >> value){
            dataSet->addToFeatures(value);
            if(ss.peek() == ',') ss.ignore();
            column++;
        }
        allData->push_back(dataSet);
    }
    featuresFile.close();
    readLabels("train-labels.idx1-ubyte");
    splitForTestAndTrainingData();
}

std::vector<mnistDataSet *> * Service::getTrainingData() {
    return trainingData;

}
std::vector<mnistDataSet *> * Service::getTestData() {
    return testData;
}

std::vector<mnistDataSet *> *Service::getAllData() {
    return allData;
}

void Service::splitForTestAndTrainingData()
{
    int counter = 0;
    int allDataSetSize = allData->size();
    int trainSize = allDataSetSize * 0.8;
    std::unordered_set<int> indexesInUse;
    while(++counter < trainSize)
    {
        int index = rand() % allDataSetSize;
        if (!(indexesInUse.find(index) == indexesInUse.end()))
            continue;
        trainingData -> push_back(allData->at(index));
        indexesInUse.insert(index);
    }

    counter = 0;
    int testSize = allDataSetSize - trainSize;
    while(++counter < testSize)
    {
        auto index = rand() % allDataSetSize;
        if (!(indexesInUse.find(index) == indexesInUse.end()))
            continue;
        testData -> push_back(allData->at(index));
        indexesInUse.insert(index);
    }
}