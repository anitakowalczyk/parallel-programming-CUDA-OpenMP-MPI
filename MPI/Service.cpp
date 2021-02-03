#include "Service.h"
#include "knn.h"
#include <stdlib.h>
#include <sstream>
#include <cstdio>
#include "MinMax.h"
#include <vector>

Service::Service()
{
    allData = new std::vector<wineDataSet *>;
    testData = new std::vector<wineDataSet *>;
    trainingData = new std::vector<wineDataSet *>;
}
Service::~Service() = default;

void Service::writeFeaturesToCsv(double *dataset, int rows, int columns){
    std::ofstream file("data.csv");

    int counter = 0;
    for(int i = 0; i < rows; ++i)    {
        for(int j = 0; j < columns; ++j)
        {
            auto value = dataset[counter];
            counter++;
            file << value;
            if (!(j != columns)) continue;
            file << ",";
        }
        file << "\n";
    }
    file.close();
}

void Service::readFeaturesFromCsv(std::string filename) {
    std::ifstream featuresFile(filename);
    std::string line;
    std::getline(featuresFile, line);
    int countColumns = countWords(line, ',') - 1;

    while(std::getline(featuresFile, line))
    {
        std::stringstream ss(line);
        int column = 0;
        wineDataSet *dataSet = new wineDataSet();
        double value;
        while(ss >> value){
            if (column != countColumns) {
                dataSet->addToFeatures(value);
            } else {
                dataSet->setLabel(value);
            }
            column++;
            if(ss.peek() == ',') ss.ignore();
        }

        allData->push_back(dataSet);
    }
    featuresFile.close();
    splitForTestAndTrainingData();
}

std::vector<wineDataSet *> * Service::getTrainingData(int np, int myRank) {
    np--;
    myRank--;
    int nMin = trainingData->size()/(np);
    int start = nMin * myRank;
    int end = start + nMin;

    if (np == myRank) {
        std::vector<wineDataSet *> *test = new std::vector<wineDataSet *>(trainingData->begin() + start, trainingData->end());
        return test;
    } else
       { std::vector<wineDataSet *> *test = new std::vector<wineDataSet *>(trainingData->begin() + start, trainingData->begin() + end);;

        return test;
}
}
std::vector<wineDataSet *> * Service::getTestData() {
    return testData;
}

std::vector<wineDataSet *> *Service::getAllData() {
    return allData;
}

void Service::splitForTestAndTrainingData()
{
    int counter = 0;
    int allDataSetSize = allData->size();
    int trainSize = allDataSetSize * 0.8;
    int testSize = allDataSetSize - trainSize;
    std::unordered_set<int> indexesInUse;

    while(++counter < testSize)
    {
        testData -> push_back(allData->at(counter));
    }

    counter = 0;
    while(++counter < trainSize)
    {
        trainingData -> push_back(allData->at(counter));
    }
}

int Service::countWords(const std::string& text, char delimiter) {
    std::stringstream stream(text);
    std::string temp;
    int counter = 0;
    while(getline(stream, temp, delimiter)) {
        counter++;
    }

    return counter;
}