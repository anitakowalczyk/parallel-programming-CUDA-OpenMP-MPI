#include "wineDataSet.h"

wineDataSet::wineDataSet() {
    featureVector = new std::vector<double>;
}

wineDataSet::~wineDataSet() = default;

void wineDataSet::addToFeatures(double val) {
    featureVector->push_back(val);
}

void wineDataSet::setLabel(int val) {
    label = val;
}

int wineDataSet::getLabel() const {
    return label;
}

int wineDataSet::getPredictedLabel() {
    return predictedLabel;
}

void wineDataSet::setPredictedLabel(int val) {
    predictedLabel = val;
}

std::vector<double> *wineDataSet::getFeatures() {
    return featureVector;
}