#include "mnistDataSet.h"

mnistDataSet::mnistDataSet() {
    featureVector = new std::vector<double>;
}

mnistDataSet::~mnistDataSet() = default;

void mnistDataSet::addToFeatures(double val) {
    featureVector->push_back(val);
}

void mnistDataSet::setLabel(int val) {
    label = val;
}

int mnistDataSet::getLabel() const {
    return label;
}

int mnistDataSet::getPredictedLabel() {
    return predictedLabel;
}

void mnistDataSet::setPredictedLabel(int val) {
    predictedLabel = val;
}

std::vector<double> *mnistDataSet::getFeatures() {
    return featureVector;
}