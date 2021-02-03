#include "stdint.h"
#include "stdio.h"
#include <vector>

class wineDataSet {
    std::vector<double> *featureVector;
    int label{};
    int predictedLabel{};

public:
    wineDataSet();
    ~wineDataSet();
    void addToFeatures(double val);
    void setLabel(int val);
    void setPredictedLabel(int val);
    int getPredictedLabel();
    std::vector<double> * getFeatures();
    int getLabel() const;
};