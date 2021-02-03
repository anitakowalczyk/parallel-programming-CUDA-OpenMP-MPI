FLAGS := -O3 -fopenmp -m64 -std=c++14
COMPILE_COMMAND := g++-10 $(FLAGS)

all: knn MinMax StandardScaler

knn: knn.cpp
	$(COMPILE_COMMAND) -o knn knn.cpp Service.cpp mnistDataSet.cpp

MinMax: MinMax.cpp
	$(COMPILE_COMMAND) -o MinMax MinMax.cpp Service.cpp mnistDataSet.cpp

StandardScaler: StandardScaler.cpp
	$(COMPILE_COMMAND) -o StandardScaler StandardScaler.cpp Service.cpp mnistDataSet.cpp