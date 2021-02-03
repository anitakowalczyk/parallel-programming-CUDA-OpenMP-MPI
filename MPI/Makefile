# export OMPI_CXX=G++-10
COMPILE_COMMAND := mpicxx

all: knn MinMax StandardScaler

knn: knn.cpp
	$(COMPILE_COMMAND) knn.cpp Service.cpp wineDataSet.cpp -o knn

MinMax: MinMax.cpp
	$(COMPILE_COMMAND) MinMax.cpp Service.cpp wineDataSet.cpp -o MinMax

StandardScaler: StandardScaler.cpp
	$(COMPILE_COMMAND) StandardScaler.cpp Service.cpp wineDataSet.cpp -o StandardScaler