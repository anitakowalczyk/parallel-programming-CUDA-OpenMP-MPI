COMPILE_COMMAND := nvcc

all: knn MinMax StandardScaler

knn: knn.cu
	$(COMPILE_COMMAND) knn.cu -o knn

MinMax: MinMax.cu
	$(COMPILE_COMMAND) MinMax.cu -o MinMax

StandardScaler: StandardScaler.cu
	$(COMPILE_COMMAND) StandardScaler.cu -o StandardScaler