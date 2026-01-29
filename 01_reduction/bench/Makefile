NVCC := nvcc
CXX := g++

STD      := -std=c++20
OPT      := -O3 -march=native -DNDEBUG
CXXFLAGS := $(STD) $(OPT) -pthread
GPU_ARCH := -arch=sm_89
NVCCFLAGS := $(STD) $(GPU_ARCH) -Xptxas=-v

ROOT_DIR := ../..
COMMON_DIR := $(ROOT_DIR)/common
INCLUDE := -I../include -I$(COMMON_DIR)

TBB_LIBS := -ltbb # Link TBB library for std::reduce(std::execution::par, ...)

SRC_CPU := ./reduce_baseline.bench.cpp 
SRC_CUDA := ../src/cuda/reduce_cuda.cu

OBJ_CPU := $(SRC_CPU:.cpp=.o)
OBJ_CUDA := $(SRC_CUDA:.cu=.o)
OBJ := $(OBJ_CPU) $(OBJ_CUDA)

BIN := reduce.bench
all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@ $(TBB_LIBS)

$(OBJ_CPU): $(SRC_CPU)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $^ -o $@

$(OBJ_CUDA): $(SRC_CUDA)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $^ -o $@

clean:
	rm -rf $(BIN) $(OBJ)