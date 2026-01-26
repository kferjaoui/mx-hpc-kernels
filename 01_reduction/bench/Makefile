NVCC := nvcc
STD  := -std=c++20
CXXFLAGS := -O3 $(STD)
GPU_ARCH := -arch=sm_89
NVCCFLAGS := $(CXXFLAGS) $(GPU_ARCH) -Xptxas=-v

ROOT_DIR := ../..
COMMON_DIR := $(ROOT_DIR)/common
INCLUDE := -I../include -I$(COMMON_DIR)

SRC := ./reduce_baseline.bench.cu
OBJ := $(SRC:.cu=.o)

BIN := reduce_baseline.bench
all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@

$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $^ -o $@

clean:
	rm -rf $(BIN) $(OBJ)