NVCC = nvcc
CXX  = g++
STD  = -std=c++20
ARCH = sm_89  # for my RTX 4070 (adjust to your GPU architecture)
CXXFLAGS = -O3 $(STD)
ROOT_DIR := ../..
SRC_DIR  := ../src
TEST_DIR := .
INC      := -I../include -I$(ROOT_DIR)/common
NVCCFLAGS := $(CXXFLAGS) -arch=$(ARCH) $(INC)
CPPFLAGS := $(CXXFLAGS) $(INC)

#sources
GPU_SRCS := $(TEST_DIR)/reduce_cuda.test.cu $(SRC_DIR)/cuda/reduce_baseline.cu
GPU_OBJS := $(GPU_SRCS:.cu=.o)

CPU_SRCS := $(TEST_DIR)/reduce_cpu.test.cpp
CPU_OBJS := $(CPU_SRCS:.cpp=.o)

#binaries
GPU_BIN := reduce_cuda.test
all: $(GPU_BIN)

CPU_BIN := reduce_cpu.test
all: $(CPU_BIN)

#linking
$(GPU_BIN): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CPU_BIN): $(CPU_OBJS)
	$(CXX) $(CPPFLAGS) $^ -o $@

#compile rules
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ../src/cpu/%.o: ../src/cpu/%.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(GPU_OBJS) $(GPU_BIN) $(CPU_OBJS) $(CPU_BIN)