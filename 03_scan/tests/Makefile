NVCC = nvcc
CXX  = g++
STD      := -std=c++20
OPT      := -O3 -march=native -DNDEBUG
CXXFLAGS ?=
CXXFLAGS += $(STD) $(OPT) -pthread -MMD -MP

ARCH = -arch=sm_89  # for my RTX 4070 (adjust to your GPU architecture)
NVCCFLAGS := -Xptxas=-v $(STD) $(ARCH)
# # NVCCFLAGS +=  -rdc=true

ROOT_DIR := ../..
SRC_DIR  := ../src
TEST_DIR := .
COMMON   := $(ROOT_DIR)/common
MXINC    := $(COMMON)/00_mx/include
INC      := -I../include -I$(COMMON) -I$(MXINC)

#sources
GPU_SRCS := $(TEST_DIR)/scan_cuda.test.cu $(SRC_DIR)/cuda/scan_cuda.cu
GPU_OBJS := $(GPU_SRCS:.cu=.o)

CPU_SRCS := $(TEST_DIR)/scan_cpu.test.cpp
CPU_OBJS := $(CPU_SRCS:.cpp=.o)
DEPS	 := $(CPU_OBJS:.o=.d)

#binaries
GPU_BIN := scan_cuda.test
all: $(GPU_BIN)

CPU_BIN := scan_cpu.test
all: $(CPU_BIN)

#linking
$(GPU_BIN): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INC) $^ -o $@

$(CPU_BIN): $(CPU_OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@

#CUDA compile rules
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

#CPP compile rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(GPU_OBJS) $(GPU_BIN) $(CPU_OBJS) $(CPU_BIN) $(DEPS)