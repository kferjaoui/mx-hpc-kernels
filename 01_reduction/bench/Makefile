NVCC ?= nvcc
CXX  ?= g++

STD      := -std=c++20
OPT      := -O3 -march=native -DNDEBUG

CXXFLAGS ?=
CXXFLAGS += $(STD) $(OPT) -pthread -MMD -MP

NVCCFLAGS ?=
GPU_ARCH  ?= -arch=sm_89
NVCCFLAGS += $(STD) $(GPU_ARCH) -Xptxas=-v --expt-relaxed-constexpr

# Optional user overrides without editing:
#   make EXTRA_CXXFLAGS="..." EXTRA_NVCCFLAGS="..."
EXTRA_CXXFLAGS ?=
EXTRA_NVCCFLAGS ?=
CXXFLAGS  += $(EXTRA_CXXFLAGS)
NVCCFLAGS += $(EXTRA_NVCCFLAGS)

# If you see: fatal error: cuda_runtime.h: No such file or directory
# set CUDA_HOME to your CUDA install, e.g.
#   make CUDA_HOME=/usr/local/cuda-12.4
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := -I$(CUDA_HOME)/include
CUDA_LIB  := -L$(CUDA_HOME)/lib64

ROOT_DIR   := ../..
COMMON_DIR := $(ROOT_DIR)/common
INCLUDE    := -I../include -I$(COMMON_DIR) $(CUDA_INC)

# TBB for std::execution::par backends (libtbb)
LDLIBS  := -ltbb
LDFLAGS := $(CUDA_LIB)

SRC_CPU  := ./benchReduction.cpp
SRC_CUDA := ../src/cuda/reduce_cuda_profiling.cu

OBJ_CPU  := $(SRC_CPU:.cpp=.o)
OBJ_CUDA := $(SRC_CUDA:.cu=.o)
OBJ      := $(OBJ_CPU) $(OBJ_CUDA)

DEPS := $(OBJ:.o=.d)

BIN := reduce.bench

.PHONY: all clean help
all: $(BIN)

help:
	@echo "Targets:"
	@echo "  all        Build $(BIN)"
	@echo "  clean      Remove build artifacts"
	@echo ""
	@echo "Common fixes:"
	@echo "  If you get 'cuda_runtime.h: No such file or directory':"
	@echo "    make CUDA_HOME=/usr/local/cuda-12.4"
	@echo ""
	@echo "Useful overrides:"
	@echo "  make GPU_ARCH=-arch=sm_90"
	@echo "  make EXTRA_CXXFLAGS='-g' EXTRA_NVCCFLAGS='-lineinfo'"

# Link with nvcc so CUDA runtime gets pulled correctly
$(BIN): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(OBJ_CPU): $(SRC_CPU)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_CUDA): $(SRC_CUDA)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -rf $(BIN) $(OBJ) $(DEPS)

-include $(DEPS)
