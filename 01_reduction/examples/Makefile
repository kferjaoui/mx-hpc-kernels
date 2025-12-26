NVCC      ?= nvcc
CXX       ?= g++
STD        = -std=c++20
ARCH      ?= sm_89  # for my RTX 4070 (adjust to your GPU architecture)

CXXFLAGS  ?= -O3 $(STD)

ROOT_DIR  := ../..
SRC_DIR   := src
INC       := -Iinclude -I$(ROOT_DIR)/common

NVCCFLAGS := $(CXXFLAGS) -arch=$(ARCH) $(INC)
LDFLAGS   :=

#sources
GPU_SRCS := $(SRC_DIR)/bench_dot.cu $(SRC_DIR)/dot_kernels.cu
GPU_OBJS := $(GPU_SRCS:.cu=.o)

#binaries
GPU_BIN := bench_dot

all: $(GPU_BIN)

#linking
$(GPU_BIN): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

#compile rules
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(GPU_OBJS) $(GPU_BIN)

.PHONY: all clean
