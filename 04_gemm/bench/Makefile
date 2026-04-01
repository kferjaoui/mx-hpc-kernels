NVCC 	   = nvcc
CXX        = g++
STD        = -std=c++20
OPT        = -O3 -march=native
DEPFLAGS   := -MMD -MP
WARN       = -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

CXXFLAGS  := $(STD) $(OPT) $(DEPFLAGS) # $(WARN)
LDLIBS    := -pthread -lopenblas
CUDALBS   := -lcublas

ARCH = -arch=sm_89 
NVCCFLAGS := -Xptxas=-v $(STD) $(ARCH) --expt-relaxed-constexpr # -rdc=true
NVCC_DEPFLAGS := -MMD -MP

ROOT           := ../..
COMMON         := $(ROOT)/common
MXINC          := $(COMMON)/00_mx/include
EIGEN_INCLUDE  := $(ROOT)/third_party/eigen
INC      	   := -I../include -I$(COMMON) -I$(MXINC)  -I$(EIGEN_INCLUDE) -DEIGEN_USE_BLAS

CPU_SRCS  := bench_gemm.cpp
CPU_OBJS  := $(CPU_SRCS:.cpp=.o)
CPU_DEPS  := $(CPU_OBJS:.o=.d)
CPU_BINS  := bench_gemm_cpu

GPU_SRCS  := bench_gemm_cuda.cu
GPU_OBJS  := $(GPU_SRCS:.cu=.o)
GPU_DEPS   := $(GPU_OBJS:.o=.d)
GPU_BINS   := bench_gemm_cuda

all: $(CPU_BINS) $(GPU_BINS)

# Linking rules
$(CPU_BINS): $(CPU_OBJS)
	$(CXX) $(CXXFLAGS)  $(INC) $^ $(LDLIBS) -o $@ 

$(GPU_BINS): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INC) $^ $(CUDALBS) -o $@

# Compile rules
%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_DEPFLAGS) $(INC) -c $< -o $@

-include $(CPU_DEPS) $(GPU_DEPS)

clean: 
	rm -f $(CPU_OBJS) $(GPU_OBJS) $(CPU_DEPS) $(GPU_DEPS) $(CPU_BINS) $(GPU_BINS)

.PHONY: all clean
