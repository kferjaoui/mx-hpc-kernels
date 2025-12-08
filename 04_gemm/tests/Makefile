CXX        = g++
STD        = -std=c++20
OPT        = -O3 -march=native
WARN       = -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

ROOT           := ../..
COMMON         := $(ROOT)/common
MXINC          := $(COMMON)/00_mx/include
EIGEN_INCLUDE  := $(ROOT)/third_party/eigen
INC      	   := -I.. -I$(COMMON) -I$(MXINC) -I$(EIGEN_INCLUDE) -MMD -MP

CXXFLAGS  := $(STD) $(OPT) # $(WARN)
LDLIBS    := -pthread

CPU_SRCS  := bench_gemm.cpp
OBJS      := $(patsubst %.cpp,%.o,$(CPU_SRCS))
DEPS      := $(OBJS:.o=.d)
BINS      := bench_gemm

all: $(BINS)

bench_gemm: bench_gemm.o
	$(CXX) $(CXXFLAGS)  $(INC) $< $(LDLIBS) -o $@ 

bench_gemm.o: bench_gemm.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(OBJS) $(DEPS) $(BINS)

.PHONY: all clean

-include $(DEPS)
