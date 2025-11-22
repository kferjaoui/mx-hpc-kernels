CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++20 -pthread #-Wall -Wextra -Wpedantic
INC      := -I. -I../common -I../common/00_mx/include -I../third_party/eigen -MMD -MP

CPU_SRCS  := main.cpp
OBJS      := $(patsubst %.cpp,%.o,$(CPU_SRCS))
DEPS      := $(OBJS:.o=.d)
BINS      := gemm_test

all: $(BINS)

gemm_test: main.o
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ 

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(OBJS) $(DEPS) $(BINS)

.PHONY: all clean

-include $(DEPS)
