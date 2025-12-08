CXX       ?= g++
STD        = -std=c++20
OPT        = -O3 -march=native
WARN       = -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

ROOT           := ../..
THIRD_PARTY    := $(ROOT)/third_party
COMMON         := $(ROOT)/common
MXINC          := $(COMMON)/00_mx/include
GEMM_INC       := $(ROOT)/04_gemm
EIGEN_INCLUDE  := $(THIRD_PARTY)/eigen

CPPFLAGS  := -I./include -I.. -I$(COMMON) -I$(GEMM_INC) -I$(MXINC) -I$(EIGEN_INCLUDE) -MMD -MP
CXXFLAGS  := $(STD) $(OPT) #$(WARN)
LDFLAGS   :=
LDLIBS    := -pthread

SRCS      := \
  lu_fullrank_unblocked.cpp \
  lu_singular_unblocked.cpp \
  lu_rectangular_unblocked.cpp \
  lu_blocked_vs_unblocked_gram_perf.cpp \
  lu_solve_single_multi_rhs.cpp

OBJS      := $(SRCS:.cpp=.o)
DEPS      := $(OBJS:.o=.d)
TARGETS   := $(SRCS:.cpp=)

all: $(TARGETS)

# One rule to link them all
%: %.o
	$(CXX) $(LDFLAGS) $< $(LDLIBS) -o $@

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

perf: lu_blocked_vs_unblocked_gram_perf

debug: CXXFLAGS := $(STD) -O0 -g $(WARN)
debug: clean all

asan:  CXXFLAGS += -O1 -g -fsanitize=address -fno-omit-frame-pointer
asan:  LDFLAGS  += -fsanitize=address
asan:  clean all

ubsan: CXXFLAGS += -O1 -g -fsanitize=undefined
ubsan: LDFLAGS  += -fsanitize=undefined
ubsan: clean all

clean:
	rm -f $(OBJS) $(DEPS) $(TARGETS)

.PHONY: all clean perf debug asan ubsan
-include $(DEPS)
