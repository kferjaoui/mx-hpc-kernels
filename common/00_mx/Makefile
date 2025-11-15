CXX      := g++
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -Wpedantic
INC      := -Iinclude -I../../third_party/eigen -MMD -MP

TEST_SRCS := $(wildcard tests/*.cpp)
OBJS      := $(patsubst tests/%.cpp,build/%.o,$(TEST_SRCS))
DEPS      := $(OBJS:.o=.d)
TEST_BINS := $(patsubst tests/%.cpp,bin/%,$(TEST_SRCS))

all: $(TEST_BINS)

bin/%: build/%.o
	@mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ 

build/%.o: tests/%.cpp
	@mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -rf build bin

.PHONY: all clean

-include $(DEPS)
