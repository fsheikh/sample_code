
# Make script to build some sample unit-tests
BINARY_NAME:=algo_test
GTEST_DIR := /home/faheem/googletest/googletest

gtest: $(GTEST_DIR)/src/gtest-all.cc
	g++ -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -c $<
	g++ -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -c $(GTEST_DIR)/src/gtest_main.cc
	ar -rv libgtest.a gtest-all.o
    
gtest_sample:gtest_sample.cc
	g++ -std=gnu++11 -isystem $(GTEST_DIR)/include -pthread $< libgtest.a gtest_main.o -o $@

all: algorithm_test.cpp
	g++ -std=gnu++11 -Wall -o $(BINARY_NAME) $<
clean:
	rm -f $(BINARY_NAME) gtest_sample *.o
