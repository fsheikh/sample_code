
# A rudimentary build script

.PHONY: clean link_gmb


# Change paths below accordingly
GMB_LIB := ../../google_benchmarks/benchmark/src
GMB_INC := ../../google_benchmarks/benchmark/include

INCLUDE = $(abspath $(GMB_INC))
LIB     = $(abspath $(GMB_LIB))
#link_benchmark.o: link_benchmark.cpp
%.o: %.cpp
	echo $(INCLUDE)
	g++ -std=c++11 -Wall -c -I$(INCLUDE) -o $@ $<

link_gmb: link_benchmark.o
	g++ -std=c++11 -L$(LIB) -o link_gmb_test $< -lbenchmark

clean:
	rm -rf link_benchmark.o link_gmb_test
