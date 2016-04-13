#include <iostream>
#include "benchmark/benchmark_api.h"


// Maintain the state globally since we need to share it across functions.
benchmark::State *irq_state;
static constexpr bool LOG = false;
static bool wait_for_int  = true;


// Utitlity function mimicking IRQ handler
static void sim_irq_handler()
{
    irq_state->PauseTiming();
    if (LOG) {
        std::cout << "Hello from the IRQ handler" << std::endl;
    }
}

// Microbenchmark simulating IRQ latency measurement
static void BM_IRQ_Sim(benchmark::State &state)
{
    if (LOG) {
        std::cout << "IRQ Simulator Microbenchmark called" << std::endl;
    }
    int irq_char;
    irq_state = &state;
    if (wait_for_int) {
        std::cout << "Press C once to simulate IRQ" << std::endl;
    }
    std::cin >> irq_char;
    uint64_t benchmark_index = 0;
    while (state.KeepRunning())
    {
        while(wait_for_int and irq_char != 'C') {
            if (LOG) {
                std::cout << "Current input " << irq_char << std::endl;
            }
            wait_for_int = false;
        }
        sim_irq_handler();
        state.ResumeTiming();
        benchmark_index++;
    }
    if (LOG) {
        std::cout << "Benchmark index" << benchmark_index << std::endl;
    }
}

// Register our IRQ simulating mirco-benchmark
BENCHMARK(BM_IRQ_Sim);

// Call library's initialize function
int main(int argc, char **argv)
{

    std::cout << "Sample program linking to Google Micro-Benchmark" << std::endl;

    benchmark::Initialize(&argc, argv);

    std::cout << "Google Micro-Benchmark initialized, Now running benchmarks" << std::endl;

    benchmark::RunSpecifiedBenchmarks();
}
