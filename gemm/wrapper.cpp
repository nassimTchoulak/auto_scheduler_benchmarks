#include "Halide.h"
#include "wrapper.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>
#include "benchmarks_configure.h"

#define MAX_RAND 200

using namespace std::chrono;
using namespace std;

int main(int, char **argv)
{
    Halide::Buffer<int32_t> buf04(384, 64);
    Halide::Buffer<int32_t> buf05(384, 64);

    int *c_buf00 = (int*)malloc(384 * 64 * sizeof(int));
    parallel_init_buffer(c_buf00, 384 * 64,  (int32_t)56);
    Halide::Buffer<int32_t> buf00(c_buf00, 384, 64);

    int *c_buf01 = (int*)malloc(384 * 384 * sizeof(int));
    parallel_init_buffer(c_buf01, 384 * 384,  (int32_t)43);
    Halide::Buffer<int32_t> buf01(c_buf01, 384, 384);

    int *c_buf02 = (int*)malloc(384 * 64 * sizeof(int));
    parallel_init_buffer(c_buf02, 384 * 64,  (int32_t)64);
    Halide::Buffer<int32_t> buf02(c_buf02, 384, 64);
    
    std::vector<double> duration_vector;
    double start, end;
    
    for (int i = 0; i < 2; ++i) 
        bench_function(buf04.raw_buffer(), buf05.raw_buffer(), buf00.raw_buffer(), buf01.raw_buffer(), buf02.raw_buffer());
    
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = rtclock();
        bench_function(buf04.raw_buffer(), buf05.raw_buffer(), buf00.raw_buffer(), buf01.raw_buffer(), buf02.raw_buffer());
        end = rtclock();
        
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << median(duration_vector) << std::endl;
    
    return 0;
}
