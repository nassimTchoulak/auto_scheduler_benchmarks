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
    Halide::Buffer<int32_t> buf01(1024, 1024, 2, 8);
    Halide::Buffer<int32_t> buf05(1024, 1024, 2, 8);

    int *c_buf00 = (int*)malloc(2 * sizeof(int));
    parallel_init_buffer(c_buf00, 2,  (int32_t)84);
    Halide::Buffer<int32_t> buf00(c_buf00, 2);

    int *c_buf02 = (int*)malloc(1026 * 1026 * 3 * 8 * sizeof(int));
    parallel_init_buffer(c_buf02, 1026 * 1026 * 3 * 8,  (int32_t)32);
    Halide::Buffer<int32_t> buf02(c_buf02, 1026, 1026, 3, 8);

    int *c_buf03 = (int*)malloc(3 * 3 * 3 * 2 * sizeof(int));
    parallel_init_buffer(c_buf03, 3 * 3 * 3 * 2,  (int32_t)85);
    Halide::Buffer<int32_t> buf03(c_buf03, 3, 3, 3, 2);
    
    std::vector<double> duration_vector;
    double start, end;
    
    for (int i = 0; i < 2; ++i) 
        bench_function(buf01.raw_buffer(), buf05.raw_buffer(), buf00.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer());
    
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = rtclock();
        bench_function(buf01.raw_buffer(), buf05.raw_buffer(), buf00.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer());
        end = rtclock();
        
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << median(duration_vector) << std::endl;
    
    return 0;
}
