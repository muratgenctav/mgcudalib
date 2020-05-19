
# TEST SPMV
# add test executable
#include_directories(${PROJECT_SOURCE_DIR})
set(CUDA_SEPARABLE_COMPILATION ON)
add_executable(test_spmv test_spmv.cu mmio.h mmio.c)
# link mgcudalib and cuda libraries
link_directories(/usr/local/cuda/lib64)
target_link_libraries(test_spmv PRIVATE mgcudalib cusparse cudart)
# register test
add_test(NAME test_spmv_row COMMAND test_spmv testrow.mtx)
add_test(NAME test_spmv_col COMMAND test_spmv testrow.mtx -c)