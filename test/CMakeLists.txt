
# TEST SPMV
# add test executable
add_executable(test_spmv test_spmv.cu mmio.c mmio.h)
# link mgcudalib and cuda libraries
link_directories(/usr/local/cuda/lib64)
target_link_libraries(test_spmv PRIVATE mmio mgcudalib cusparse cudart)
# register test
add_test(NAME test_spmv_row COMMAND test_spmv testrow.mtx)
add_test(NAME test_spmv_col COMMAND test_spmv testrow.mtx -c)