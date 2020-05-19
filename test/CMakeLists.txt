
# TEST SPMV
# add test executable
add_executable(test_spmv test_spmv.cpp mmio.c)
# link mgcudalib and cuda libraries
target_link_libraries(test_spmv PRIVATE mgcudalib cusparse cudart)
# register test
add_test(NAME test_spmv_row COMMAND test_spmv testrow.mtx)
add_test(NAME test_spmv_col COMMAND test_spmv testrow.mtx -c)