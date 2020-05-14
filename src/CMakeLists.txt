
# Define list of headers
set(HEADER_LIST 
    "${MGCudaLib_SOURCE_DIR}/include/mgcucommon/mgcucommon.cuh"
    "${MGCudaLib_SOURCE_DIR}/include/mgcublas/mgcublas.cuh"
    "${MGCudaLib_SOURCE_DIR}/include/mgcuutils/mgcuutils.cuh")

# Set target library
add_library(mgcudalib MGCudaLib.cu ${HEADER_LIST})

# Set include directory (for any user of this library)
target_include_directories(mgcudalib PUBLIC ../include)

# Put headers in a nice place
# source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})