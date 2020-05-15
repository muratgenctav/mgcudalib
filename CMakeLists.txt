# Minimum required version
cmake_minimum_required(VERSION 3.1.0)

# Project settings
project(MGCudaLib VERSION 0.1.0
        DESCRIPTION "Basic GPU algorithms library"
        LANGUAGES CUDA CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    enable_testing()

    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    include(CPack)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add library source directory
add_subdirectory(src)

# Add test sourcer directory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    #add_subdirectory(test)
endif()