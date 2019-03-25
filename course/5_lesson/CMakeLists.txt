# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(VERSION 3.9)

set(CMAKE_C_COMPILER /usr/bin/gcc-7 CACHE PATH "")
set(CMAKE_CXX_COMPILER /usr/bin/g++-7 CACHE PATH "")

# # For MSVC and FindThreads, respectively.
enable_language(CXX C)

project(cudaProj VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pass options to NVCC
# set(CMAKE_CUDA_FLAGS "-arch=sm_30 \
#                       -gencode=arch=compute_30,code=sm_30 \
#                       -gencode=arch=compute_50,code=sm_50 \
#                       -gencode=arch=compute_52,code=sm_52 \
#                       -gencode=arch=compute_60,code=sm_60 \
#                       -gencode=arch=compute_61,code=sm_61 \
#                       -gencode=arch=compute_61,code=compute_61")

cuda_add_executable ( ${PROJECT_NAME} Runner.cu )


# Specify include directories
include_directories(
    common/CUDA_10-0_SDK/inc
    common/FreeImage/inc
    common/Utils
    # common/Glew/inc
    src
    ${CUDA_INCLUDE_DIRS}
)

# add_library(free_image STATIC common/FreeImage)
target_link_libraries (${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libfreeimage-3.17.0.so")


#target_link_libraries (${PROJECT_NAME} FREE_IMAGE_LIBRARY)
