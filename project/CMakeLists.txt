cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(VERSION 3.9)

set(CMAKE_C_COMPILER /usr/bin/gcc-7 CACHE PATH "")
set(CMAKE_CXX_COMPILER /usr/bin/g++-7 CACHE PATH "")

# # For MSVC and FindThreads, respectively.
enable_language(CXX C)

project(cga VERSION 1.0.0 LANGUAGES CXX CUDA)

set(OpenGL_GL_PREFERENCE LEGACY)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

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

cuda_add_executable ( ${PROJECT_NAME} main.cu
    src/image.cpp 
    src/cell_grid.cu 
    )

set(OpenGL_GL_PREFERENCE LEGACY)

find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})


find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    target_link_libraries (${PROJECT_NAME} ${GLEW_LIBRARIES})
endif()

find_package(GLUT REQUIRED)
if(GLUT_FOUND)
    include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
    target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )    
endif()
    

# Specify include directories
include_directories(
    include
    src
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries (${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libfreeimage-3.17.0.so")