cmake_minimum_required(VERSION 3.25)
 
project(TDBase)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
 
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(OpenMP REQUIRED)

set (CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

file(GLOB SPATIAL_SRCS
	spatial/*.cpp
)

file(GLOB GEOMETRY_SRCS
	geometry/*.cpp
)

file(GLOB GEOMETRY_CUDA_SRCS
	geometry/*.cu
)

file(GLOB INDEX_SRCS
	index/*.cpp
)

file(GLOB STORAGE_SRCS
	storage/*.cpp
)

file(GLOB JOIN_SRCS
	join/*.cpp
)

file(GLOB MYMESH_SRCS
	mymesh/*.cpp
)

file(GLOB TDBASE_SRCS
	tools/*.cpp
)

list(REMOVE_ITEM TDBASE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tools/range_analysis.cpp")

include_directories(include mytest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_HAS_THREADS -DCGAL_HAS_THREADS -DCGAL_DONT_USE_LAZY_KERNEL")

if(DEFINED ENV{USE_GPU})
    message (STATUS "cuda")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "CUDA compiler")
    
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fpermissive")
    enable_language(CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU")
    add_executable (tdbase ${TDBASE_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_SRCS} ${GEOMETRY_CUDA_SRCS} ${INDEX_SRCS} ${STORAGE_SRCS} ${JOIN_SRCS} mytest/face_classifier.cpp)
    target_link_libraries(tdbase PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX cuda)
    
else()
    message (STATUS "non-cuda")
    add_executable (tdbase ${TDBASE_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_SRCS} ${INDEX_SRCS} ${STORAGE_SRCS} ${JOIN_SRCS} mytest/face_classifier.cpp)
    target_link_libraries(tdbase PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX)
endif()
    
add_executable (wrap simplifier/wrap.cpp)
target_link_libraries(wrap PRIVATE gmp mpfr)

add_executable (mymesh ${MYMESH_SRCS})

file(GLOB MYTEST_SRCS
	mytest/*.cpp
)

file(GLOB MYTEST_CUDA_SRCS
	mytest/*.cu
)

file(GLOB GEOMETRY_CPU_SRCS
	geometry/TriInt.cpp
	geometry/computing_cpu.cpp
)

if(DEFINED ENV{USE_GPU})
    add_executable (face_classifier ${MYTEST_SRCS} ${MYTEST_CUDA_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_CPU_SRCS} ${STORAGE_SRCS} ${INDEX_SRCS})
    target_link_libraries(face_classifier PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX cuda)
else()
    add_executable (face_classifier ${MYTEST_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_CPU_SRCS} ${STORAGE_SRCS} ${INDEX_SRCS})
    target_link_libraries(face_classifier PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX)
endif()

