project(parallel-hashtable)
cmake_minimum_required(VERSION 3.20)

# ----- Setup ----- #

find_package(hip REQUIRED)
set(CMAKE_CXX_STANDARD 20)

include_directories(
    include
)

# ----- Tests ----- #

add_executable(gpu_lookup
    test/gpu_lookup.cpp
)

target_link_libraries(gpu_lookup
    hip::device
)

add_executable(simple_insert
    test/simple_insert.cpp
)

target_link_libraries(simple_insert
    hip::device
)
