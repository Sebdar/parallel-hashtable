project(parallel-hashtable)
cmake_minimum_required(VERSION 3.20)

# ----- Setup ----- #

find_package(hip REQUIRED)
set(CMAKE_CXX_STANDARD 20)

include_directories(
    include
)

# ----- Tests ----- #

add_executable(fill
    test/fill.cpp
)

target_link_libraries(fill
    hip::device
)

