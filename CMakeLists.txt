project(parallel-hashtable)
cmake_minimum_required(VERSION 3.20)

# ----- Setup ----- #

find_package(hip REQUIRED)
set(CMAKE_CXX_STANDARD 20)

set(HIP_PATH /opt/rocm)

set(CMAKE_CXX_FLAGS "-Wall -fpass-plugin=/home/sdarche/hip-analyzer/build/libhip-analyzer-pass.so -O3")
#set(CMAKE_CXX_FLAGS "-Wall -fpass-plugin=/home/sdarche/hip-analyzer/build/libhip-kernel-timer-pass.so -O3")

include_directories(
    include
)

# ----- Tests ----- #

link_directories(
    ${HIP_PATH}/lib
    ${HIP_CLANG_PATH}/../lib/clang/16.0.0/lib/linux
    /home/sdarche/hip-analyzer/build
)

add_executable(gpu_lookup
    test/gpu_lookup.cpp
)

target_link_libraries(gpu_lookup
    hip::device
    hip_instrumentation
    dl
    jsoncpp
    pthread
    roctracer64
    rocprofiler64

)

add_executable(simple_insert
    test/simple_insert.cpp
)

target_link_libraries(simple_insert
    hip::device
)
