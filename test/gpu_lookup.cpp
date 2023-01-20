/** \file gpu_lookup.cpp
 * \brief Lookup a list of values on the device
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"
#include "parallel_hashtable.hpp"

#include <chrono>
#include <vector>

constexpr auto default_map_size = 8388608;
constexpr auto default_rate = 0.8;

__global__ void parallel_lookup(ParallelHashtable::Entry* hashtable, size_t n,
                                uint32_t* keys, uint32_t* values,
                                size_t n_query) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (auto i = offset; i < n_query; i += stride) {
        auto k = keys[i];

#ifndef INLINE_LOOKUP
        auto v = ParallelHashtable::lookup(hashtable, k, n);
#else
        ParallelHashtable::Value v;
        auto slot = ParallelHashtable::hash(k, n);
        while (true) {
            auto entry = hashtable[slot];
            if (entry.key == k) {
                v = entry.value;
                break;
            } else if (entry.key == ParallelHashtable::Entry::Empty) {
                v = ParallelHashtable::Entry::Empty;
                break;
            }

            slot = (slot + 1) & (n - 1);
        }
#endif

        // Simulate computation
        __syncthreads();

        values[i] = v;
    }
}
size_t compute_fill(size_t size, double rate) {
    return static_cast<double>(size) * rate;
}

constexpr auto thread_count = 64u * 64u;
/** \fn compute_geometry
 * \brief Compute grid geometry while keeping total thread count constant for
 * tests
 */
constexpr std::pair<dim3, dim3> compute_geometry(size_t block_size) {
    return {thread_count / block_size, block_size};
}

int main(int argc, char** argv) {
    size_t map_size = argc > 1 ? std::atoll(argv[1]) : default_map_size;
    double fill_rate = argc > 2 ? std::atof(argv[2]) : default_rate;
    auto fill_size = compute_fill(map_size, fill_rate);

    ParallelHashtable map{map_size};
    std::vector<uint32_t> keys_host;
    std::vector<uint32_t> values_host;

    keys_host.reserve(fill_size);
    values_host.reserve(fill_size);

    for (auto i = 0u; i < fill_size; ++i) {
        map.insert(i, i);
        keys_host.push_back(i);
        values_host.push_back(i);
    }

    uint32_t* keys_device;
    auto size = sizeof(uint32_t) * keys_host.size();
    hip::check(hipMalloc(&keys_device, size));
    hip::check(
        hipMemcpy(keys_device, keys_host.data(), size, hipMemcpyHostToDevice));

    uint32_t* values_device;
    hip::check(hipMalloc(&values_device, size));

    auto* hashtable_device = map.toDevice();

    // Empty kernel to start the queue
    parallel_lookup<<<1, 1>>>(nullptr, 0, nullptr, nullptr, 0);
    hip::check(hipDeviceSynchronize());

    auto [blocks, threads] = compute_geometry(512);
    auto t0 = std::chrono::steady_clock::now();
    parallel_lookup<<<blocks, threads>>>(hashtable_device, map.getCapacity(),
                                         keys_device, values_device, fill_size);
    hip::check(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    std::cout
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        << '\n';

    hip::check(hipMemcpy(values_host.data(), values_device, size,
                         hipMemcpyDeviceToHost));

    for (auto i = 0u; i < fill_size; ++i) {
        if (keys_host[i] != values_host[i]) {
            std::cout << i << " : "
                      << "Got " << values_host[i] << ", expected "
                      << keys_host[i] << '\n';
            throw std::runtime_error("Unexpected value");
        }
    }

    return 0;
}
