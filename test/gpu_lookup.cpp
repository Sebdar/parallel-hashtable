/** \file gpu_lookup.cpp
 * \brief Lookup a list of values on the device
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"
#include "parallel_hashtable.hpp"

#include <vector>

constexpr auto map_size = 1048576;
constexpr auto fill_size = 838860; // About 0.8 rate

__global__ void parallel_lookup(ParallelHashtable::Entry* hashtable, size_t n,
                                uint32_t* keys, uint32_t* values) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (auto i = offset; i < n; i += stride) {
        auto k = keys[i];
        values[i] = ParallelHashtable::lookup(hashtable, k, n);
    }
}

int main() {
    ParallelHashtable map{map_size};
    std::vector<uint32_t> keys_host{fill_size};
    std::vector<uint32_t> values_host{fill_size};

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

    parallel_lookup<<<1, 64>>>(hashtable_device, map.getCapacity(), keys_device,
                               values_device);
    hip::check(hipDeviceSynchronize());

    hip::check(hipMemcpy(values_host.data(), values_device, size,
                         hipMemcpyDeviceToHost));

    for (auto i = 0u; i < fill_size; ++i) {
        if (keys_host[i] != values_host[i]) {
            throw std::runtime_error("Unexpected value");
        }
    }

    return 0;
}
