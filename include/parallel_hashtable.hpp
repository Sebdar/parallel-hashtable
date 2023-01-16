/** \file parallel_hashtable.hpp
 * \brief Parallel hashtabl definition and implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>

#include <iostream>

#include "hip/hip_runtime.h"

namespace hip {
inline void check(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "error : " << hipGetErrorString(err) << " (" << err
                  << ")\n";
        throw std::runtime_error(std::string("Encountered hip error ") +
                                 hipGetErrorString(err));
    }
}
} // namespace hip

class ParallelHashtable {
  public:
    using Key = uint32_t;
    using Value = uint32_t;

    struct Entry {
        Key key;
        Value value;

        constexpr static Key Empty = 0xFFFFFFFF;
    } __attribute__((packed));

    static __device__ __host__ uint32_t hash(Key k, size_t capacity) {
        // Murmur3 hash
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k & (capacity - 1);
    }

    static __device__ __host__ Value lookup(Entry* hashtable, Key key,
                                            size_t capacity) {
        auto slot = hash(key, capacity);

        while (true) {
            auto entry = hashtable[slot];
            if (entry.key == key) {
                return entry.value;
            } else if (entry.key == Entry::Empty) {
                return Entry::Empty;
            }

            slot = (slot + 1) & (capacity - 1);
        }
    }

    // ----- Init ----- //

    ParallelHashtable(size_t capacity)
        : capacity(capacity), array(new Entry[capacity]) {
        memset(array.get(), Entry::Empty, capacity * sizeof(Entry));
    }

    size_t getCapacity() const { return capacity; }

    // ----- Modifiers ----- //

    void insert(Key key, Value value) {
        auto slot = hash(key, capacity);

        while (true) {
            // No support for concurrent insert yet, should use CAS
            auto entry = array[slot];

            if (entry.key == Entry::Empty || entry.key == key) {
                array[slot].key = key;
                array[slot].value = value;
                break;
            }

            slot = (slot + 1) & (capacity - 1);
        }
    }

    Value operator[](Key key) { return lookup(array.get(), key, capacity); }

    // ----- Device I/O ----- //

    Entry* toDevice() const {
        Entry* device;

        auto size = capacity * sizeof(Entry);

        hip::check(hipMalloc(&device, size));
        hip::check(hipMemcpy(device, array.get(), size, hipMemcpyHostToDevice));

        return device;
    }

    void fromDevice(Entry* device_ptr) {
        auto size = capacity * sizeof(Entry);

        hip::check(
            hipMemcpy(array.get(), device_ptr, size, hipMemcpyDeviceToHost));
    }

  private:
    std::unique_ptr<Entry[]> array;
    size_t capacity;
};
