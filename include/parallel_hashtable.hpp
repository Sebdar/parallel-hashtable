/** \file parallel_hashtable.hpp
 * \brief Parallel hashtabl definition and implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <cstdint>

class ParallelHashtable {
  public:
    using Key = uint32_t;
    using Value = uint32_t;

    struct Entry {
        Key key;
        Value value;

        constexpr static Key Empty = 0xFFFF;
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

    static __device__ Value lookup(Entry* hashtable, Key key, size_t capacity) {
        uint32_t slot = hash(key, capacity);

        while (true) {
            auto entry = hashtable[slot];
            if (entry.key == key) {
                return entry.value;
            } else if (entry.key == Entry::empty) {
                return Entry::empty;
            }

            slot = (slot + 1) & (capacity - 1);
        }
    }

    // ----- Init ----- //

    ParallelHashtable(size_t capacity) {
        // TODO
    }

    size_t getCapacity() const { return capacity; }

    // ----- Modifiers ----- //

    // ----- Device I/O ----- //

    Pair* toDevice() const {
        // TODO
        return nullptr;
    }

    void fromDevice() {
        // TODO
    }

  private:
    Entry* array;
    size_t capacity;
};
