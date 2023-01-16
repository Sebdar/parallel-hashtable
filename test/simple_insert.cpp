/** \file simple_insert.cpp
 * \brief Insert a few values, then look them up
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "parallel_hashtable.hpp"

#include <iostream>

constexpr auto map_size = 2048;
constexpr auto fill_size = 64;

int main() {
    ParallelHashtable map{map_size};
    for (auto i = 0u; i < fill_size; ++i) {
        std::cout << "Inserting " << i << " : " << i << '\n';
        map.insert(i, i);
    }

    for (auto i = 0u; i < fill_size; ++i) {
        std::cout << "Looking up " << i << '\n';
        auto v = map[i];
        if (v != i) {
            throw std::runtime_error("Unexpected value");
        }
    }
}
