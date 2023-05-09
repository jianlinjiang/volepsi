#include "rvole.h"
extern "C" {
    void init_silent_vole(int role, const char* ip, const uint64_t seed[2], uint64_t vector_size);
    void silent_receive_inplace(uint64_t vector_size);
}

int main() {
    uint64_t seed[2] = {12, 12}; 
    init_silent_vole(2, "127.0.0.1:8000", seed, 1<<20);
    uint64_t delta[2] = {1, 2};
    silent_receive_inplace(1<<20);

    return 0;
}