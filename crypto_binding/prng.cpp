#include <cryptoTools/Crypto/PRNG.h>
using PRNG = osuCrypto::PRNG;
using block  = oc::block;
template<typename T>
using span = oc::span<T>;
extern "C" {
    void* new_prng(const void* seed) {
        PRNG* prng = new PRNG(*((block*) seed));
        return (void*)prng;
    }

    void get_blocks_raw(const void* prng_ptr, void* blocks, size_t block_num) {
        PRNG* prng = (PRNG*) prng_ptr;
        prng->get<block>((block*) blocks, block_num);
    }

    void delete_prng(const void* prng) {
        if (prng != nullptr) {
            PRNG* p = (PRNG*) prng;
            delete p;
        }
    }
}