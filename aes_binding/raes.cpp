#include <cryptoTools/Crypto/AES.h>
using block = osuCrypto::block;
extern "C" {
    void set_key(const void *aes, const void* key) {
        osuCrypto::AES* p = (osuCrypto::AES*) aes;
        const block* k = (const block*)key;
        p->setKey(*k);
    }

    void hash_blocks(const void *aes, const void* plaintext, const size_t block_length, void* ciphertext) {
        osuCrypto::AES* paes = (osuCrypto::AES*) aes;
        const block* p = (const block*)plaintext;
        block* c = (block*) ciphertext;
        paes->hashBlocks(p, block_length, c);
    }

    void* new_aes() {
        osuCrypto::AES* p = new osuCrypto::AES();
        return (void*)p;
    }

    void delete_aes(void* aes) {
        osuCrypto::AES* p = (osuCrypto::AES*) aes;
        if (p != nullptr) {
            delete p;
        }
    }
}