#include <vector>
// #include <cryptoTools/Common/config.h>
#define LIBDIVIDE_AVX2
#include "libdivide.h"
#include <cryptoTools/Crypto/AES.h>
using u64 = oc::u64;
using u32 = oc::u32;
using u16 = oc::u16;
using u8 = oc::u8;
using block  = oc::block;
template<typename T>
using span = oc::span<T>;

template<typename IdxType>
struct PaxosHash
{
    u64 mWeight, mSparseSize, mIdxSize;
    oc::AES mAes;
    std::vector<libdivide::libdivide_u64_t> mMods;
    //std::vector<libdivide::libdivide_u64_branchfree_t> mModsBF;
    std::vector<u64> mModVals;
    void init(block seed, u64 weight, u64 paxosSize)
    {
        mWeight = weight;
        mSparseSize = paxosSize;
        mIdxSize = static_cast<IdxType>(oc::roundUpTo(oc::log2ceil(mSparseSize), 8) / 8);
        mAes.setKey(seed);

        mModVals.resize(weight);
        mMods.resize(weight);
        //mModsBF.resize(weight);
        for (u64 i = 0; i < weight; ++i)
        {
            mModVals[i] = mSparseSize - i;
            mMods[i] = libdivide::libdivide_u64_gen(mModVals[i]);
            //mModsBF[i] = libdivide::libdivide_u64_branchfree_gen(mModVals[i]);
        }
    }
    void mod32(u64* vals, u64 modIdx) const;
    void hashBuildRow32(const block* input, IdxType* rows, block* hash) const;
    void hashBuildRow1(const block* input, IdxType* rows, block* hash) const;
    void buildRow(const block& hash, IdxType* row) const;
    void buildRow32(const block* hash, IdxType* row) const;
};