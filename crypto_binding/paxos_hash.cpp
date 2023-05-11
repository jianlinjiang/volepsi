#include "paxos_hash.h"
#include <immintrin.h>
using block256 = __m256i;
inline block256 my_libdivide_u64_do_vec256(const block256& x, const libdivide::libdivide_u64_t* divider)
{
    return libdivide::libdivide_u64_do_vec256(x, divider);
}

inline void doMod32(u64* vals, const libdivide::libdivide_u64_t* divider, const u64& modVal)
{
    {
        u64 i = 0;
        block256 row256a = _mm256_loadu_si256((block256*)&vals[i]);
        block256 row256b = _mm256_loadu_si256((block256*)&vals[i + 4]);
        block256 row256c = _mm256_loadu_si256((block256*)&vals[i + 8]);
        block256 row256d = _mm256_loadu_si256((block256*)&vals[i + 12]);
        block256 row256e = _mm256_loadu_si256((block256*)&vals[i + 16]);
        block256 row256f = _mm256_loadu_si256((block256*)&vals[i + 20]);
        block256 row256g = _mm256_loadu_si256((block256*)&vals[i + 24]);
        block256 row256h = _mm256_loadu_si256((block256*)&vals[i + 28]);
        auto tempa = my_libdivide_u64_do_vec256(row256a, divider);	// u256 / divider
        auto tempb = my_libdivide_u64_do_vec256(row256b, divider);
        auto tempc = my_libdivide_u64_do_vec256(row256c, divider);
        auto tempd = my_libdivide_u64_do_vec256(row256d, divider);
        auto tempe = my_libdivide_u64_do_vec256(row256e, divider);
        auto tempf = my_libdivide_u64_do_vec256(row256f, divider);
        auto tempg = my_libdivide_u64_do_vec256(row256g, divider);
        auto temph = my_libdivide_u64_do_vec256(row256h, divider);
        //auto temp = libdivide::libdivide_u64_branchfree_do_vec256(row256, divider);
        auto temp64a = (u64*)&tempa;
        auto temp64b = (u64*)&tempb;
        auto temp64c = (u64*)&tempc;
        auto temp64d = (u64*)&tempd;
        auto temp64e = (u64*)&tempe;
        auto temp64f = (u64*)&tempf;
        auto temp64g = (u64*)&tempg;
        auto temp64h = (u64*)&temph;
        vals[i + 0] -= temp64a[0] * modVal;
        vals[i + 1] -= temp64a[1] * modVal;
        vals[i + 2] -= temp64a[2] * modVal;
        vals[i + 3] -= temp64a[3] * modVal;
        vals[i + 4] -= temp64b[0] * modVal;
        vals[i + 5] -= temp64b[1] * modVal;
        vals[i + 6] -= temp64b[2] * modVal;
        vals[i + 7] -= temp64b[3] * modVal;
        vals[i + 8] -= temp64c[0] * modVal;
        vals[i + 9] -= temp64c[1] * modVal;
        vals[i + 10] -= temp64c[2] * modVal;
        vals[i + 11] -= temp64c[3] * modVal;
        vals[i + 12] -= temp64d[0] * modVal;
        vals[i + 13] -= temp64d[1] * modVal;
        vals[i + 14] -= temp64d[2] * modVal;
        vals[i + 15] -= temp64d[3] * modVal;
        vals[i + 16] -= temp64e[0] * modVal;
        vals[i + 17] -= temp64e[1] * modVal;
        vals[i + 18] -= temp64e[2] * modVal;
        vals[i + 19] -= temp64e[3] * modVal;
        vals[i + 20] -= temp64f[0] * modVal;
        vals[i + 21] -= temp64f[1] * modVal;
        vals[i + 22] -= temp64f[2] * modVal;
        vals[i + 23] -= temp64f[3] * modVal;
        vals[i + 24] -= temp64g[0] * modVal;
        vals[i + 25] -= temp64g[1] * modVal;
        vals[i + 26] -= temp64g[2] * modVal;
        vals[i + 27] -= temp64g[3] * modVal;
        vals[i + 28] -= temp64h[0] * modVal;
        vals[i + 29] -= temp64h[1] * modVal;
        vals[i + 30] -= temp64h[2] * modVal;
        vals[i + 31] -= temp64h[3] * modVal;
    }
}


template<typename IdxType>
void PaxosHash<IdxType>::mod32(u64* vals, u64 modIdx) const
{
    auto divider = &mMods[modIdx];
    auto modVal = mModVals[modIdx];
    doMod32(vals, divider, modVal);
}


template<typename IdxType>
void PaxosHash<IdxType>::buildRow32(const block* hash, IdxType* row) const
{
    if (mWeight == 3 /* && mSparseSize < std::numeric_limits<u32>::max()*/)
    {
        const auto weight = 3;
        block row128_[3][16];

        for (u64 i = 0; i < weight; ++i)
        {
            auto ll = (u64*)row128_[i];

            for (u64 j = 0; j < 32; ++j)
            {
                memcpy(&ll[j], hash[j].data() + sizeof(u32) * i, sizeof(u64));
            }
            mod32(ll, i);
        }


        for (u64 i = 0; i < 2; ++i)
        {
            std::array<block, 8> mask, max, min;

            std::array<block*, 3> row128{
                row128_[0] + i * 8,
                row128_[1] + i * 8,
                row128_[2] + i * 8 };
            
            // mask = a > b ? -1 : 0;
            mask[0] = _mm_cmpgt_epi64(row128[0][0], row128[1][0]);
            mask[1] = _mm_cmpgt_epi64(row128[0][1], row128[1][1]);
            mask[2] = _mm_cmpgt_epi64(row128[0][2], row128[1][2]);
            mask[3] = _mm_cmpgt_epi64(row128[0][3], row128[1][3]);
            mask[4] = _mm_cmpgt_epi64(row128[0][4], row128[1][4]);
            mask[5] = _mm_cmpgt_epi64(row128[0][5], row128[1][5]);
            mask[6] = _mm_cmpgt_epi64(row128[0][6], row128[1][6]);
            mask[7] = _mm_cmpgt_epi64(row128[0][7], row128[1][7]);


            min[0] = row128[0][0] ^ row128[1][0];
            min[1] = row128[0][1] ^ row128[1][1];
            min[2] = row128[0][2] ^ row128[1][2];
            min[3] = row128[0][3] ^ row128[1][3];
            min[4] = row128[0][4] ^ row128[1][4];
            min[5] = row128[0][5] ^ row128[1][5];
            min[6] = row128[0][6] ^ row128[1][6];
            min[7] = row128[0][7] ^ row128[1][7];


            // max = max(a,b)
            max[0] = (min[0]) & mask[0];
            max[1] = (min[1]) & mask[1];
            max[2] = (min[2]) & mask[2];
            max[3] = (min[3]) & mask[3];
            max[4] = (min[4]) & mask[4];
            max[5] = (min[5]) & mask[5];
            max[6] = (min[6]) & mask[6];
            max[7] = (min[7]) & mask[7];
            max[0] = max[0] ^ row128[1][0];
            max[1] = max[1] ^ row128[1][1];
            max[2] = max[2] ^ row128[1][2];
            max[3] = max[3] ^ row128[1][3];
            max[4] = max[4] ^ row128[1][4];
            max[5] = max[5] ^ row128[1][5];
            max[6] = max[6] ^ row128[1][6];
            max[7] = max[7] ^ row128[1][7];

            // min = min(a,b)
            min[0] = min[0] ^ max[0];
            min[1] = min[1] ^ max[1];
            min[2] = min[2] ^ max[2];
            min[3] = min[3] ^ max[3];
            min[4] = min[4] ^ max[4];
            min[5] = min[5] ^ max[5];
            min[6] = min[6] ^ max[6];
            min[7] = min[7] ^ max[7];

            //if (max == b)
            //  ++b
            //  ++max
            mask[0] = _mm_cmpeq_epi64(max[0], row128[1][0]);
            mask[1] = _mm_cmpeq_epi64(max[1], row128[1][1]);
            mask[2] = _mm_cmpeq_epi64(max[2], row128[1][2]);
            mask[3] = _mm_cmpeq_epi64(max[3], row128[1][3]);
            mask[4] = _mm_cmpeq_epi64(max[4], row128[1][4]);
            mask[5] = _mm_cmpeq_epi64(max[5], row128[1][5]);
            mask[6] = _mm_cmpeq_epi64(max[6], row128[1][6]);
            mask[7] = _mm_cmpeq_epi64(max[7], row128[1][7]);
            row128[1][0] = _mm_sub_epi64(row128[1][0], mask[0]);
            row128[1][1] = _mm_sub_epi64(row128[1][1], mask[1]);
            row128[1][2] = _mm_sub_epi64(row128[1][2], mask[2]);
            row128[1][3] = _mm_sub_epi64(row128[1][3], mask[3]);
            row128[1][4] = _mm_sub_epi64(row128[1][4], mask[4]);
            row128[1][5] = _mm_sub_epi64(row128[1][5], mask[5]);
            row128[1][6] = _mm_sub_epi64(row128[1][6], mask[6]);
            row128[1][7] = _mm_sub_epi64(row128[1][7], mask[7]);
            max[0] = _mm_sub_epi64(max[0], mask[0]);
            max[1] = _mm_sub_epi64(max[1], mask[1]);
            max[2] = _mm_sub_epi64(max[2], mask[2]);
            max[3] = _mm_sub_epi64(max[3], mask[3]);
            max[4] = _mm_sub_epi64(max[4], mask[4]);
            max[5] = _mm_sub_epi64(max[5], mask[5]);
            max[6] = _mm_sub_epi64(max[6], mask[6]);
            max[7] = _mm_sub_epi64(max[7], mask[7]);

            // if (c >= min)
            //   ++c
            mask[0] = _mm_cmpgt_epi64(min[0], row128[2][0]);
            mask[1] = _mm_cmpgt_epi64(min[1], row128[2][1]);
            mask[2] = _mm_cmpgt_epi64(min[2], row128[2][2]);
            mask[3] = _mm_cmpgt_epi64(min[3], row128[2][3]);
            mask[4] = _mm_cmpgt_epi64(min[4], row128[2][4]);
            mask[5] = _mm_cmpgt_epi64(min[5], row128[2][5]);
            mask[6] = _mm_cmpgt_epi64(min[6], row128[2][6]);
            mask[7] = _mm_cmpgt_epi64(min[7], row128[2][7]);
            mask[0] = mask[0] ^ oc::AllOneBlock;
            mask[1] = mask[1] ^ oc::AllOneBlock;
            mask[2] = mask[2] ^ oc::AllOneBlock;
            mask[3] = mask[3] ^ oc::AllOneBlock;
            mask[4] = mask[4] ^ oc::AllOneBlock;
            mask[5] = mask[5] ^ oc::AllOneBlock;
            mask[6] = mask[6] ^ oc::AllOneBlock;
            mask[7] = mask[7] ^ oc::AllOneBlock;
            row128[2][0] = _mm_sub_epi64(row128[2][0], mask[0]);
            row128[2][1] = _mm_sub_epi64(row128[2][1], mask[1]);
            row128[2][2] = _mm_sub_epi64(row128[2][2], mask[2]);
            row128[2][3] = _mm_sub_epi64(row128[2][3], mask[3]);
            row128[2][4] = _mm_sub_epi64(row128[2][4], mask[4]);
            row128[2][5] = _mm_sub_epi64(row128[2][5], mask[5]);
            row128[2][6] = _mm_sub_epi64(row128[2][6], mask[6]);
            row128[2][7] = _mm_sub_epi64(row128[2][7], mask[7]);

            // if (c >= max)
            //   ++c
            mask[0] = _mm_cmpgt_epi64(max[0], row128[2][0]);
            mask[1] = _mm_cmpgt_epi64(max[1], row128[2][1]);
            mask[2] = _mm_cmpgt_epi64(max[2], row128[2][2]);
            mask[3] = _mm_cmpgt_epi64(max[3], row128[2][3]);
            mask[4] = _mm_cmpgt_epi64(max[4], row128[2][4]);
            mask[5] = _mm_cmpgt_epi64(max[5], row128[2][5]);
            mask[6] = _mm_cmpgt_epi64(max[6], row128[2][6]);
            mask[7] = _mm_cmpgt_epi64(max[7], row128[2][7]);
            mask[0] = mask[0] ^ oc::AllOneBlock;
            mask[1] = mask[1] ^ oc::AllOneBlock;
            mask[2] = mask[2] ^ oc::AllOneBlock;
            mask[3] = mask[3] ^ oc::AllOneBlock;
            mask[4] = mask[4] ^ oc::AllOneBlock;
            mask[5] = mask[5] ^ oc::AllOneBlock;
            mask[6] = mask[6] ^ oc::AllOneBlock;
            mask[7] = mask[7] ^ oc::AllOneBlock;
            row128[2][0] = _mm_sub_epi64(row128[2][0], mask[0]);
            row128[2][1] = _mm_sub_epi64(row128[2][1], mask[1]);
            row128[2][2] = _mm_sub_epi64(row128[2][2], mask[2]);
            row128[2][3] = _mm_sub_epi64(row128[2][3], mask[3]);
            row128[2][4] = _mm_sub_epi64(row128[2][4], mask[4]);
            row128[2][5] = _mm_sub_epi64(row128[2][5], mask[5]);
            row128[2][6] = _mm_sub_epi64(row128[2][6], mask[6]);
            row128[2][7] = _mm_sub_epi64(row128[2][7], mask[7]);
            {
                for (u64 j = 0; j < mWeight; ++j)
                {
                    IdxType* __restrict rowi = row + mWeight * 16 * i;
                    u64* __restrict row64 = (u64*)(row128[j]);
                    rowi[mWeight * 0 + j] = row64[0];
                    rowi[mWeight * 1 + j] = row64[1];
                    rowi[mWeight * 2 + j] = row64[2];
                    rowi[mWeight * 3 + j] = row64[3];
                    rowi[mWeight * 4 + j] = row64[4];
                    rowi[mWeight * 5 + j] = row64[5];
                    rowi[mWeight * 6 + j] = row64[6];
                    rowi[mWeight * 7 + j] = row64[7];

                    rowi += 8 * mWeight;
                    row64 += 8;

                    rowi[mWeight * 0 + j] = row64[0];
                    rowi[mWeight * 1 + j] = row64[1];
                    rowi[mWeight * 2 + j] = row64[2];
                    rowi[mWeight * 3 + j] = row64[3];
                    rowi[mWeight * 4 + j] = row64[4];
                    rowi[mWeight * 5 + j] = row64[5];
                    rowi[mWeight * 6 + j] = row64[6];
                    rowi[mWeight * 7 + j] = row64[7];
                }
            }
        }
    }
    else
    {
        for (u64 k = 0; k < 32; ++k)
        {
            buildRow(hash[k], row);
            row += mWeight;
        }
    }
}


template<typename IdxType>
void PaxosHash<IdxType>::buildRow(const block& hash, IdxType* row) const
{
    if (mWeight == 3)
    {
        u32* rr = (u32*)&hash;
        auto rr0 = *(u64*)(&rr[0]);
        auto rr1 = *(u64*)(&rr[1]);
        auto rr2 = *(u64*)(&rr[2]);
        row[0] = (IdxType)(rr0 % mSparseSize);
        row[1] = (IdxType)(rr1 % (mSparseSize - 1));
        row[2] = (IdxType)(rr2 % (mSparseSize - 2));

        assert(row[0] < mSparseSize);
        assert(row[1] < mSparseSize);
        assert(row[2] < mSparseSize);

        auto min = std::min<IdxType>(row[0], row[1]);
        auto max = row[0] + row[1] - min;

        if (max == row[1])
        {
            ++row[1];
            ++max;
        }

        if (row[2] >= min)
            ++row[2];

        if (row[2] >= max)
            ++row[2];
    }
    else
    {
        auto hh = hash;
        for (u64 j = 0; j < mWeight; ++j)
        {
            auto modulus = (mSparseSize - j);

            hh = hh.gf128Mul(hh);
            //std::memcpy(&h, (u8*)&hash + byteIdx, mIdxSize);
            auto colIdx = hh.get<u64>(0) % modulus;

            auto iter = row;
            auto end = row + j;
            while (iter != end)
            {
                if (*iter <= colIdx)
                    ++colIdx;
                else
                    break;
                ++iter;
            }


            while (iter != end)
            {
                end[0] = end[-1];
                --end;
            }

            *iter = static_cast<IdxType>(colIdx);
        }
    }
}

template<typename IdxType>
void PaxosHash<IdxType>::hashBuildRow1(
    const block* inIter,
    IdxType* rows,
    block* hash) const
{
    *hash = mAes.hashBlock(inIter[0]);
    buildRow(*hash, rows);
}

template<typename IdxType>
void PaxosHash<IdxType>::hashBuildRow32(
    const block* inIter,
    IdxType* rows,
    block* hash) const
{
    mAes.hashBlocks(span<const block>(inIter, 32), span<block>(hash, 32));
    buildRow32(hash, rows);
}

extern "C" {
    void* new_paxos_hash(size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* hasher = new PaxosHash<u16>();
            return (void *) hasher;
        } else if (bit_length <= 32) {
            PaxosHash<u16>* hasher = new PaxosHash<u16>();
            return (void *) hasher;
        } else {
            PaxosHash<u16>* hasher = new PaxosHash<u16>();
            return (void *) hasher;
        }
        
    }

    void init_paxos_hash(const void* hasher, const void* seed, size_t weight, size_t paxos_size, size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
            h->init(*((block*)seed), weight, paxos_size);
        } else if (bit_length <= 32) {
            PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
            h->init(*((block*)seed), weight, paxos_size);
        } else {
            PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
            h->init(*((block*)seed), weight, paxos_size);
        }
    }

    void delete_paxos_hash(const void* hasher, size_t bit_length) {
        if (hasher != nullptr) {
            if (bit_length <= 16) {
                PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
                delete h;
            } else if (bit_length <= 32) {
                PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
                delete h;
            } else {
                PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
                delete h;
            }
        }
    }

    void build_row_raw(const void* hasher, const void* hash, void* row, size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
            h->buildRow(*((const block*)hash), (u16*)row);
        } else if (bit_length <= 32) {
            PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
            h->buildRow(*((const block*)hash), (u32*)row);
        } else {
            PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
            h->buildRow(*((const block*)hash), (u64*)row);
        }
    }

    void build_row32_raw(const void* hasher, const void* hash, void* row, size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
            h->buildRow32(((const block*)hash), (u16*)row);
        } else if (bit_length <= 32) {
            PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
            h->buildRow32(((const block*)hash), (u32*)row);
        } else {
            PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
            h->buildRow32(((const block*)hash), (u64*)row);
        }   
    }

    void hash_build_row1_raw(const void* hasher, const void* input, void* rows, void* hash, size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
            h->hashBuildRow1((const block*)input, (u16*)rows, (block*) hash);
        } else if (bit_length <= 32) {
            PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
            h->hashBuildRow1((const block*)input, (u32*)rows, (block*) hash);
        } else {
            PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
            h->hashBuildRow1((const block*)input, (u64*)rows, (block*) hash);
        }  
        
    }

    void hash_build_row32_raw(const void* hasher, const void* input, void* rows, void* hash, size_t bit_length) {
        if (bit_length <= 16) {
            PaxosHash<u16>* h = (PaxosHash<u16>*) hasher;
            h->hashBuildRow32((const block*)input, (u16*)rows, (block*) hash);
        } else if (bit_length <= 32) {
            PaxosHash<u32>* h = (PaxosHash<u32>*) hasher;
            h->hashBuildRow32((const block*)input, (u32*)rows, (block*) hash);
        } else {
            PaxosHash<u64>* h = (PaxosHash<u64>*) hasher;
            h->hashBuildRow32((const block*)input, (u64*)rows, (block*) hash);
        } 
    }
}
