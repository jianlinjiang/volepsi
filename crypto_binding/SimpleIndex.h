#pragma once
// © 2016 Peter Rindal.
// © 2022 Visa.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cryptoTools/Common/Defines.h"
#include "cryptoTools/Common/Matrix.h"
#include "cryptoTools/Common/BitVector.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "coproto/coproto.h"

namespace volePSI
{
    using u64 = oc::u64;
	using u32 = oc::u32;
	using u16 = oc::u16;
	using u8 = oc::u8;

	using i64 = oc::i64;
	using i32 = oc::i32;
	using i16 = oc::i16;
	using i8 = oc::i8;

	using block  = oc::block;

	template<typename T>
	using span = oc::span<T>;
	template<typename T>
	using Matrix = oc::Matrix<T>;
	template<typename T>
	using MatrixView = oc::MatrixView<T>;

	enum Mode {
		Sender = 1,
		Receiver = 2
		//Dual = 3
	};

	struct RequiredBase
	{
		u64 mNumSend;
		oc::BitVector mRecvChoiceBits;
	};


	using PRNG = oc::PRNG;
	using Socket = coproto::Socket;
	using Proto = coproto::task<void>;

    class SimpleIndex
    {
    public:


        struct Item
        {
            Item() :mVal(-1) {}

            Item& operator=(const Item&) = default;

            bool isEmpty() const { return mVal == u64(-1); }

            // The index is the index of the input that currently
            // occupies this bin position. The index is encode in the
            // first 7 bytes.
            u64 idx() const { return mVal  & (u64(-1) >> 8); }

            // The index of the hash function that this item is 
            // currently using. This in is encoded in the 8th byte.
            u64 hashIdx() const { return ((u8*)&mVal)[7] & 127; }

            // Return true if this item was set with a collition.
            bool isCollision() const { return  (((u8*)&mVal)[7] >> 7) > 0; }

            // The this item to contain the index idx under the given hash index.
            // The collision value is also encoded.
            void set(u64 idx, u8 hashIdx, bool collision)
            {
                mVal = idx;
                ((u8*)&mVal)[7] = hashIdx | ((collision & 1) << 7);
            }
#ifdef THREAD_SAFE_SIMPLE_INDEX
            Item(const Item& b) : mVal(b.mVal.load(std::memory_order_relaxed)) {}
            Item(Item&& b) : mVal(b.mVal.load(std::memory_order_relaxed)) {}
            std::atomic<u64> mVal;
#else
            Item(const Item& b) : mVal(b.mVal) {}
            Item(Item&& b) : mVal(b.mVal) {}
            u64 mVal;
#endif
        };

        u64 mMaxBinSize, mNumHashFunctions;

        // The current assignment of items to bins. Only
        // the index of the input item is stored in the bin,
        // not the actual item itself.
        Matrix<Item> mBins;
        u64 mNumBins;

        // numBalls x mNumHashFunctions matrix, (i,j) contains the i'th items
        // hash value under hash index j.
        Matrix<u64> mItemToBinMap;

        // The some of each bin.
        std::vector<u64> mBinSizes;

        block mHashSeed;
        void print() ;
        static  u64 get_bin_size(u64 numBins, u64 numBalls, u64 statSecParam, bool approx = true);
        

        void init(u64 numBins, u64 numBalls, u64 statSecParam = 40, u64 numHashFunction = 3);
        void insertItems(span<block> items, block hashingSeed);
    };

}
