use rand::Rng;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::{Ordering, PartialEq};
use std::ops::{Add, BitAnd, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub};
#[derive(Copy, Clone, Debug)]
pub struct Block(pub __m128i);

lazy_static! {
    pub static ref ZERO_BLOCK: Block = Block::from_i64(0, 0);
    pub static ref ONE_BLOCK: Block = Block::from_i64(0, 1);
    pub static ref ALL_ONE_BLOCK: Block = Block::from_i64(i64::MIN, i64::MIN);
    pub static ref ZERO_AND_ALL_ONE: [Block; 2] = [*ZERO_BLOCK, *ALL_ONE_BLOCK];
    #[allow(arithmetic_overflow)]
    pub static ref CC_BLOCK: Block = Block::from_i64(-3689348814741910324i64, -3689348814741910324i64);
}

impl Block {
    pub fn from_i64(x1: i64, x2: i64) -> Self {
        Block(unsafe { _mm_set_epi64x(x1, x2) })
    }

    fn gf128_mul(&self, y: &Block) -> (Block, Block) {
        unsafe {
            let x = *self;
            let mut t1 = _mm_clmulepi64_si128(x.0, y.0, 0x00);
            let mut t2 = _mm_clmulepi64_si128(x.0, y.0, 0x10);
            let mut t3 = _mm_clmulepi64_si128(x.0, y.0, 0x01);
            let mut t4 = _mm_clmulepi64_si128(x.0, y.0, 0x11);
            t2 = _mm_xor_si128(t2, t3);
            t3 = _mm_slli_si128(t2, 8);
            t2 = _mm_srli_si128(t2, 8);
            t1 = _mm_xor_si128(t1, t3);
            t4 = _mm_xor_si128(t4, t2);
            (Block(t1), Block(t4))
        }
    }

    fn gf128_reduce(&self, x1: Block) -> Block {
        unsafe {
            let mut mul256_low = self.0;
            let mut mul256_high = x1.0;
            let m: __m128i = _mm_set_epi32(0, 0, 0, 0b10000111);
            let modulus = _mm_loadl_epi64(&m as *const __m128i);
            let mut tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x01);
            mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(tmp, 8));
            mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(tmp, 8));
            tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x00);
            mul256_low = _mm_xor_si128(mul256_low, tmp);
            Block(mul256_low)
        }
    }

    pub fn gf128_mul_reduce(&self, y: &Block) -> Block {
        let (xy1, xy2) = self.gf128_mul(y);
        xy1.gf128_reduce(xy2)
    }

    fn gf128_pow(&self, i: u64) -> Block {
        if *self == Block::from_i64(0, 0) {
            return Block::from_i64(0, 0);
        }

        let mut pow2 = *self;
        let mut s = Block::from_i64(0, 1);
        let mut i = i;
        while i != 0 {
            if i & 1 != 0 {
                s = s.gf128_reduce(pow2);
            }
            pow2 = pow2.gf128_reduce(pow2);
            i >>= 1;
        }
        s
    }

    pub fn rand<R: Rng>(rng: &mut R) -> Self {
        Block::from_i64(rng.next_u64() as i64, rng.next_u64() as i64)
    }

    pub fn clear(&mut self) {
        unsafe {
            self.0 = _mm_setzero_si128();
        }
    }
}

impl BitXor for Block {
    type Output = Block;

    fn bitxor(self, rhs: Block) -> Block {
        Block(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }
}

impl BitXorAssign for Block {
    fn bitxor_assign(&mut self, rhs: Block) {
        *self = Block(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }
}

impl Not for Block {
    type Output = Block;
    fn not(self) -> Block {
        Block(unsafe { _mm_xor_si128(self.0, _mm_set_epi64x(-1, -1)) })
    }
}

impl BitAnd for Block {
    type Output = Block;
    fn bitand(self, rhs: Block) -> Block {
        Block(unsafe { _mm_and_si128(self.0, rhs.0) })
    }
}

impl BitOr for Block {
    type Output = Block;
    fn bitor(self, rhs: Block) -> Block {
        Block(unsafe { _mm_or_si128(self.0, rhs.0) })
    }
}

impl BitOrAssign for Block {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = Block(unsafe { _mm_or_si128(self.0, rhs.0) })
    }
}

impl Add for Block {
    type Output = Block;
    fn add(self, other: Block) -> Block {
        Block(unsafe { _mm_add_epi64(self.0, other.0) })
    }
}

impl Sub for Block {
    type Output = Block;

    fn sub(self, other: Block) -> Block {
        Block(unsafe { _mm_sub_epi64(self.0, other.0) })
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Block) -> bool {
        unsafe {
            let c = _mm_xor_si128(self.0, other.0);
            _mm_test_all_zeros(c, c) != 0
        }
    }
}

impl Eq for Block {}

impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        unsafe {
            let lhsa = &self.0 as *const __m128i as *const u64;
            let rhsa = &other.0 as *const __m128i as *const u64;
            if *(lhsa.offset(1)) < *(rhsa.offset(1)) {
                Some(Ordering::Less)
            } else if *(lhsa.offset(1)) == *(rhsa.offset(1)) && *lhsa < *rhsa {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            }
        }
    }
}

impl Ord for Block {
    fn cmp(&self, other: &Self) -> Ordering {
        unsafe {
            let lhsa = &self.0 as *const __m128i as *const u64;
            let rhsa = &other.0 as *const __m128i as *const u64;
            if *(lhsa.offset(1)) < *(rhsa.offset(1)) {
                Ordering::Less
            } else if *(lhsa.offset(1)) == *(rhsa.offset(1)) && *lhsa < *rhsa {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
    }
}

pub fn gf128_inv(x: &Block) -> Block {
    let mut a = *x;
    let mut result = *ZERO_BLOCK;
    for i in 0..=6 {
        let mut b = a;
        let mut j = 0;
        while j < (1 << i) {
            b = b.gf128_mul_reduce(&b);
            j += 1;
        }
        a = a.gf128_mul_reduce(&b);

        if i == 0 {
            result = b;
        } else {
            result = result.gf128_mul_reduce(&b);
        }
    }
    assert_eq!(result.gf128_mul_reduce(x), *ONE_BLOCK);
    result
}

#[cfg(test)]
mod tests {
    #![allow(arithmetic_overflow)]
    use super::*;
    use rand::{thread_rng, RngCore};
    use std::time::{Duration, Instant};
    #[test]
    fn block_inv_test() {
        let mut rng = thread_rng();
        for i in 0..10000 {
            let a = Block::rand(&mut rng);
            let b = gf128_inv(&a);
            let c = a.gf128_mul_reduce(&b);
            assert_eq!(c, *ONE_BLOCK);
        }
    }

    #[test]
    fn block_add_test() {
        let mut rng = thread_rng();
        for i in 0..10000 {
            let a = Block::rand(&mut rng);
            let b = Block::rand(&mut rng);
            let c = a + b;
            let d = a ^ b;
            assert_eq!(c, d);
        }
    }
}
