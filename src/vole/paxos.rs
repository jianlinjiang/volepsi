use super::block::{gf128_inv, Block, ONE_BLOCK, ZERO_BLOCK};
use super::matrix::Matrix;
use super::prng::Prng;
use super::utils::{div_ceil, round_up_to};
use core::panic;
use log::{error, info};
use rand::Rng;
use std::ffi::c_void;
use super::weight_data::WeightData;
#[link(name = "rcrypto")]
extern "C" {
    pub fn new_paxos_hash(bit_length: usize) -> *const c_void;
    pub fn delete_paxos_hash(pointer: *const c_void, bit_length: usize);
    pub fn init_paxos_hash(
        pointer: *const c_void,
        seed: *const c_void,
        weight: usize,
        paxos_size: usize,
        bit_length: usize,
    );
    pub fn build_row_raw(
        pointer: *const c_void,
        hash: *const c_void,
        row: *mut c_void,
        bit_length: usize,
    );
    pub fn build_row32_raw(
        pointer: *const c_void,
        hash: *const c_void,
        row: *mut c_void,
        bit_length: usize,
    );

    pub fn hash_build_row1_raw(
        pointer: *const c_void,
        input: *const c_void,
        row: *mut c_void,
        hash: *mut c_void,
        bit_length: usize,
    );

    pub fn hash_build_row32_raw(
        pointer: *const c_void,
        input: *const c_void,
        row: *mut c_void,
        hash: *mut c_void,
        bit_length: usize,
    );
}

pub fn gf128_matrix_inv(mut mtx: Matrix<Block>) -> Matrix<Block> {
    assert_eq!(mtx.rows(), mtx.cols());

    let n = mtx.rows();

    let mut inv = Matrix::new(n, n, *ZERO_BLOCK);
    for i in 0..n {
        inv[(i, i)] = *ONE_BLOCK;
    }

    for i in 0..n {
        if mtx[(i, i)] == *ZERO_BLOCK {
            for j in i + 1..n {
                if mtx[(j, i)] == *ONE_BLOCK {
                    // swap rows, can be optimized
                    for k in 0..n {
                        let tmp = mtx[(j, k)];
                        mtx[(j, k)] = mtx[(i, k)];
                        mtx[(i, k)] = tmp;

                        let tmp = inv[(j, k)];
                        inv[(j, k)] = inv[(i, k)];
                        inv[(i, k)] = tmp;
                    }
                    break;
                }
            }

            if mtx[(i, i)] == *ZERO_BLOCK {
                panic!("the matrix is not invertable");
            }
        }

        let mtx_ii_inv = gf128_inv(&mtx[(i, i)]);
        for j in 0..n {
            mtx[(i, j)] = mtx[(i, j)].gf128_mul_reduce(&mtx_ii_inv);
            inv[(i, j)] = inv[(i, j)].gf128_mul_reduce(&mtx_ii_inv);
        }

        for j in 0..n {
            if j != i {
                let mtx_ji = mtx[(j, i)];
                for k in 0..n {
                    mtx[(j, k)] = mtx[(j, k)] ^ (mtx[(i, k)].gf128_mul_reduce(&mtx_ji));
                    inv[(j, k)] = inv[(j, k)] ^ (inv[(i, k)].gf128_mul_reduce(&mtx_ji));
                }
            }
        }
    }

    inv
}

pub fn gf128_matrix_mul(m0: &Matrix<Block>, m1: &Matrix<Block>) -> Matrix<Block> {
    assert_eq!(m0.cols(), m1.rows());
    let mut ret: Matrix<Block> = Matrix::new(m0.rows(), m1.cols(), *ZERO_BLOCK);
    for i in 0..ret.rows() {
        for j in 0..ret.cols() {
            let v: &mut Block = &mut ret[(i, j)];
            for k in 0..m0.cols() {
                *v = *v ^ m0[(i, k)].gf128_mul_reduce(&m1[(k, j)]);
            }
        }
    }
    ret
}

#[derive(Clone, Debug)]
struct PaxosParam {
    sparse_size: usize,
    dense_size: usize,
    weight: usize,
    g: usize,
    ssp: usize, // static safe parameter = 40
}

impl PaxosParam {
    pub fn init(items_num: usize, weight: usize, ssp: usize) -> PaxosParam {
        if weight < 2 {
            panic!("weight must be 2 or greater");
        }
        let log_n = (items_num as f64).log2();
        let mut dense_size = 0;
        let mut sparse_size = 0;
        let mut g: usize = 0;
        if weight == 2 {
            let a = 7.529;
            let b = 0.61;
            let c = 2.556;
            let lambda_vs_gap = a / (log_n - c) + b;

            g = (ssp as f64 / lambda_vs_gap + 1.9).ceil() as usize;
            dense_size = g;
            sparse_size = 2 * items_num;
        } else {
            let mut ee: f64 = 0.0;
            if weight == 3 {
                ee = 1.223;
            } else if weight == 4 {
                ee = 1.293;
            } else {
                ee = 0.1485 * weight as f64 + 0.6845;
            }

            let log_w = (weight as f64).log2();
            let log_lambda_vs_e = 0.555 * log_n + 0.093 * log_w.powi(3) - 1.01 * log_w.powi(2)
                + 2.925 * log_w
                - 0.133;
            // let lambda_vs_e = log_lambda_vs_e.powi(2);
            let lambda_vs_e = (2.0_f64).powf(log_lambda_vs_e);
            let b = -9.2 - lambda_vs_e * ee;
            let e = (ssp as f64 - b) / lambda_vs_e;
            g = ((ssp as f64) / ((weight as f64 - 2.0) * (e * items_num as f64).log2())).floor()
                as usize;

            dense_size = g;
            sparse_size = (items_num as f64 * e).floor() as usize;
        }

        PaxosParam {
            sparse_size,
            dense_size,
            weight,
            g,
            ssp,
        }
    }
}

#[derive(Clone, Debug)]
struct Paxos {
    items_num: usize,
    seed: Block,
    params: PaxosParam,
    hasher: PaxosHash,
    dense: Vec<Block>,
    rows: Matrix<u64>,
    cols: Vec<Vec<u64>>,
    col_backing: Vec<u64>,
    weight_sets: Option<WeightData>
}

impl Paxos
{
    pub fn size(&self) -> usize {
        self.params.sparse_size + self.params.dense_size
    }

    /// solve/encode
    pub fn solve<R: Rng>(&self, inputs: &Vec<Block>, values: Vec<Block>, prng: Option<R>) {
        if self.items_num != inputs.len() {
            panic!("items and input length doesn't match!")
        }
        let val = (self.params.sparse_size + 1) as f64;
        let bit_length = round_up_to(val.log2().ceil() as u64, 8);
    }

    pub fn new(items_num: usize, weight: usize, ssp: usize, seed: Block) -> Paxos {
        let params = PaxosParam::init(items_num, weight, ssp);
        if params.sparse_size + params.dense_size < items_num {
            panic!("params error");
        }
        let hasher = PaxosHash::new::<u64>(seed, weight, params.sparse_size);
        Paxos {
            items_num,
            seed,
            params,
            hasher,
            dense: Vec::new(),
            rows: Matrix::new(0, 0, 0),
            cols: Vec::new(),
            col_backing: Vec::new(),
            weight_sets: None
        }
    }

    pub fn encode(&self, values: &Vec<Block>, output: &mut Vec<Block>, prng: Option<Prng>) {
        if output.len() != self.size() {
            panic!("output size doesn't match");
        }
        let mut main_rows: Vec<u64> = Vec::with_capacity(self.items_num);
        let mut main_cols: Vec<u64> = Vec::with_capacity(self.items_num);

        let mut gap_rows: Vec<[u64; 2]>;
    }

    // pub fn triangulate(&self, )
}

/// convert row items to sparse vector with length m'
#[derive(Clone, Debug)]
struct PaxosHash {
    weight: usize,
    sparse_size: usize,
    idx_size: usize,
    pointer: *const c_void,
    idbit_length: usize,
}

impl PaxosHash {
    pub fn new<T>(seed: Block, weight: usize, paxos_size: usize) -> PaxosHash {
        unsafe {
            let idbit_length = std::mem::size_of::<T>() * 8;
            let bit_length = round_up_to((paxos_size as f64).log2().ceil() as u64, 8) as usize;
            assert_eq!(bit_length, idbit_length);

            let idx_size = bit_length / 8;

            let pointer: *const c_void = new_paxos_hash(bit_length);
            init_paxos_hash(
                pointer,
                &seed as *const Block as *const c_void,
                weight,
                paxos_size,
                idbit_length,
            );
            PaxosHash {
                weight,
                sparse_size: paxos_size,
                idx_size,
                pointer,
                idbit_length,
            }
        }
    }

    pub fn build_row<T>(&self, hash: Block, row: &mut [T]) {
        unsafe {
            build_row_raw(
                self.pointer,
                &hash as *const Block as *const c_void,
                row.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn build_row32<T>(&self, hash: &[Block], row: &mut [T]) {
        unsafe {
            build_row32_raw(
                self.pointer,
                hash.as_ptr() as *const c_void,
                row.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn hash_build_row1<T>(&self, input: &[Block], rows: &mut [T], hash: &mut [Block]) {
        unsafe {
            hash_build_row1_raw(
                self.pointer,
                input.as_ptr() as *const c_void,
                rows.as_mut_ptr() as *mut c_void,
                hash.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn hash_build_row32<T>(&self, input: &[Block], rows: &mut [T], hash: &mut [Block]) {
        unsafe {
            hash_build_row32_raw(
                self.pointer,
                input.as_ptr() as *const c_void,
                rows.as_mut_ptr() as *mut c_void,
                hash.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }
}

impl Drop for PaxosHash {
    fn drop(&mut self) {
        unsafe {
            delete_paxos_hash(self.pointer, self.idbit_length);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(arithmetic_overflow)]
    use super::super::block::ZERO_BLOCK;
    use super::super::matrix::Matrix;
    use super::super::prng::Prng;
    use super::*;
    use rand::thread_rng;
    #[test]
    fn matrix_inv_test() {
        let mut rng = thread_rng();
        let n = 10;
        let mut mtx = Matrix::new(n, n, *ZERO_BLOCK);
        for i in 0..n {
            for j in 0..n {
                mtx[(i, j)] = Block::rand(&mut rng);
            }
        }
        let inv = gf128_matrix_inv(mtx.clone());
        let I: Matrix<Block> = gf128_matrix_mul(&inv, &mtx);
        for i in 0..n {
            assert_eq!(I[(i, i)], *ONE_BLOCK);
            for j in 0..n {
                if i != j && I[(i, j)] != *ZERO_BLOCK {
                    assert!(false);
                }
            }
        }
    }

    #[test]
    fn paxos_params_test() {
        let param = PaxosParam::init(1 << 20, 3, 40);
        println!("{:?}", param);
    }

    #[test]
    fn paxos_buildrow_test() {
        let n = 1 << 10;
        let t = 1 << 4;
        let s = 0; // seed
        let h = PaxosHash::new::<u16>(Block::from_i64(0, s), 3, n);
        let prng = Prng::new(Block::from_i64(1, s));
        let exp: [[u16; 3]; 32] = [
            [858, 751, 414],
            [677, 590, 375],
            [857, 240, 0],
            [18, 373, 879],
            [990, 62, 458],
            [894, 667, 301],
            [1023, 438, 301],
            [532, 815, 202],
            [64, 507, 82],
            [664, 739, 158],
            [4, 523, 573],
            [719, 282, 86],
            [156, 396, 473],
            [810, 916, 850],
            [959, 1017, 449],
            [3, 841, 546],
            [703, 146, 19],
            [935, 983, 830],
            [689, 804, 550],
            [237, 661, 393],
            [25, 817, 387],
            [112, 531, 45],
            [799, 747, 158],
            [986, 444, 949],
            [916, 954, 410],
            [736, 219, 732],
            [111, 628, 750],
            [272, 627, 160],
            [191, 610, 628],
            [1018, 213, 894],
            [1, 609, 948],
            [570, 60, 896],
        ];
        // let mut hash: [Block; 32] = [Block::from_i64(0, 0); 32];
        let mut rows = Matrix::<u16>::new(32, 3, 0);
        for tt in 0..t {
            let hash = prng.get_blocks(32);
            h.build_row32(&hash, rows.mut_data());

            for i in 0..32 {
                let mut rr: [u16; 3] = [0; 3];
                h.build_row(hash[i], &mut rr);

                for j in 0..3 {
                    assert_eq!(rows[(i, j)], rr[j]);

                    if tt == 0 {
                        assert_eq!(rows[(i, j)], exp[i][j]);
                    }
                }
            }
        }
    }
}
