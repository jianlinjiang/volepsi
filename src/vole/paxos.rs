use core::panic;

use super::block::{gf128_inv, Block, ONE_BLOCK, ZERO_BLOCK};
use super::matrix::Matrix;
use super::utils::{div_ceil, round_up_to};
use log::{error, info};
use rand::Rng;
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
    params: PaxosParam,
}

impl Paxos {
    /// solve/encode
    pub fn solve<T: Copy + PartialEq, R: Rng>(
        &self,
        inputs: &Vec<Block>,
        values: Vec<Block>,
        prng: Option<R>,
    ) {
        if self.items_num != inputs.len() {
            panic!("items and input length doesn't match!")
        }
        let val = (self.params.sparse_size + 1) as f64;
        let bit_length = round_up_to(val.log2().ceil() as u64, 8);
    }

    pub fn init<T: Copy + PartialEq>(items_num: usize, weight: usize, ssp: usize, seed: Block) {
        let params = PaxosParam::init(items_num, weight, ssp);
        if params.sparse_size + params.dense_size < items_num {
            panic!("params error");
        }
    }
}

/// convert row items to sparse vector with length m'
#[derive(Clone)]
struct PaxosHash {}

#[cfg(test)]
mod tests {
    #![allow(arithmetic_overflow)]
    use super::super::block::ZERO_BLOCK;
    use super::*;
    use rand::{thread_rng, RngCore};
    use std::time::{Duration, Instant};
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
}
