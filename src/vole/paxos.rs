use crate::vole::weight_data::WeightNode;

use super::block::{gf128_inv, Block, ONE_BLOCK, ZERO_BLOCK};
use super::matrix::Matrix;
use super::prng::Prng;
use super::utils::round_up_to;
use super::weight_data::WeightData;
use core::panic;
use log::{error, info};
use rand::Rng;
use std::collections::{BTreeSet, HashSet};
use std::ffi::c_void;

const PAXOS_BUILD_ROW_SIZE: usize = 32;

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
struct FCInv {
    pub mtx: Vec<Vec<usize>>,
}

impl FCInv {
    pub fn new(n: usize) -> FCInv {
        FCInv {
            mtx: vec![vec![]; n],
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
    rows: Matrix<usize>,
    cols: Vec<Vec<usize>>,
    col_backing: Vec<usize>,
    weight_sets: Option<WeightData>,
}

impl Paxos {
    pub fn size(&self) -> usize {
        self.params.sparse_size + self.params.dense_size
    }

    /// solve/encode
    pub fn solve<R: Rng>(&self, inputs: &Vec<Block>, values: Vec<Block>, prng: Option<R>) {
        if self.items_num != inputs.len() {
            panic!("items and input length doesn't match!")
        }
        let val = (self.params.sparse_size + 1) as f64;
        let bit_length = round_up_to(val.log2().ceil() as usize, 8);
    }

    pub fn new(items_num: usize, weight: usize, ssp: usize, seed: Block) -> Paxos {
        let params = PaxosParam::init(items_num, weight, ssp);
        if params.sparse_size + params.dense_size < items_num {
            panic!("params error");
        }
        let hasher = PaxosHash::new::<usize>(seed, weight, params.sparse_size);
        let sparse_size = params.sparse_size;
        Paxos {
            items_num,
            seed,
            params,
            hasher,
            dense: vec![Block::from_i64(0, 0); items_num],
            rows: Matrix::new(items_num, weight, 0),
            cols: vec![vec![]; sparse_size],
            col_backing: vec![0; items_num * weight],
            weight_sets: None,
        }
    }

    pub fn set_input(&mut self, inputs: &Vec<Block>) {
        assert_eq!(self.items_num, inputs.len());

        let mut col_weights = vec![0; self.params.sparse_size];

        // check inputs is unique, TODO: Remove when release build
        {
            let mut input_set: BTreeSet<Block> = BTreeSet::new();
            for i in inputs {
                assert!(input_set.insert(*i))
            }
        }

        let main = inputs.len() / PAXOS_BUILD_ROW_SIZE * PAXOS_BUILD_ROW_SIZE;

        let mut i = 0;
        while i < main {
            let rr = self.rows.mut_row_data(i, PAXOS_BUILD_ROW_SIZE);
            let hash = &mut self.dense[i..i + PAXOS_BUILD_ROW_SIZE];
            self.hasher
                .hash_build_row32(&inputs[i..i + PAXOS_BUILD_ROW_SIZE], rr, hash);

            rr.iter().for_each(|c| {
                col_weights[*c] += 1;
            });

            i += PAXOS_BUILD_ROW_SIZE;
        }

        while i < self.items_num {
            self.hasher.hash_build_row1(
                &inputs[i..i + 1],
                self.rows.mut_row_data(i, 1),
                &mut self.dense[i..i + 1],
            );
            self.rows.row_data(i, 1).iter().for_each(|c| {
                col_weights[*c] += 1;
            });
            i += 1;
        }
        // rebuild columns
        self.rebuild_columns(&col_weights, self.params.weight * self.items_num);

        assert!(self.weight_sets.is_none());
        self.weight_sets = Some(WeightData::init(&col_weights));
    }

    pub fn encode(&mut self, values: &Vec<Block>, output: &mut Vec<Block>, prng: Option<Prng>) {
        if output.len() != self.size() {
            panic!("output size doesn't match");
        }
        let mut main_rows: Vec<usize> = Vec::with_capacity(self.items_num);
        let mut main_cols: Vec<usize> = Vec::with_capacity(self.items_num);

        let mut gap_rows: Vec<[usize; 2]> = Vec::new();

        self.triangulate(&mut main_rows, &mut main_cols, &mut gap_rows);
        output.fill(*ZERO_BLOCK);

        self.backfill(
            &mut main_rows,
            &mut main_cols,
            &mut gap_rows,
            values,
            output,
        );
    }

    pub fn decode(&mut self, inputs: &Vec<Block>, values: &mut Vec<Block>, pp: &Vec<Block>) {
        assert_eq!(pp.len(), self.size());

        let main = inputs.len() / PAXOS_BUILD_ROW_SIZE * PAXOS_BUILD_ROW_SIZE;

        let mut rows = Matrix::new(PAXOS_BUILD_ROW_SIZE, self.params.weight, 0usize);

        let mut dense = vec![Block::from_i64(0, 0); PAXOS_BUILD_ROW_SIZE];

        let mut i: usize = 0;
        while i < main {
            let iter = &inputs[i..i + PAXOS_BUILD_ROW_SIZE];
            self.hasher.hash_build_row32(
                iter,
                rows.mut_row_data(0, PAXOS_BUILD_ROW_SIZE),
                &mut dense,
            );
            self.decode32(
                rows.row_data(0, PAXOS_BUILD_ROW_SIZE),
                &dense,
                &mut values[i..i + PAXOS_BUILD_ROW_SIZE],
                pp,
            );
            i += PAXOS_BUILD_ROW_SIZE;
        }
        while i < inputs.len() {
            let iter = &inputs[i..i + 1];
            self.hasher
                .hash_build_row1(iter, rows.mut_row_data(0, 1), &mut dense);
            self.decode1(rows.row_data(0, 1), &dense[0], &mut values[i], pp);
            i += 1;
        }
    }

    pub fn decode32(
        &mut self,
        rows: &[usize],
        dense: &[Block],
        values: &mut [Block],
        pp: &Vec<Block>,
    ) {
        let weight = self.params.weight;
        for j in 0..4 {
            let rows = &rows[j * 8 * weight..];
            let values = &mut values[j * 8..];
            values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
                let c = rows[weight * i + 0];
                *vv = pp[c];
            });
        }

        for j in 1..weight {
            for k in 0..4 {
                let rows = &rows[k * 8 * weight..];
                let values = &mut values[k * 8..];
                values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
                    let c = rows[weight * i + j];
                    *vv = *vv ^ pp[c];
                });
            }
        }
        let sparse_size = self.params.sparse_size;
        let dense_size = self.params.dense_size;
        let p2: &[Block] = &pp[sparse_size..];
        let mut xx: [Block; 32] = [Block::from_i64(0, 0); 32];
        xx.iter_mut().enumerate().for_each(|(i, x)| {
            *x = dense[i];
        });
        for k in 0..4 {
            let values = &mut values[k * 8..];
            let x = &mut xx[k * 8..];
            values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
                *vv = *vv ^ (p2[0].gf128_mul_reduce(&x[i]));
            });
        }

        for j in 1..dense_size {
            for k in 0..4 {
                let x = &mut xx[k * 8..];
                let dense = &dense[k * 8..];
                let values = &mut values[k * 8..];
                values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
                    x[i] = x[i].gf128_mul_reduce(&dense[i]);
                    *vv = *vv ^ (p2[j].gf128_mul_reduce(&x[i]));
                });
            }
        }
    }

    pub fn decode1(&mut self, rows: &[usize], dense: &Block, value: &mut Block, pp: &[Block]) {
        *value = pp[rows[0]];
        assert_eq!(rows.len(), self.params.weight);
        for j in 1..self.params.weight {
            *value = *value ^ pp[rows[j]];
        }
        let mut x = *dense;
        *value = *value ^ (pp[self.params.sparse_size].gf128_mul_reduce(&x));
        pp[self.params.sparse_size + 1..].iter().for_each(|pp| {
            x = x.gf128_mul_reduce(dense);
            *value = *value ^ pp.gf128_mul_reduce(&x);
        });
    }

    pub fn backfill(
        &mut self,
        main_rows: &mut Vec<usize>,
        main_cols: &mut Vec<usize>,
        gap_rows: &mut Vec<[usize; 2]>,
        x: &Vec<Block>,
        p: &mut Vec<Block>,
    ) {
        assert_eq!(main_rows.len(), main_cols.len());

        let g = gap_rows.len();
        let p2 = &mut p[self.params.sparse_size..];

        assert!(g <= self.params.dense_size);

        if g > 0 {
            let fcinv: FCInv = self.get_fcinv(main_rows, main_cols, gap_rows);
            let size = g;

            let mut EE: Matrix<Block> = Matrix::new(size, size, *ZERO_BLOCK);

            let mut xx = vec![Block::from_i64(0, 0); size];

            // let fcb = vec![Block::from_i64(0, 0); size];

            for i in 0..g {
                let e = self.dense[gap_rows[i][0]];
                let mut ej = e;
                EE[(i, 0)] = e;
                for j in 1..size {
                    ej = ej.gf128_mul_reduce(&e);
                    EE[(i, j)] = ej;
                }
                xx[i] = x[gap_rows[i][0]];
                fcinv.mtx[i].iter().for_each(|&j: &usize| {
                    xx[i] = xx[i] ^ x[j];
                    let fcb = self.dense[j];
                    let mut fcbk = fcb;
                    EE[(i, 0)] = EE[(i, 0)] ^ fcbk;
                    for k in 1..size {
                        fcbk = fcbk.gf128_mul_reduce(&fcb);
                        EE[(i, k)] = EE[(i, k)] ^ fcbk;
                    }
                });
            }

            // PRNG is None, TODO

            EE = gf128_matrix_inv(EE);

            assert!(EE.capacity != 0);

            for i in 0..size {
                let pp = &mut p2[i];
                for j in 0..size {
                    *pp = (*pp) ^ (xx[j].gf128_mul_reduce(&EE[(i, j)]));
                }
            }
        }

        let do_dense = g != 0;

        let mut y = Block::from_i64(0, 0);

        if self.params.weight == 3 {
            main_rows
                .iter()
                .rev()
                .zip(main_cols.iter().rev())
                .for_each(|(&i, &c)| {
                    y = x[i];

                    let row = self.rows.row_data(i, 1);
                    let cc0 = row[0];
                    let cc1 = row[1];
                    let cc2 = row[2];

                    y = y ^ p[cc0];
                    y = y ^ p[cc1];
                    y = y ^ p[cc2];

                    if do_dense {
                        let d = self.dense[i];
                        let mut x = d;
                        y = y ^ p[self.params.sparse_size].gf128_mul_reduce(&x);

                        for i in 1..self.params.dense_size {
                            x = x.gf128_mul_reduce(&d);
                            y = y ^ p[self.params.sparse_size + i].gf128_mul_reduce(&x);
                        }
                    }
                    p[c] = y;
                });
        } else {
            panic!("weight is not 3");
        }
    }

    pub fn get_fcinv(
        &self,
        main_rows: &Vec<usize>,
        main_cols: &Vec<usize>,
        gap_rows: &Vec<[usize; 2]>,
    ) -> FCInv {
        let mut col_mapping: Vec<usize> = Vec::new();
        let mut ret = FCInv::new(gap_rows.len());
        let m = main_rows.len();
        let invert_row_idx = |i: usize| m - i - 1;
        for i in 0..gap_rows.len() {
            if self.rows.row_data(gap_rows[i][0], 1) == self.rows.row_data(gap_rows[i][1], 1) {
                ret.mtx[i].push(gap_rows[i][1]);
            } else {
                if col_mapping.len() == 0 {
                    col_mapping.resize(self.size(), usize::MAX);
                    for i in 0..m {
                        col_mapping[main_cols[invert_row_idx(i)]] = i;
                    }
                }
                let mut row: BTreeSet<usize> = BTreeSet::new();
                for j in 0..self.params.weight {
                    let c1 = self.rows[(gap_rows[i][0], j)];
                    if col_mapping[c1] != usize::MAX {
                        row.insert(col_mapping[c1]);
                    }
                }
                while row.len() > 0 {
                    let c_col = *row.last().unwrap();
                    let c_row = c_col;
                    let h_row = main_rows[invert_row_idx(c_row)];
                    ret.mtx[i].push(h_row);
                    self.rows.row_data(h_row, 1).iter().for_each(|h_col| {
                        let c_col2 = col_mapping[*h_col];
                        if c_col2 != usize::MAX {
                            assert!(c_col2 <= c_col);
                            let iter = row.get(&c_col2);
                            if iter.is_none() {
                                row.insert(c_col2);
                            } else {
                                let target = *iter.unwrap();
                                row.remove(&target);
                            }
                        }
                    });
                    assert!(row.len() == 0 || *row.last().unwrap() != c_col);
                }
            }
        }
        ret
    }

    pub fn triangulate(
        &mut self,
        main_row: &mut Vec<usize>,
        main_col: &mut Vec<usize>,
        gap_rows: &mut Vec<[usize; 2]>,
    ) {
        assert!(self.weight_sets.is_some());

        let mut row_set = vec![0u8; self.items_num];
        let weight_sets = self.weight_sets.as_mut().unwrap();
        while weight_sets.weight_sets.len() > 1 {
            let col = weight_sets.get_min_weightnode();
            weight_sets.pop_node(col);

            unsafe {
                (*col).weight = 0;
                let col_index = weight_sets.idx_of(&(*col));
                let mut first: bool = true;
                self.cols[col_index].iter().for_each(|&row_idx| {
                    if row_set[row_idx] == 0 {
                        row_set[row_idx] = 1;
                        self.rows.row_data(row_idx, 1).iter().for_each(|&col_idx2| {
                            let node: *mut WeightNode =
                                &mut weight_sets.nodes[col_idx2] as *mut WeightNode;

                            if (*node).weight != 0 {
                                weight_sets.pop_node(node as *mut WeightNode);
                                (*node).weight -= 1;
                                weight_sets.push_node(node);
                                // TODO prefetch next column
                            }
                        });

                        if first {
                            main_col.push(col_index);
                            main_row.push(row_idx);
                            first = false;
                        } else {
                            assert_ne!(*main_row.last().unwrap(), row_idx);
                            gap_rows.push([row_idx, *main_row.last().unwrap()]);
                        }
                    }
                });
                assert_eq!(first, false);
            }
        }
    }

    pub fn rebuild_columns(&mut self, col_weights: &[usize], total_weight: usize) {
        assert_eq!(self.col_backing.len(), total_weight);

        // TODO: remove when release
        let mut col_sum: usize = 0;
        col_weights.iter().for_each(|x| {
            col_sum += x;
        });
        assert_eq!(col_sum, total_weight);

        if self.rows.cols() == 3 {
            for i in 0..self.items_num {
                // rows
                let row = self.rows.row_data(i, 1);
                self.cols[row[0]].push(i);
                self.cols[row[1]].push(i);
                self.cols[row[2]].push(i);
            }
            // copy cols to colbacking
            let mut col_backing_start = 0;
            self.cols.iter().for_each(|x| {
                if x.len() == 0 {
                    return ;
                }
                assert!(col_backing_start < self.col_backing.len());
                self.col_backing[col_backing_start..col_backing_start + x.len()].copy_from_slice(x);
                col_backing_start += x.len();
            });
        } else {
            panic!("weight is not 3, TODO");
        }
    }
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
            let idx_size: usize = idbit_length / 8;

            let pointer: *const c_void = new_paxos_hash(idbit_length);
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

    #[test]
    fn paxos_encode_test() {
        let n: usize = 1 << 20;
        let w = 3usize;
        let s = 0usize;
        let t = 1usize;
        
        for tt in 0..t {
            let mut paxos = Paxos::new(n, w, 40, *ZERO_BLOCK);
            let mut paxos2 = Paxos::new(n, w, 40, *ZERO_BLOCK);
            let prng = Prng::new(Block::from_i64(tt as i64, s as i64));
            let items = prng.get_blocks(n);
            let values = prng.get_blocks(n);
            let mut values2 = vec![Block::from_i64(0, 0); n];
            let mut p = vec![Block::from_i64(0, 0); paxos.size()];
            paxos.set_input(&items);
            paxos2.set_input(&items);
            for i in 0..paxos.rows.rows() {
                for j in 0..w {
                    assert_eq!(paxos.rows[(i, j)], paxos2.rows[(i, j)]);
                }
            }
            paxos.encode(&values, &mut p, None);
            paxos.decode(&items, &mut values2, &p);
            values.iter().zip(values2.iter()).enumerate().for_each(|(i, (x1, x2))| {
                assert_eq!(x1, x2);
            });

        }
    }
}
