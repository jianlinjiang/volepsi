use super::block::{gf128_inv, Block, ONE_BLOCK, ZERO_BLOCK};
use super::matrix::Matrix;
use super::prng::Prng;
use super::weight_data::WeightData;
use crate::vole::weight_data::WeightNode;
use core::panic;
use log::debug;
use std::collections::BTreeSet;
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
    debug_assert_eq!(mtx.rows(), mtx.cols());

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
    debug_assert_eq!(m0.cols(), m1.rows());
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
pub struct PaxosParam {
    pub sparse_size: usize,
    pub dense_size: usize,
    pub weight: usize,
    pub g: usize,
    pub ssp: usize, // static safe parameter = 40
}

impl PaxosParam {
    pub fn init(items_num: usize, weight: usize, ssp: usize) -> PaxosParam {
        if weight < 2 {
            panic!("weight must be 2 or greater");
        }
        let log_n = (items_num as f64).log2();
        let dense_size;
        let sparse_size;
        let g;
        if weight == 2 {
            let a = 7.529;
            let b = 0.61;
            let c = 2.556;
            let lambda_vs_gap = a / (log_n - c) + b;

            g = (ssp as f64 / lambda_vs_gap + 1.9).ceil() as usize;
            dense_size = g;
            sparse_size = 2 * items_num;
        } else {
            let ee: f64;
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

    pub fn size(&self) -> usize {
        self.sparse_size + self.dense_size
    }
}

#[derive(Clone, Debug)]
pub struct FCInv {
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
pub struct Paxos {
    pub items_num: usize,
    pub seed: Block,
    pub params: PaxosParam,
    pub hasher: PaxosHash,
    pub dense: Vec<Block>,
    pub rows: Matrix<usize>,   // 稀疏矩阵每行只有weight个元素为1，记录1的列数
    pub cols: Vec<Vec<usize>>, // 记录每列有哪些行为1
    pub col_backing: Vec<usize>, // 用来做备份的记录
    pub weight_sets: Option<WeightData>,
}

impl Paxos {
    pub fn size(&self) -> usize {
        self.params.sparse_size + self.params.dense_size
    }

    pub fn new(items_num: usize, weight: usize, ssp: usize, seed: Block) -> Paxos {
        let params = PaxosParam::init(items_num, weight, ssp);
        if params.sparse_size + params.dense_size < items_num {
            panic!("params error");
        }
        let hasher: PaxosHash = PaxosHash::new(seed, weight, params.sparse_size);
        let sparse_size = params.sparse_size;
        Paxos {
            items_num,
            seed,
            params,
            hasher,
            dense: vec![*ZERO_BLOCK; items_num],
            rows: Matrix::new(items_num, weight, 0),
            cols: vec![vec![]; sparse_size],
            col_backing: vec![0; items_num * weight],
            weight_sets: None,
        }
    }

    pub fn new_with_params(items_num: usize, params: &PaxosParam, seed: Block) -> Paxos {
        let weight = params.weight;
        let hasher = PaxosHash::new(seed, weight, params.sparse_size);
        Paxos {
            items_num,
            seed,
            params: params.clone(),
            hasher,
            dense: vec![Block::from_i64(0, 0); items_num],
            rows: Matrix::new(items_num, weight, 0),
            cols: vec![vec![]; params.sparse_size],
            col_backing: vec![0; items_num * weight],
            weight_sets: None,
        }
    }

    pub fn set_input(&mut self, inputs: &Vec<Block>) {
        debug_assert_eq!(self.items_num, inputs.len());

        let mut col_weights = vec![0; self.params.sparse_size];

        // TODO: Remove when release build
        // {
        //     let mut input_set: BTreeSet<Block> = BTreeSet::new();
        //     for i in inputs {
        //         debug_assert!(input_set.insert(*i))
        //     }
        // }
        let mut i = 0;
        // 32 行一次操作
        self.rows
            .storage
            .iter_mut()
            .step_by(PAXOS_BUILD_ROW_SIZE * self.params.weight)
            .zip(self.dense.iter_mut().step_by(PAXOS_BUILD_ROW_SIZE))
            .zip(inputs.iter().step_by(PAXOS_BUILD_ROW_SIZE))
            .for_each(|((row, dense), input)| {
                if i + PAXOS_BUILD_ROW_SIZE > inputs.len() {
                    return;
                }
                self.hasher.hash_build_row32_pointer(
                    input as *const Block,
                    row as *mut usize,
                    dense as *mut Block,
                );
                i += PAXOS_BUILD_ROW_SIZE;
            });
        debug_assert_eq!(
            self.rows.storage[i * self.params.weight..].len() / self.params.weight,
            self.dense[i..].len()
        );
        debug_assert_eq!(
            self.rows.storage[i * self.params.weight..].len() / self.params.weight,
            inputs[i..].len()
        );
        self.rows.storage[i * self.params.weight..]
            .iter_mut()
            .step_by(self.params.weight)
            .zip(self.dense[i..].iter_mut())
            .zip(inputs[i..].iter())
            .for_each(|((row, dense), input)| {
                self.hasher
                    .hash_build_row1_pointer(input, row as *mut usize, dense);
            });
        self.rows.storage.iter().for_each(|&x| {
            col_weights[x] += 1;
        });
        // rebuild columns
        self.rebuild_columns(&col_weights, self.params.weight * self.items_num);

        debug_assert!(self.weight_sets.is_none());
        self.weight_sets = Some(WeightData::init(&col_weights));
    }

    pub fn encode(&mut self, values: &[Block], output: &mut [Block], _prng: Option<Prng>) {
        if output.len() != self.size() {
            panic!("output size doesn't match");
        }
        debug_assert_eq!(self.items_num, values.len());
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
        debug_assert_eq!(pp.len(), self.size());

        let main = inputs.len() / PAXOS_BUILD_ROW_SIZE * PAXOS_BUILD_ROW_SIZE;

        let mut rows = Matrix::new(PAXOS_BUILD_ROW_SIZE, self.params.weight, 0usize);

        let mut dense = vec![*ZERO_BLOCK; PAXOS_BUILD_ROW_SIZE];

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
        pp: &[Block],
    ) {
        debug_assert_eq!(rows.len(), 32 * 3);
        debug_assert_eq!(dense.len(), 32);
        debug_assert_eq!(values.len(), 32);
        debug_assert_eq!(pp[self.params.sparse_size..].len(), self.params.dense_size);

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
        xx.copy_from_slice(dense);
        // for k in 0..4 {
        //     let values = &mut values[k * 8..];
        //     let x = &mut xx[k * 8..];
        //     values

        //     values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
        //         *vv = *vv ^ (p2[0].gf128_mul_reduce(&x[i]));
        //     });
        // }

        values.iter_mut().zip(xx).for_each(|(vv, x)| {
            *vv = *vv ^ p2[0].gf128_mul_reduce(&x);
        });

        for j in 1..dense_size {
            values
                .iter_mut()
                .zip(xx.iter_mut())
                .zip(dense.iter())
                .for_each(|((vv, x), d)| {
                    *x = x.gf128_mul_reduce(d);
                    *vv = *vv ^ (p2[j].gf128_mul_reduce(x));
                });
            // for k in 0..4 {
            //     let x = &mut xx[k * 8..];
            //     let dense = &dense[k * 8..];
            //     let values = &mut values[k * 8..];
            //     values[0..8].iter_mut().enumerate().for_each(|(i, vv)| {
            //         x[i] = x[i].gf128_mul_reduce(&dense[i]);
            //         *vv = *vv ^ (p2[j].gf128_mul_reduce(&x[i]));
            //     });
            // }
        }
    }

    pub fn decode1(&mut self, rows: &[usize], dense: &Block, value: &mut Block, pp: &[Block]) {
        *value = pp[rows[0]];
        debug_assert_eq!(rows.len(), self.params.weight);
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
        x: &[Block],
        p: &mut [Block],
    ) {
        debug_assert_eq!(main_rows.len(), main_cols.len());

        let g = gap_rows.len();
        let p2 = &mut p[self.params.sparse_size..];

        debug!("g: {}, dense_size: {}", g, self.params.dense_size);
        debug_assert!(g <= self.params.dense_size);

        if g > 0 {
            let fcinv: FCInv = self.get_fcinv(main_rows, main_cols, gap_rows);
            let size = g;

            let mut ee: Matrix<Block> = Matrix::new(size, size, *ZERO_BLOCK);

            let mut xx = vec![Block::from_i64(0, 0); size];

            // let fcb = vec![Block::from_i64(0, 0); size];

            for i in 0..g {
                let e = self.dense[gap_rows[i][0]];
                let mut ej = e;
                ee[(i, 0)] = e;
                for j in 1..size {
                    ej = ej.gf128_mul_reduce(&e);
                    ee[(i, j)] = ej;
                }
                xx[i] = x[gap_rows[i][0]];
                fcinv.mtx[i].iter().for_each(|&j: &usize| {
                    xx[i] = xx[i] ^ x[j];
                    let fcb = self.dense[j];
                    let mut fcbk = fcb;
                    ee[(i, 0)] = ee[(i, 0)] ^ fcbk;
                    for k in 1..size {
                        fcbk = fcbk.gf128_mul_reduce(&fcb);
                        ee[(i, k)] = ee[(i, k)] ^ fcbk;
                    }
                });
            }

            // PRNG is None, TODO

            ee = gf128_matrix_inv(ee);

            debug_assert!(ee.capacity != 0);

            for i in 0..size {
                let pp = &mut p2[i];
                for j in 0..size {
                    *pp = (*pp) ^ (xx[j].gf128_mul_reduce(&ee[(i, j)]));
                }
            }
        }

        let do_dense = g != 0;

        let mut y = *ZERO_BLOCK;

        if self.params.weight == 3 {
            main_rows
                .iter()
                .rev()
                .zip(main_cols.iter().rev())
                .for_each(|(&i, &c)| {
                    y = x[i];

                    let row = self.rows.row_data(i, 1);
                    row.iter().for_each(|&cc| y = y ^ p[cc]);

                    if do_dense {
                        let d = self.dense[i];
                        let mut x = d;
                        y = y ^ p[self.params.sparse_size].gf128_mul_reduce(&x);

                        p[self.params.sparse_size + 1..].iter().for_each(|pp| {
                            x = x.gf128_mul_reduce(&d);
                            y = y ^ pp.gf128_mul_reduce(&x);
                        });
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
                // special/common case where FC^-1 [i] = 0000100000
                // where the 1 is at position gapRows[i][1]. This code is
                // used to speed up this common case.
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
                            debug_assert!(c_col2 <= c_col);
                            let iter = row.get(&c_col2);
                            if iter.is_none() {
                                row.insert(c_col2);
                            } else {
                                let target = *iter.unwrap();
                                row.remove(&target);
                            }
                        }
                    });
                    debug_assert!(row.len() == 0 || *row.last().unwrap() != c_col);
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
        debug_assert!(self.weight_sets.is_some());

        let mut row_set = vec![0u8; self.items_num];
        let weight_sets = self.weight_sets.as_mut().unwrap();
        while weight_sets.weight_sets.len() > 1 {
            // triangulate 各个权重的列表，每个node代表列号
            let col = weight_sets.get_min_weightnode(); // 最小权重的列
            weight_sets.pop_node(col);

            unsafe {
                let col_index: usize = weight_sets.idx_of(&(*col)); // 列号
                (*col).weight = 0;
                let mut first: bool = true;
                self.cols[col_index].iter().for_each(|&row_idx| {
                    if row_set[row_idx] == 0 {
                        //每一行是否处理过
                        row_set[row_idx] = 1; //
                        self.rows.row_data(row_idx, 1).iter().for_each(|&col_idx2| {
                            //每行为1的列号
                            let node: *mut WeightNode =
                                &mut weight_sets.nodes[col_idx2] as *mut WeightNode;

                            if (*node).weight != 0 {
                                weight_sets.pop_node(node as *mut WeightNode); // remove this row
                                (*node).weight -= 1; // 把这个节点弹出后再压入，将
                                weight_sets.push_node(node);
                                // TODO prefetch next column
                            }
                        });
                        if first {
                            main_col.push(col_index);
                            main_row.push(row_idx);
                            first = false;
                        } else {
                            debug_assert_ne!(*main_row.last().unwrap(), row_idx);
                            gap_rows.push([row_idx, *main_row.last().unwrap()]);
                        }
                    }
                });
                debug_assert_eq!(first, false);
            }
        }
    }

    pub fn rebuild_columns(&mut self, _col_weights: &[usize], total_weight: usize) {
        debug_assert_eq!(self.col_backing.len(), total_weight);
        // TODO: Remove when release build
        // {
        //     let mut col_sum: usize = 0;
        //     col_weights.iter().for_each(|x| {
        //         col_sum += x;
        //     });
        //     debug_assert_eq!(col_sum, total_weight);
        // }
        // self cols 存储
        if self.rows.cols() == 3 {
            for i in 0..self.items_num {
                // rows
                let row = self.rows.row_data(i, 1); // row 代表哪几列为1
                row.iter().for_each(|&c| {
                    self.cols[c].push(i); // 第c列的第i行为1
                });
            }
            // copy cols to colbacking
            let mut col_backing_start = 0;
            self.cols.iter().for_each(|x| {
                if x.len() == 0 {
                    return;
                }
                debug_assert!(col_backing_start < self.col_backing.len());
                self.col_backing[col_backing_start..col_backing_start + x.len()].copy_from_slice(x);
                col_backing_start += x.len();
            });
        } else {
            panic!("weight is not 3, TODO");
        }
    }

    pub fn set_mul_inputs(&mut self, rows: &Matrix<usize>, dense: &[Block], col_weights: &[usize]) {
        // debug_assert_eq!(rows.rows(), self.items_num);
        debug_assert_eq!(dense.len(), self.items_num);
        debug_assert_eq!(rows.cols(), self.params.weight);
        // debug_assert_eq!(col_backing.len(), self.items_num * self.params.weight);
        debug_assert_eq!(col_weights.len(), self.params.sparse_size);
        self.rows.storage.copy_from_slice(&rows.storage);
        self.dense.copy_from_slice(&dense);
        self.rebuild_columns(col_weights, self.params.weight * self.items_num);
        debug_assert!(self.weight_sets.is_none());
        self.weight_sets = Some(WeightData::init(col_weights))
    }
}

/// convert row items to sparse vector with length m'
#[derive(Clone, Debug)]
pub struct PaxosHash {
    pointer: *const c_void,
    idbit_length: usize,
}

impl PaxosHash {
    pub fn new(seed: Block, weight: usize, paxos_size: usize) -> PaxosHash {
        unsafe {
            let pointer: *const c_void = new_paxos_hash(64);
            init_paxos_hash(
                pointer,
                &seed as *const Block as *const c_void,
                weight,
                paxos_size,
                64,
            );
            PaxosHash {
                pointer,
                idbit_length: 64usize,
            }
        }
    }

    pub fn build_row(&self, hash: &Block, row: &mut [usize]) {
        unsafe {
            build_row_raw(
                self.pointer,
                hash as *const Block as *const c_void,
                row.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn build_row32(&self, hash: &[Block], row: &mut [usize]) {
        debug_assert_eq!(hash.len(), 32);
        unsafe {
            build_row32_raw(
                self.pointer,
                hash.as_ptr() as *const c_void,
                row.as_mut_ptr() as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn hash_build_row1(&self, input: &[Block], rows: &mut [usize], hash: &mut [Block]) {
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

    pub fn hash_build_row1_pointer(&self, input: &Block, rows: *mut usize, hash: &mut Block) {
        unsafe {
            hash_build_row1_raw(
                self.pointer,
                input as *const Block as *const c_void,
                rows as *mut c_void,
                hash as *mut Block as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn hash_build_row32_pointer(
        &self,
        input: *const Block,
        rows: *mut usize,
        hash: *mut Block,
    ) {
        unsafe {
            hash_build_row32_raw(
                self.pointer,
                input as *const c_void,
                rows as *mut c_void,
                hash as *mut c_void,
                self.idbit_length,
            );
        }
    }

    pub fn hash_build_row32(&self, input: &[Block], rows: &mut [usize], hash: &mut [Block]) {
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
            debug_assert_eq!(I[(i, i)], *ONE_BLOCK);
            for j in 0..n {
                if i != j && I[(i, j)] != *ZERO_BLOCK {
                    debug_assert!(false);
                }
            }
        }
    }

    #[test]
    fn paxos_hash_test() {
        let n: usize = 1 << 32;
        for _i in 0..n {
            let prng = Prng::new(Block::from_i64(0 as i64, 0 as i64));
            let items = prng.get_blocks(1 << 20);

            let hasher = PaxosHash::new(*ZERO_BLOCK, 3, 1 << 10);
            let mut res = vec![0usize; 32];
            hasher.build_row32(&items, &mut res);
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
        let h = PaxosHash::new(Block::from_i64(0, s), 3, n);
        let prng = Prng::new(Block::from_i64(1, s));
        let exp: [[usize; 3]; 32] = [
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
        let mut rows: Matrix<usize> = Matrix::<usize>::new(32, 3, 0);
        for tt in 0..t {
            let hash = prng.get_blocks(32);
            h.build_row32(&hash, rows.mut_data());

            for i in 0..32 {
                let mut rr: [usize; 3] = [0; 3];
                h.build_row(&hash[i], &mut rr);
                println!("{:?}", rr);
                for j in 0..3 {
                    debug_assert_eq!(rows[(i, j)], rr[j]);

                    if tt == 0 {
                        debug_assert_eq!(rows[(i, j)], exp[i][j]);
                    }
                }
            }
        }
    }

    #[test]
    fn paxos_encode_test() {
        let n: usize = (1 << 20) + 3;
        let w = 3usize;
        let s = 0usize;
        let t = 1usize;
        for tt in 0..t {
            let mut paxos = Paxos::new(n, w, 40, *ZERO_BLOCK);
            let mut paxos2 = Paxos::new(n, w, 40, *ZERO_BLOCK);
            let prng = Prng::new(Block::from_i64(5 as i64, 12 as i64));
            let items = prng.get_blocks(n);
            let values = prng.get_blocks(n);
            let mut values2 = vec![Block::from_i64(0, 0); n];
            let mut p = vec![Block::from_i64(0, 0); paxos.size()];
            paxos.set_input(&items);
            paxos2.set_input(&items);
            for i in 0..paxos.rows.rows() {
                for j in 0..w {
                    debug_assert_eq!(paxos.rows[(i, j)], paxos2.rows[(i, j)]);
                }
            }
            paxos.encode(&values, &mut p, None);
            paxos.decode(&items, &mut values2, &p);
            values
                .iter()
                .zip(values2.iter())
                .enumerate()
                .for_each(|(i, (x1, x2))| {
                    debug_assert_eq!(x1, x2);
                });
        }
    }

    #[test]
    fn paxos_test_step_test() {
        let a = vec![0, 1, 2, 3, 4];
        a.iter().step_by(3).for_each(|&x| {
            println!("{}", x);
        })
    }
}
