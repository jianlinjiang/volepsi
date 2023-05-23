use super::block::{gf128_inv, Block, ONE_BLOCK, ZERO_BLOCK};
use super::matrix::Matrix;
use super::paxos::{Paxos, PaxosParam};
use crate::vole::aes::Aes;
use libc::memmove;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::ffi::c_void;
use std::slice::from_raw_parts_mut;

#[link(name = "rcrypto")]
extern "C" {
    pub fn get_bin_size(bins: usize, balls: usize, ssp: usize) -> usize;
    pub fn libdivide_u64_gen(bins: usize);
    pub fn do_mod32(vals: *mut usize, mod_val: usize);
    pub fn bin_idx_compress(block: *const c_void) -> usize;
}

#[derive(Clone, Debug)]
pub struct Baxos {
    pub items: usize,
    pub bins: usize,
    pub items_per_bin: usize,
    pub weight: usize,
    pub ssp: usize,
    pub seed: Block,
    pub params: PaxosParam,
}

impl Baxos {
    pub fn size(&self) -> usize {
        self.bins * (self.params.sparse_size + self.params.dense_size)
    }

    pub fn new(items: usize, bin_size: usize, weight: usize, ssp: usize, seed: Block) -> Baxos {
        let bins: usize = (items + bin_size - 1) / bin_size;
        let items_per_bin =
            unsafe { get_bin_size(bins, items, ssp + (bins as f64).log2().floor() as usize) };
        Baxos {
            items,
            bins: bins,
            items_per_bin: unsafe {
                get_bin_size(bins, items, ssp + (bins as f64).log2().floor() as usize)
            },
            weight,
            ssp,
            seed,
            params: PaxosParam::init(items_per_bin, weight, ssp),
        }
    }

    pub fn solve<'a>(
        &mut self,
        inputs: &[Block],
        values: &[Block],
        output: &mut [Block],
        threads: usize,
    ) {
        assert_eq!(output.len(), self.size());
        const BATCH_SIZE: usize = 32usize;
        let total_bins = self.bins * threads;
        let items_per_thread = (self.items + threads - 1) / threads;

        // let per_thread_max_bins = self.items_per_bin + 10;
        let per_thread_max_bins = unsafe { get_bin_size(self.bins, items_per_thread, self.ssp) };
        let combined_max_bins = per_thread_max_bins * threads;

        let mut thread_bin_sizes = Matrix::new(threads, self.bins, 0usize);
        println!("{} {}", self.items_per_bin, per_thread_max_bins);
        let mut input_mapping = vec![0usize; total_bins * per_thread_max_bins];

        let mut set_input_mapping = |thread_idx: usize, bin_idx: usize, bs: usize, value: usize| {
            let bin_begin = combined_max_bins * bin_idx;
            let thread_begin = per_thread_max_bins * thread_idx;
            input_mapping[bin_begin + thread_begin + bs] = value;
        };

        let mut val_backing = vec![*ZERO_BLOCK; total_bins * per_thread_max_bins];
        let mut set_values = |thread_idx: usize, bin_idx: usize, bs: usize, value: Block| {
            let bin_begin = combined_max_bins * bin_idx;
            let thread_begin = per_thread_max_bins * thread_idx;
            val_backing[bin_begin + thread_begin + bs] = value;
        };

        let mut hash_backing = vec![*ZERO_BLOCK; total_bins * per_thread_max_bins];

        let mut set_hashes = |thread_idx: usize, bin_idx: usize, bs: usize, value: Block| {
            let bin_begin: usize = combined_max_bins * bin_idx;
            let thread_begin = per_thread_max_bins * thread_idx;
            hash_backing[bin_begin + thread_begin + bs] = value;
        };
        unsafe {
            libdivide_u64_gen(self.bins);
        }
        let threads_id: Vec<usize> = (0..threads).collect();
        threads_id.iter().for_each(|&thread_idx| {
            let mut hasher = Aes::new();
            hasher.set_key(self.seed);
            let begin = inputs.len() * thread_idx / threads;
            let end = inputs.len() * (thread_idx + 1) / threads;
            let inputs = &inputs[begin..end];
            {
                let bin_sizes = thread_bin_sizes.mut_row_data(thread_idx, 1);

                let mut hashes = [*ZERO_BLOCK; BATCH_SIZE];

                let main = inputs.len() / BATCH_SIZE * BATCH_SIZE;

                let mut bin_idxs = [0; BATCH_SIZE];

                let mut i = 0;
                let mut in_idx = begin;
                while i < main {
                    hasher.hash_blocks(&inputs[i..i + BATCH_SIZE], BATCH_SIZE, &mut hashes);
                    assert_eq!(bin_idxs.len(), hashes.len());
                    bin_idxs.iter_mut().zip(hashes).for_each(|(bin_id, hash)| {
                        *bin_id = self.bin_idx_compress(&hash);
                    });
                    unsafe {
                        do_mod32(&mut bin_idxs as *mut usize, self.bins);
                    }

                    for k in 0..BATCH_SIZE {
                        let bin_idx = bin_idxs[k];

                        let bs = bin_sizes[bin_idx];
                        bin_sizes[bin_idx] += 1;

                        set_input_mapping(thread_idx, bin_idx, bs, in_idx);
                        set_values(thread_idx, bin_idx, bs, values[in_idx]);
                        set_hashes(thread_idx, bin_idx, bs, hashes[k]);

                        in_idx += 1;
                    }
                    i += BATCH_SIZE;
                }
                let mut hash;
                while i < inputs.len() {
                    hash = hasher.hash_block(&inputs[i]);
                    let bin_idx = self.mod_num_bins(&hash);
                    let bs = bin_sizes[bin_idx];
                    bin_sizes[bin_idx] += 1;
                    assert!(bs < per_thread_max_bins);

                    set_input_mapping(thread_idx, bin_idx, bs, in_idx);
                    set_values(thread_idx, bin_idx, bs, values[in_idx]);
                    set_hashes(thread_idx, bin_idx, bs, hash);
                    in_idx += 1;
                    i += 1;
                }
            }
        });

        threads_id.iter().for_each(|&thread_idx| {
            let paxos_size_per = self.params.size();
            let mut bin_idx = thread_idx;
            while bin_idx < self.bins {
                let mut bin_size = 0usize;
                for i in 0..threads {
                    bin_size += thread_bin_sizes[(i, bin_idx)];
                }
                assert!(bin_size <= self.items_per_bin);

                let mut paxos: Paxos = Paxos::new_with_params(bin_size, &self.params, self.seed);

                let mut rows = Matrix::new(bin_size, self.weight, 0usize);
                let mut col_weights = vec![0usize; self.params.sparse_size];

                let hash_backing_start = &hash_backing[0] as *const Block;
                let val_backing_start = &val_backing[0] as *const Block;

                let bin_begin = combined_max_bins * bin_idx;

                let hashes = &mut hash_backing[bin_begin..bin_begin + bin_size];
                let values = &mut val_backing[bin_begin..bin_begin + bin_size];

                let output: &mut [Block] = &mut output
                    [paxos_size_per * bin_idx..paxos_size_per * bin_idx + paxos_size_per];

                let mut bin_pos = thread_bin_sizes[(0, bin_idx)];
                assert!(bin_pos <= per_thread_max_bins);

                for i in 1..threads {
                    let size = thread_bin_sizes[(i, bin_idx)];
                    assert!(size <= per_thread_max_bins);
                    let local_bin_begin = combined_max_bins * bin_idx;
                    let local_thread_begin: usize = per_thread_max_bins * i;

                    unsafe {
                        let thread_hashes = hash_backing_start
                            .offset((local_bin_begin + local_thread_begin) as isize);
                        memmove(
                            hashes.as_mut_ptr() as *mut Block as *mut c_void,
                            thread_hashes as *const c_void,
                            16 * size,
                        );
                    }
                    let thread_vals = unsafe {
                        val_backing_start.offset((local_bin_begin + local_thread_begin) as isize)
                    };

                    for j in 0..size {
                        values[bin_pos + j] = unsafe { *thread_vals.add(j) }
                    }
                    bin_pos += size;
                }
                col_weights.iter_mut().for_each(|x| {
                    *x = 0;
                });

                if self.weight == 3 {
                    let main = bin_size / BATCH_SIZE * BATCH_SIZE;
                    let mut i = 0usize;

                    while i < main {
                        let riter = rows.mut_row_data(i, 32);
                        paxos.hasher.build_row32(&hashes[i..i + BATCH_SIZE], riter);
                        for j in 0..BATCH_SIZE {
                            col_weights[riter[j * self.weight + 0]] += 1;
                            col_weights[riter[j * self.weight + 1]] += 1;
                            col_weights[riter[j * self.weight + 2]] += 1;
                        }
                        i += BATCH_SIZE;
                    }
                    while i < bin_size {
                        let riter = rows.mut_row_data(i, 1);
                        paxos.hasher.build_row(&hashes[i], riter);
                        col_weights[riter[0]] += 1;
                        col_weights[riter[1]] += 1;
                        col_weights[riter[2]] += 1;
                        i += 1;
                    }
                } else {
                    panic!("weight is not 3");
                }
                paxos.set_mul_inputs(&rows, hashes, &col_weights);
                paxos.encode(values, output, None);
                bin_idx += threads;
            }
        });
    }

    pub fn decode(&self, inputs: &[Block], values: &mut [Block], pp: &[Block], threads: usize) {
        let thread_ids: Vec<usize> = (0..threads).collect();
        thread_ids.iter().for_each(|&i| {
            let begin = (inputs.len() * i) / threads;
            let end = (inputs.len() * (i + 1)) / threads;
            let inputs = &inputs[begin..end];
            let values = &mut values[begin..end];
            {
                // decode batch
                let decode_size = std::cmp::min(512, inputs.len());
                let mut batches = Matrix::new(self.bins, decode_size, *ZERO_BLOCK);
                let mut in_idxs = Matrix::new(self.bins, decode_size, 0usize);

                let mut batch_sizes = vec![0usize; self.bins];

                let mut hasher = Aes::new();
                hasher.set_key(self.seed);

                let size_per = self.size() / self.bins;
                let mut paxos = Paxos::new_with_params(1, &self.params, self.seed);

                let mut buff = [*ZERO_BLOCK; 32];

                const BATCH_SIZE: usize = 32;
                let main = inputs.len() / BATCH_SIZE * BATCH_SIZE;
                let mut buffer: [Block; 32] = [*ZERO_BLOCK; BATCH_SIZE];
                let mut bin_idxs = [0usize; BATCH_SIZE];

                unsafe {
                    libdivide_u64_gen(self.bins);
                }

                let mut i = 0usize;
                while i < main {
                    hasher.hash_blocks(&inputs[i..i + BATCH_SIZE], 32, &mut buffer);
                    bin_idxs
                        .iter_mut()
                        .zip(buffer.iter())
                        .for_each(|(bin_id, b)| {
                            *bin_id = self.bin_idx_compress(b);
                        });
                    unsafe {
                        do_mod32(&mut bin_idxs as *mut usize, self.bins);
                    }

                    bin_idxs.iter().enumerate().for_each(|(k, &bin_idx)| {
                        batches[(bin_idx, batch_sizes[bin_idx])] = buffer[k];
                        in_idxs[(bin_idx, batch_sizes[bin_idx])] = i + k;
                        batch_sizes[bin_idx] += 1;
                        if batch_sizes[bin_idx] == decode_size {
                            let p = &pp[bin_idx * size_per..(bin_idx + 1) * size_per];
                            let idx = in_idxs.row_data(bin_idx, 1);
                            self.impl_decode_bin(
                                batches.row_data(bin_idx, 1),
                                values,
                                &mut buff,
                                idx,
                                p,
                                &mut paxos,
                            );
                            batch_sizes[bin_idx] = 0;
                        }
                    });
                    i += BATCH_SIZE;
                }

                while i < inputs.len() {
                    let k = 0usize;
                    buffer[k] = hasher.hash_block(&inputs[i]);
                    let bin_idx = self.mod_num_bins(&buffer[k]);
                    batches[(bin_idx, batch_sizes[bin_idx])] = buffer[k];
                    in_idxs[(bin_idx, batch_sizes[bin_idx])] = i + k;
                    batch_sizes[bin_idx] += 1;
                    if batch_sizes[bin_idx] == decode_size {
                        let p = &pp[bin_idx * size_per..bin_idx * size_per + size_per];
                        self.impl_decode_bin(
                            batches.row_data(bin_idx, 1),
                            values,
                            &mut buff,
                            in_idxs.row_data(bin_idx, 1),
                            p,
                            &mut paxos,
                        );
                        batch_sizes[bin_idx] = 0;
                    }
                    i += 1;
                }

                for bin_idx in 0..self.bins {
                    if batch_sizes[bin_idx] != 0 {
                        let p = &pp[bin_idx * size_per..bin_idx * size_per + size_per];
                        let b = &batches.row_data(bin_idx, 1)[0..batch_sizes[bin_idx]];

                        self.impl_decode_bin(
                            b,
                            values,
                            &mut buff,
                            in_idxs.row_data(bin_idx, 1),
                            p,
                            &mut paxos,
                        );
                    }
                }
            }
        });
    }

    pub fn impl_decode_bin(
        &self,
        // bin_idx: usize,
        hashes: &[Block],
        values: &mut [Block],
        values_buff: &mut [Block],
        in_idx: &[usize],
        pp: &[Block],
        paxos: &mut Paxos,
    ) {
        const BATCH_SIZE: usize = 32;
        const MAX_WEIGHT_SIZE: usize = 20;

        let main = hashes.len() / BATCH_SIZE * BATCH_SIZE;
        assert!(self.weight <= MAX_WEIGHT_SIZE);

        let mut row = Matrix::new(BATCH_SIZE, self.weight, 0usize);

        assert!(values_buff.len() >= BATCH_SIZE);

        let mut i = 0;
        while i < main {
            paxos
                .hasher
                .build_row32(&hashes[i..i + BATCH_SIZE], &mut row.storage);
            paxos.decode32(&row.storage, &hashes[i..i + BATCH_SIZE], values_buff, pp);

            for k in 0..BATCH_SIZE {
                // values[in_idx[i + k]] = values[in_idx[i + k]] ^ values_buff[k];
                values[in_idx[i + k]] = values_buff[k];
            }
            i += BATCH_SIZE;
        }
        while i < hashes.len() {
            paxos.hasher.build_row(&hashes[i], &mut row.storage);
            let v = &mut values[in_idx[i]];
            paxos.decode1(row.row_data(0, 1), &hashes[i], v, pp);
            i += 1;
        }
    }

    pub fn bin_idx_compress(&self, h: &Block) -> usize {
        unsafe { bin_idx_compress(h as *const Block as *const c_void) }
    }

    pub fn mod_num_bins(&self, h: &Block) -> usize {
        self.bin_idx_compress(h) % self.bins
    }
}

#[cfg(test)]
mod tests {
    #![allow(arithmetic_overflow)]
    use core::panic;

    use super::super::block::ZERO_BLOCK;
    use super::super::prng::Prng;
    use super::*;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    #[test]
    fn baxos_test() {
        let n: usize = 1 << 20;
        // let b = n / (1 << 14);
        let b = (1 << 14);
        let w: usize = 3;
        let s = 0usize;
        let t = 1usize;
        let nt: usize = 8usize;
        {
            for tt in 0..t {
                let prng = Prng::new(Block::from_i64(0 as i64, s as i64));
                let mut baxos = Baxos::new(n, b, w, 40, prng.get_blocks(1)[0]);
                println!("{:?}", baxos);
                let mut values2 = vec![*ZERO_BLOCK; n];
                let mut p = vec![*ZERO_BLOCK; baxos.size()];
                let items = prng.get_blocks(n);
                let values = prng.get_blocks(n);

                baxos.solve(&items, &values, &mut p, 1);
                let file = File::options()
                    .create(true)
                    .write(true)
                    .open("result_tmp.txt")
                    .unwrap();
                let mut file = BufWriter::new(file);
                // for pp in &p {
                //     file.write(format!("{:?}\n", pp).as_bytes()).unwrap();
                // }

                baxos.decode(&items, &mut values2, &p, 1);
                for pp in &values2 {
                    file.write(format!("{:?}\n", pp).as_bytes()).unwrap();
                }
                // println!("===============");
                // for i in values2.len() - 100..values2.len() {
                //     println!("{:?}", values2[i]);
                // }
                // for i in 0..100 {
                //     println!("{:?}", values2[i]);
                // }
                assert_eq!(values2.len(), values.len());
                // assert_eq!(values2, values);
                let mut i = 0;
                let mut count = 0;
                values2.iter().zip(values.iter()).for_each(|(v1, v2)| {
                    if *v1 != *v2 {
                        println!("{}", i);
                        assert_eq!(*v1, *v2);
                        count += 1;
                    }
                    i += 1;
                });
                println!("{}", count);
            }
        }
    }

    #[test]
    fn baxos_para_test() {
        let n: usize = 1 << 20;
        let b = n / (1 << 14);
        let w: usize = 3;
        let s = 0usize;
        let t = 1usize;
        let nt = 1usize;
        {
            for tt in 0..t {
                let prng = Prng::new(Block::from_i64(0 as i64, s as i64));
                let mut baxos = Baxos::new(n, b, w, 40, *ZERO_BLOCK);
                let mut values2 = vec![*ZERO_BLOCK; n];
                let mut p = vec![*ZERO_BLOCK; baxos.size()];
                let items = prng.get_blocks(n);
                let values = prng.get_blocks(n);
                // baxos.solve(&items, &values, &mut p, 1);
                baxos.solve(&items, &values, &mut p, nt);
                println!("{} {}", n, baxos.size());
                baxos.decode(&items, &mut values2, &p, nt);
                // for pp in p {
                //     println!("{:?}", pp);
                // }
                assert_eq!(values2.len(), values.len());
                assert_eq!(values2, values);
                values2.iter().zip(values.iter()).for_each(|(v1, v2)| {
                    assert_eq!(*v1, *v2);
                });
            }
        }
    }
}
