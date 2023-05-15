use super::block::{Block, ZERO_BLOCK};
use std::ffi::c_void;

#[link(name = "rcrypto")]
extern "C" {
    pub fn new_prng(seed: *const c_void) -> *const c_void;
    pub fn delete_prng(pointer: *const c_void);
    pub fn get_blocks_raw(pointer: *const c_void, blocks: *mut c_void, block_num: usize);
}

#[derive(Clone, Debug)]
pub struct Prng {
    pointer: *const c_void,
}

impl Prng {
    pub fn new(seed: Block) -> Prng {
        unsafe {
            Prng {
                pointer: new_prng(&seed as *const Block as *const c_void),
            }
        }
    }

    pub fn get_blocks(&self, block_num: usize) -> Vec<Block> {
        let mut blocks = vec![Block::from_i64(0, 0); block_num];
        unsafe {
            get_blocks_raw(self.pointer, blocks.as_mut_ptr() as *mut c_void, block_num);
        }
        blocks
    }

    pub fn get_block(&self) -> Block {
        let mut block = *ZERO_BLOCK;
        unsafe {
            get_blocks_raw(self.pointer, &mut block as *mut Block as *mut c_void, 1);
        }
        block
    }
}

impl Drop for Prng {
    fn drop(&mut self) {
        unsafe {
            delete_prng(self.pointer);
        }
    }
}
