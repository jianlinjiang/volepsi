use super::block::Block;
use std::ffi::c_void;

#[link(name = "rcrypto")]
extern "C" {
    pub fn new_aes() -> *const c_void;
    pub fn set_key(aes: *const c_void, key: *const c_void);
    pub fn hash_blocks(
        aes: *const c_void,
        plain_text: *const c_void,
        block_length: usize,
        cipher_text: *mut c_void,
    );
    pub fn delete_aes(pointer: *const c_void);
}

#[derive(Clone, Debug)]
pub struct Aes {
    aes_pointer: *const c_void,
    key: Block,
}

impl Drop for Aes {
    fn drop(&mut self) {
        unsafe {
            delete_aes(self.aes_pointer);
        }
    }
}

impl Aes {
    pub fn new() -> Aes {
        unsafe {
            Aes {
                aes_pointer: new_aes(),
                key: Block::from_i64(-1, -1),
            }
        }
    }

    pub fn set_key(&mut self, user_key: Block) {
        self.key = user_key;
        unsafe {
            set_key(self.aes_pointer, &user_key as *const Block as *const c_void);
        }
    }

    /// H(x) = AES(x) + x.
    pub fn hash_block(&self, plain_text: &Block) -> Block {
        assert_ne!(self.key, Block::from_i64(-1, -1));
        let mut cipher = Block::from_i64(0, 0);
        unsafe {
            hash_blocks(
                self.aes_pointer,
                plain_text as *const Block as *const c_void,
                1,
                &mut cipher as *mut Block as *mut c_void,
            );
        }
        cipher
    }

    pub fn hash_blocks(
        &self,
        plain_texts: &[Block],
        block_length: usize,
        ciphertexts: &mut [Block],
    ) {
        assert_ne!(self.key, Block::from_i64(-1, -1));
        unsafe {
            hash_blocks(
                self.aes_pointer,
                plain_texts.as_ptr() as *const c_void,
                block_length,
                ciphertexts.as_mut_ptr() as *mut c_void,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(arithmetic_overflow)]
    use super::*;
    use rand::{thread_rng, RngCore};
    #[test]
    fn aes_setkey() {
        let mut rng = thread_rng();
        let mut aes = Aes::new();
        let key = Block::from_i64(0, 0);
        let plain = Block::from_i64(0, 0);
        aes.set_key(key);
        let cipher = aes.hash_block(&plain);
        let result = Block::from_i64(3326810793440857224, 4263935709876578662);
        assert_eq!(result, cipher);
        let count = 100;
        let mut plaintext = Vec::new();
        for _i in 0..count {
            plaintext.push(Block::rand(&mut rng));
        }
        let mut ciphertexts = Vec::new();
        ciphertexts.resize(count, Block::from_i64(0, 0));
        aes.hash_blocks(&plaintext, count, &mut ciphertexts);
        for i in 0..count {
            let cipher = aes.hash_block(&plaintext[i]);
            assert_eq!(cipher, ciphertexts[i]);
        }
    }
}
