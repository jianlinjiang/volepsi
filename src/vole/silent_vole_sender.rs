use super::block::{Block, ZERO_BLOCK};
use super::VoleRole;
use log::debug;
use rand::Rng;
use std::ffi::c_void;
use std::ffi::CString;
use std::net::SocketAddr;
use std::os::raw::c_char;
#[link(name = "rvole")]
extern "C" {
    pub fn init_silent_vole(role: i32, ip: *const c_char, seed: *const u64, vector_size: u64);
    pub fn silent_send_inplace(delta: *const u64, vector_size: u64);
    pub fn get_sender_b(b: *mut c_void, n: usize);
}

#[derive(Debug, Clone)]
pub struct SilentVoleSender {
    seed: [u64; 2],
    address: SocketAddr,
}

impl SilentVoleSender {
    pub fn new<R: Rng>(address: SocketAddr, rng: &mut R) -> Self {
        Self {
            seed: [rng.next_u64(), rng.next_u64()],
            address,
        }
    }

    pub fn init(&self, delta: [u64; 2], n: usize) {
        let ip_str = CString::new(self.address.to_string()).unwrap();
        debug!("sender ip {:?}", ip_str);
        unsafe {
            init_silent_vole(
                VoleRole::Sender as i32,
                ip_str.as_ptr(),
                self.seed.as_ptr(),
                n as u64,
            );
            silent_send_inplace(delta.as_ptr(), n as u64);
        }
    }

    pub fn get_b(&self, n: usize) -> Vec<Block> {
        let mut res = vec![*ZERO_BLOCK; n];
        unsafe {
            get_sender_b(res.as_mut_ptr() as *mut c_void, n);
        }
        res
    }
}
