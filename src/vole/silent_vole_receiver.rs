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
    pub fn silent_receive_inplace(vector_size: u64);
    pub fn get_receiver_c(c: *mut c_void, n: usize);
    pub fn get_receiver_a(a: *mut c_void, n: usize);
}

pub struct SilentVoleReceiver {
    seed: [u64; 2],
    sender_address: SocketAddr,
}

impl SilentVoleReceiver {
    pub fn new<R: Rng>(sender_address: SocketAddr, rng: &mut R) -> Self {
        Self {
            seed: [rng.next_u64(), rng.next_u64()],
            sender_address,
        }
    }

    pub fn init(&self, n: usize) {
        let ip_str = CString::new(self.sender_address.to_string()).unwrap();
        debug!("sender ip {:?}", ip_str);
        unsafe {
            init_silent_vole(
                VoleRole::Receiver as i32,
                ip_str.as_ptr(),
                self.seed.as_ptr(),
                n as u64,
            );
            silent_receive_inplace(n as u64);
        }
    }

    pub fn get_a_c(&self, n: usize) -> (Vec<Block>, Vec<Block>) {
        let mut a = vec![*ZERO_BLOCK; n];
        let mut c = vec![*ZERO_BLOCK; n];
        unsafe {
            get_receiver_a(a.as_mut_ptr() as *mut c_void, n);
            get_receiver_c(c.as_mut_ptr() as *mut c_void, n);
        }
        (a, c)
    }
}
