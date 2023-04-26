use std::os::raw::c_char;
use std::net::SocketAddr;
use rand::Rng;
// use ::VoleRole;
use super::VoleRole;
use std::ffi::CString;
use log::debug;
#[link(name = "rvole")]
extern "C" {
    pub fn init_silent_vole(role: i32, ip: *const c_char, seed: *const u64, vector_size: u64);
    pub fn silent_send_inplace(delta: *const u64, vector_size: u64);
}



pub struct SlientVoleSender {
    seed: [u64; 2],
    address: SocketAddr
}

impl SlientVoleSender {
    pub fn new<R: Rng>(address: SocketAddr, rng: &mut R) -> Self {
        Self {
            seed: [rng.next_u64(), rng.next_u64()],
            address
        }
    }

    pub fn init(&self) {
        let ip_str = CString::new(self.address.to_string()).unwrap();
        debug!("sender ip {:?}", ip_str);
        unsafe {
            init_silent_vole(VoleRole::Sender as i32, ip_str.as_ptr(), self.seed.as_ptr(), 2<<20);
        }
    }
}