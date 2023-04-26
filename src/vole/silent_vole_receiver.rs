use std::os::raw::c_char;

#[link(name = "rvole")]
extern "C" {
    pub fn init_silent_vole(role: i32, ip: *const c_char, seed: *const u64, vector_size: u64);
    pub fn silent_receive_inplace(vector_size: u64);
}