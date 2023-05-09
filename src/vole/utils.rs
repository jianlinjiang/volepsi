pub fn div_ceil(val: u64, d: u64) -> u64 {
    (val + d - 1) / d
}

pub fn round_up_to(val: u64, step: u64) -> u64 {
    div_ceil(val, step) * step
}
