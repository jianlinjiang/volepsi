pub fn div_ceil(val: usize, d: usize) -> usize {
    (val + d - 1) / d
}

pub fn round_up_to(val: usize, step: usize) -> usize {
    div_ceil(val, step) * step
}
