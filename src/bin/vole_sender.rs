use env_logger::Env;
use log::info;
use rand::{thread_rng, RngCore};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Instant;
use volepsi::vole::silent_vole_sender::SilentVoleSender;
#[tokio::main]
async fn main() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    let mut rng = thread_rng();
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000);
    let start = Instant::now();
    let sender = SilentVoleSender::new(address, &mut rng);
    let delta = [rng.next_u64(), rng.next_u64()];
    sender.init(delta);
    let duration = start.elapsed();
    info!("Time elapsed is: {:?}", duration);
}
