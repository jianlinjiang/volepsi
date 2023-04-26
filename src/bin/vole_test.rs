use env_logger::Env;
use rand::thread_rng;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use volepsi::vole::silent_vole_sender::SlientVoleSender;
#[tokio::main]
async fn main() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    let mut rng = thread_rng();
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000);
    let sender = SlientVoleSender::new(address, &mut rng);
    sender.init();
}
