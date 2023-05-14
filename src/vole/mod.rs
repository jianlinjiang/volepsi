pub mod aes;
pub mod baxos;
pub mod block;
pub mod matrix;
pub mod paxos;
pub mod prng;
pub mod silent_vole_receiver;
pub mod silent_vole_sender;
pub mod utils;
pub mod weight_data;
pub enum VoleRole {
    Sender = 1,
    Receiver = 2,
}
