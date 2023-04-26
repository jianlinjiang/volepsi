pub mod block;
pub mod matrix;
pub mod paxos;
pub mod silent_vole_receiver;
pub mod silent_vole_sender;

pub enum VoleRole {
    Sender = 1,
    Receiver = 2
}