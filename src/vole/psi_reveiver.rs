use super::aes::{fixed_key_hash_blocks, set_block_mask};
use super::baxos::Baxos;
use super::block::{Block, ZERO_BLOCK};
use super::prng::Prng;
use super::psi_sender::RequestType;
use super::silent_vole_receiver::SilentVoleReceiver;
use bytes::Bytes;
use libc::memcpy;
use log::error;
use network::{ReliableSender, SimpleSender};
use rand::{thread_rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::net::{IpAddr, SocketAddr};
use std::thread;
use log::info;
#[derive(Debug)]
pub struct Receiver {
    // origin_input: Vec<usize>,
    sender_size: usize,
    receiver_size: usize,
    a: Vec<Block>,
    c: Vec<Block>,
    bin_size: usize,
    ssp: usize,
    prng: Prng,
    pub res: Vec<usize>,
    sender_port: u16,
    sender_ip: IpAddr,
}

impl Receiver {
    pub fn new(seed: Block, sender_ip: IpAddr, sender_port: u16) -> Receiver {
        Receiver {
            sender_size: 0,
            receiver_size: 0,
            a: vec![],
            c: vec![],
            bin_size: 1 << 14,
            ssp: 40,
            prng: Prng::new(seed),
            res: vec![],
            sender_port: sender_port,
            sender_ip: sender_ip,
        }
    }

    pub async fn receive(&mut self, inputs: &[Block]) {
        self.sender_size = inputs.len();
        self.receiver_size = inputs.len();

        let mask_size = ((self.ssp as f64
            + ((self.sender_size * self.receiver_size) as f64)
                .log2()
                .ceil())
            / 8f64)
            .ceil() as usize;
        let compress = mask_size != 16;
        debug_assert_eq!(compress, true);
        info!("mask size {}", mask_size);
        let seed: Block = self.prng.get_block();

        let mut baxos = Baxos::new(inputs.len(), self.bin_size, 3, self.ssp, seed);

        // network send seed
        let mut network = SimpleSender::new();
        network
            .send(
                SocketAddr::new(self.sender_ip, self.sender_port),
                Bytes::from(
                    bincode::serialize(&RequestType::BaxosSeed([
                        seed.get(0) as i64,
                        seed.get(1) as i64,
                    ]))
                    .unwrap(),
                ),
            )
            .await;

        let baxos_size = baxos.size();

        let vole_address = SocketAddr::new(self.sender_ip, self.sender_port + 1);
        let handler = thread::spawn(move || {
            let mut rng = thread_rng();
            // thread code
            let vole_receiver: SilentVoleReceiver = SilentVoleReceiver::new(vole_address, &mut rng);
            vole_receiver.init(baxos_size);
            vole_receiver.get_a_c(baxos_size)
        });

        let mut self_hash = vec![*ZERO_BLOCK; inputs.len()];
        // inputs values = hash(inputs, fixed_key)
        unsafe {
            fixed_key_hash_blocks(
                inputs.as_ptr() as *const c_void,
                inputs.len(),
                self_hash.as_mut_ptr() as *mut c_void,
            );
        }

        let mut p = vec![*ZERO_BLOCK; baxos_size];
        baxos.solve(inputs, &self_hash, &mut p, 1);

        (self.a, self.c) = handler.join().unwrap();
        info!("vole finish");
        let pp: Vec<[i64; 2]> = cfg_iter_mut!(p)
            .zip(&self.c)
            .map(|(pp, cc)| {
                *pp = *pp ^ *cc;
                [pp.get(0) as i64, pp.get(1) as i64]
            })
            .collect();
        let mut network = ReliableSender::new();
        // send pp
        let ret = network
            .send(
                SocketAddr::new(self.sender_ip, self.sender_port),
                Bytes::from(bincode::serialize(&RequestType::BaxosPp(pp)).unwrap()),
            )
            .await;
        match ret.await {
            Ok(_) => {},
            Err(e) => {
                error!("error when send encoded p {:?}", e);
            }
        }
        let mut outputs = vec![*ZERO_BLOCK; inputs.len()];
        baxos.decode(inputs, &mut outputs, &self.a, 1);
        

        let mut outputs_res = vec![*ZERO_BLOCK; inputs.len()];
        unsafe {
            fixed_key_hash_blocks(
                outputs.as_ptr() as *const c_void,
                outputs.len(),
                outputs_res.as_mut_ptr() as *mut c_void,
            );
        }

        let mut mask = *ZERO_BLOCK;
        unsafe {
            set_block_mask(&mut mask as *mut Block as *mut c_void, mask_size);
        }
        let mut map: HashMap<Block, usize> = HashMap::with_capacity(outputs_res.len());
        outputs_res.iter().enumerate().for_each(|(i, h)| {
            if map.insert(*h & mask, i).is_some() {
                panic!("conflicts");
            }
        });

        // receive peer hashes
        let ret = network
            .send(
                SocketAddr::new(self.sender_ip, self.sender_port),
                Bytes::from(bincode::serialize(&RequestType::RequestHash).unwrap()),
            )
            .await;

        match ret.await {
            Ok(data) => {
                info!("receive res");
                let peer_hashes: Vec<u8> = bincode::deserialize(&data).unwrap();
                let mut peer_hashes_ptr = peer_hashes.as_ptr();
                let mut h = *ZERO_BLOCK;
                for _i in 0..self.sender_size {
                    unsafe {
                        memcpy(
                            &mut h as *mut Block as *mut c_void,
                            peer_hashes_ptr as *const c_void,
                            mask_size,
                        );
                        peer_hashes_ptr = peer_hashes_ptr.add(mask_size);
                        if let Some(id) = map.get(&h) {
                            self.res.push(*id);
                        }
                    }
                }
            }
            Err(e) => {
                error!("error when receive result from sender {:?}", e);
            }
        }
        // write data to csv
    }
}
