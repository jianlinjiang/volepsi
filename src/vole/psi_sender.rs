use super::aes::fixed_key_hash_blocks;
use super::baxos::Baxos;
use super::block::{Block, ZERO_BLOCK};
use super::prng::Prng;
use super::silent_vole_sender::SilentVoleSender;
use async_trait::async_trait;
use bytes::Bytes;
use crossbeam_channel::{bounded, Receiver as CrossReceiver, Sender as CrossSender};
use futures::SinkExt;
use libc::memcpy;
use log::error;
use network::{MessageHandler, Receiver as NetworkReceiver, Writer};
use rand::{thread_rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::ffi::c_void;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::thread;
use tokio::sync::mpsc::{channel, Receiver as MpscReceiver, Sender as MpscSender};
use log::info;
#[derive(Clone, Serialize, Deserialize)]
pub enum RequestType {
    BaxosSeed([i64; 2]),
    BaxosPp(Vec<[i64; 2]>),
    RequestHash,
}

#[derive(Clone)]
pub struct SenderHandler {
    channel_sender: MpscSender<Vec<Block>>,
    cross_receiver: CrossReceiver<Vec<u8>>,
}

#[async_trait]
impl MessageHandler for SenderHandler {
    async fn dispatch(&self, writer: &mut Writer, message: Bytes) -> Result<(), Box<dyn Error>> {
        let request: RequestType = bincode::deserialize(&message).unwrap();
        match request {
            RequestType::BaxosSeed(block) => {
                self.channel_sender
                    .send(vec![Block::from_i64(block[1], block[0])])
                    .await?;
            }
            RequestType::BaxosPp(pps) => {
                self.channel_sender
                    .send(
                        pps.iter()
                            .map(|[b0, b1]| Block::from_i64(*b1, *b0))
                            .collect(),
                    )
                    .await?;
                match writer.send("ACK".into()).await {
                    Ok(_) => {}
                    Err(e) => {
                        error!("{:?}", e)
                    }
                }
            }
            RequestType::RequestHash => {
                self.channel_sender.send(vec![]).await?;
                let hash = self.cross_receiver.recv().unwrap();
                let data = bincode::serialize(&hash).unwrap();
                match writer.send(Bytes::from(data)).await {
                    Ok(_) => {}
                    Err(e) => {
                        error!("{:?}", e)
                    }
                }
                self.channel_sender.send(vec![]).await?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Sender {
    sender_size: usize,
    receiver_size: usize,
    b: Vec<Block>,
    d: Block, // vole delta
    bin_size: usize,
    ssp: usize,
    prng: Prng,
    baxos: Option<Baxos>,
    channel_recv: MpscReceiver<Vec<Block>>,
    cross_sender: CrossSender<Vec<u8>>,
    port: u16,
}

impl Sender {
    pub fn new(port: u16, seed: Block) -> Sender {
        let (channel_send, channel_recv) = channel(1);
        let (corss_send, cross_recv) = bounded(1);
        let handler = SenderHandler {
            channel_sender: channel_send,
            cross_receiver: cross_recv,
        };
        tokio::spawn(async move {
            NetworkReceiver::spawn(
                SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port),
                handler,
            );
        });
        Sender {
            sender_size: 0,
            receiver_size: 0,
            b: vec![],
            d: *ZERO_BLOCK,
            bin_size: 1 << 14,
            ssp: 40,
            prng: Prng::new(seed),
            baxos: None,
            channel_recv: channel_recv,
            cross_sender: corss_send,
            port: port,
        }
    }

    pub async fn run(&mut self, inputs: &[Block]) {
        let vole_address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), self.port + 1);

        self.receiver_size = inputs.len();
        self.sender_size = self.receiver_size;

        let mask_size: usize = ((self.ssp as f64
            + ((self.sender_size * self.receiver_size) as f64)
                .log2()
                .ceil())
            / 8f64)
            .ceil() as usize;
        let compress = mask_size != 16;

        // send
        let mut baxos = Baxos::new(self.receiver_size, self.bin_size, 3, self.ssp, *ZERO_BLOCK);
        
        let baxos_size = baxos.size();
        self.d = self.prng.get_block();
        let delta: [u64; 2] = [self.d.get(0), self.d.get(1)];
        let handler = thread::spawn(move || {
            let mut rng = thread_rng();
            // thread code
            let vole_sender = SilentVoleSender::new(vole_address, &mut rng);
            vole_sender.init(delta, baxos_size);
            vole_sender.get_b(baxos_size)
        });

        // receive paxos seed
        let seed = self.channel_recv.recv().await.unwrap();
        info!("receiver seed");
        debug_assert_eq!(seed.len(), 1);
        baxos.seed = seed[0];
        self.baxos = Some(baxos);

        // wait vole
        self.b = handler.join().unwrap();
        info!("vole finish");
        let pp: Vec<Block> = self.channel_recv.recv().await.unwrap();
        debug_assert_eq!(pp.len(), baxos_size);
        info!("receive pp");
        // receive from reveiver with pp

        debug_assert_eq!(self.b.len(), pp.len());

        cfg_iter_mut!(self.b).zip(&pp).for_each(|(b, p)| {
            *b = *b ^ self.d.gf128_mul_reduce(p);
        });

        // send finish

        let mut hashes = vec![*ZERO_BLOCK; inputs.len()];

        self.eval(inputs, &mut hashes);
        info!("eval hashes");
        // mask_size = 16;
        let mut res = vec![0u8; hashes.len() * mask_size];
        if compress {
            unsafe {
                let mut src = hashes.as_ptr();
                let mut dest: *mut u8 = res.as_mut_ptr();
                for _i in 0..self.sender_size {
                    memcpy(dest as *mut c_void, src as *const c_void, mask_size);
                    dest = dest.add(mask_size);
                    src = src.add(1);
                }
            }
        } else {
            panic!("compress error");
        }
        // sender to receivers
        // network sender
        assert_eq!(self.channel_recv.recv().await.unwrap().len(), 0);
        self.cross_sender.send(res).unwrap();
        info!("{}", self.channel_recv.recv().await.unwrap().len());
        // network send hashes
    }

    pub fn eval(&self, val: &[Block], output: &mut [Block]) {
        self.baxos.as_ref().unwrap().decode(val, output, &self.b, 1);

        let mut h = vec![*ZERO_BLOCK; val.len()];
        unsafe {
            fixed_key_hash_blocks(
                val.as_ptr() as *const c_void,
                val.len(),
                h.as_mut_ptr() as *mut c_void,
            );
        }
        let delta = self.d;
        cfg_iter_mut!(output).zip(&h).for_each(|(o, h)| {
            *o = *o ^ delta.gf128_mul_reduce(h);
            unsafe {
                let mut c = *ZERO_BLOCK;
                fixed_key_hash_blocks(
                    o as *const Block as *const c_void,
                    1,
                    &mut c as *mut Block as *mut c_void,
                );
                *o = c;
            }
        });
    }
}
