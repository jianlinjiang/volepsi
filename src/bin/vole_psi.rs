#[macro_use]
extern crate ark_std;
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use clap::Parser;
use csv::Reader;
use env_logger::Env;
use log::info;
use rand::thread_rng;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};
use volepsi::app::data::Id;
use volepsi::vole::block::Block;
use volepsi::vole::psi_reveiver::Receiver;
use volepsi::vole::psi_sender::Sender;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
pub const PSI_RES_PATH: &str = "data/PSI_RES.csv";
pub const PSI_ID_NUM: usize = 1000_000;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Ip for bob
    #[arg(long)]
    ip: String,

    /// port for bob
    #[arg(short, long)]
    port: u16,

    /// file to process
    #[arg(short, long)]
    file: String,

    /// roles, role 0 holds the database, role 1 executes the query
    #[arg(short, long)]
    role: i32,
}

enum Role {
    Sender = 0,
    Receiver,
}

impl TryFrom<i32> for Role {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            x if x == Role::Sender as i32 => Ok(Role::Sender),
            x if x == Role::Receiver as i32 => Ok(Role::Receiver),
            _ => Err(()),
        }
    }
}

#[tokio::main]
async fn main() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let role: Role =
        Role::try_from(args.role).expect("can't to parse role! Only 0 and 1 are valid. Role 0 is the sender and role 1 is the receiver.");
    let file = File::options()
        .create(true)
        .write(true)
        .open(PSI_RES_PATH)
        .unwrap();

    match role {
        Role::Sender => {
            info!("Run as sender");
            let mut reader = Reader::from_path(&args.file).unwrap();
            let mut iter = reader.deserialize::<Id>();
            let mut ids: Vec<usize> = Vec::with_capacity(PSI_ID_NUM);
            while let Some(result) = iter.next() {
                let id = result.unwrap().id;
                ids.push(id);
            }

            let mut rng = thread_rng();
            let mut sender = Sender::new(args.port, Block::rand(&mut rng));

            let inputs: Vec<Block> = cfg_iter!(ids)
                .map(|&id| {
                    let mut hasher = Blake2bVar::new(16).unwrap();
                    hasher.update(&id.to_le_bytes());
                    let mut buf = [0u8; 16];
                    hasher.finalize_variable(&mut buf).unwrap();
                    Block::copy_from_u8(buf)
                })
                .collect();
            sender.run(&inputs).await;
        }

        Role::Receiver => {
            info!("Run as receiver");
            let mut file = BufWriter::new(file);
            file.write(String::from("id\n").as_bytes()).unwrap();
            let mut reader = Reader::from_path(&args.file).unwrap();
            let mut iter = reader.deserialize::<Id>();
            let mut ids: Vec<usize> = Vec::with_capacity(PSI_ID_NUM);
            while let Some(result) = iter.next() {
                let id = result.unwrap().id;
                ids.push(id);
            }
            let inputs: Vec<Block> = cfg_iter!(ids)
                .map(|&id| {
                    let mut hasher = Blake2bVar::new(16).unwrap();
                    let mut buf = [0u8; 16];
                    hasher.update(&id.to_le_bytes());
                    hasher.finalize_variable(&mut buf).unwrap();
                    Block::copy_from_u8(buf)
                })
                .collect();

            println!("{}", inputs.len());

            let mut rng = thread_rng();
            let mut receiver = Receiver::new(
                Block::rand(&mut rng),
                IpAddr::V4(args.ip.parse::<Ipv4Addr>().unwrap()),
                args.port,
            );

            receiver.receive(&inputs).await;
            receiver.res.iter().for_each(|&x: &usize| {
                file.write(format!("{}\n", ids[x]).as_bytes()).unwrap();
            });
        }
    }
}
