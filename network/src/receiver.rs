// Copyright(C) Facebook, Inc. and its affiliates.
use crate::error::NetworkError;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::SplitSink;
use futures::stream::StreamExt as _;
use log::{debug, warn};
use rustls_pemfile::{certs, ec_private_keys};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::rustls::{self, Certificate, PrivateKey};
use tokio_rustls::{server::TlsStream, TlsAcceptor};
use tokio_util::codec::{Framed, LengthDelimitedCodec};
/// Convenient alias for the writer end of the TCP channel.
pub type Writer = SplitSink<Framed<TlsStream<TcpStream>, LengthDelimitedCodec>, Bytes>;

pub fn load_certs(path: &Path) -> io::Result<Vec<Certificate>> {
    certs(&mut BufReader::new(File::open(path)?))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid cert"))
        .map(|mut certs| certs.drain(..).map(Certificate).collect())
}

pub fn load_keys(path: &Path) -> io::Result<Vec<PrivateKey>> {
    ec_private_keys(&mut BufReader::new(File::open(path)?))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid key"))
        .map(|mut keys| keys.drain(..).map(PrivateKey).collect())
}

#[async_trait]
pub trait MessageHandler: Clone + Send + Sync + 'static {
    /// Defines how to handle an incoming message. A typical usage is to define a `MessageHandler` with a
    /// number of `Sender<T>` channels. Then implement `dispatch` to deserialize incoming messages and
    /// forward them through the appropriate delivery channel. Then `writer` can be used to send back
    /// responses or acknowledgements to the sender machine (see unit tests for examples).
    async fn dispatch(&self, writer: &mut Writer, message: Bytes) -> Result<(), Box<dyn Error>>;
}

/// For each incoming request, we spawn a new runner responsible to receive messages and forward them
/// through the provided deliver channel.
pub struct Receiver<Handler: MessageHandler> {
    /// Address to listen to.
    address: SocketAddr,
    /// Struct responsible to define how to handle received messages.
    handler: Handler,
}

impl<Handler: MessageHandler> Receiver<Handler> {
    /// Spawn a new network receiver handling connections from any incoming peer.
    pub fn spawn(address: SocketAddr, handler: Handler) {
        tokio::spawn(async move {
            Self { address, handler }.run().await;
        });
    }

    /// Main loop responsible to accept incoming connections and spawn a new runner to handle it.
    async fn run(&self) {
        let listener = TcpListener::bind(&self.address)
            .await
            .expect("Failed to bind TCP port");
        let certs = load_certs(Path::new("pems/server.crt")).unwrap();
        let mut keys = load_keys(Path::new("pems/server.key")).unwrap();
        let config = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, keys.remove(0))
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))
            .unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(config));
        debug!("Listening on {}", self.address);
        loop {
            let (socket, peer) = match listener.accept().await {
                Ok(value) => value,
                Err(e) => {
                    warn!("{}", NetworkError::FailedToListen(e));
                    continue;
                }
            };
            let acceptor = acceptor.clone();
            let stream = acceptor.accept(socket).await.unwrap();
            // let (io_stream, _connection) = stream.into_inner();
            // info!("Incoming connection established with {}", peer);
            Self::spawn_runner(stream, peer, self.handler.clone()).await;
        }
    }

    /// Spawn a new runner to handle a specific TCP connection. It receives messages and process them
    /// using the provided handler.
    async fn spawn_runner(socket: TlsStream<TcpStream>, peer: SocketAddr, handler: Handler) {
        tokio::spawn(async move {
            let mut codec = LengthDelimitedCodec::new();
            codec.set_max_frame_length(24 * 1024 * 1024);
            let transport = Framed::new(socket, codec);
            let (mut writer, mut reader) = transport.split();
            while let Some(frame) = reader.next().await {
                match frame.map_err(|e| NetworkError::FailedToReceiveMessage(peer, e)) {
                    Ok(message) => {
                        if let Err(e) = handler.dispatch(&mut writer, message.freeze()).await {
                            debug!("{}", e);
                            return;
                        }
                    }
                    Err(e) => {
                        debug!("{}", e);
                        return;
                    }
                }
            }
            // warn!("Connection closed by peer {}", peer);
        });
    }
}
